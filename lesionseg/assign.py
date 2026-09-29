"""Assign each cell to a zone, a lesion and a signed distance to the lesion edge."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import ZONE_NAMES
from .density import Grid
from .lesion import LesionResult


def assign_cells(cells: pd.DataFrame, res: LesionResult, grid: Grid, *,
                 distance_bins_um: list[float] | None = None) -> pd.DataFrame:
    """Add ``zone``, ``zone_code``, ``lesion_id``, ``dist_to_lesion_um`` (signed) and
    optional ``dist_bin`` columns.

    ``lesion_id`` is the nearest lesion for cells within ``peri`` (and inside
    lesions); 0 for distal cells.
    """
    cells = cells.copy()
    gx, gy = grid.to_grid(cells["x_um"].to_numpy(), cells["y_um"].to_numpy())
    rows, cols = grid.shape
    coords = np.vstack([np.clip(gy, 0, rows - 1), np.clip(gx, 0, cols - 1)])

    zone = ndi.map_coordinates(res.zones, coords, order=0, mode="nearest")
    sd = res.signed_distance_um
    finite = np.isfinite(sd)
    if finite.all():
        dist = ndi.map_coordinates(sd, coords, order=1, mode="nearest")
    else:
        dist = np.full(len(cells), np.inf, np.float32)

    # nearest lesion id via EDT indices on the lesion label map
    if res.lesion_labels.max() > 0:
        _, inds = ndi.distance_transform_edt(res.lesion_labels == 0, return_indices=True)
        nearest = res.lesion_labels[inds[0], inds[1]]
        lid = ndi.map_coordinates(nearest, coords, order=0, mode="nearest").astype(int)
    else:
        lid = np.zeros(len(cells), int)
    lid = np.where(zone >= 2, lid, 0)  # only peri / rim / core get a lesion id

    cells["zone_code"] = zone.astype(np.uint8)
    cells["zone"] = pd.Categorical([ZONE_NAMES[int(z)] for z in zone],
                                   categories=["background", "distal", "peri", "rim", "core"])
    cells["lesion_id"] = lid
    cells["dist_to_lesion_um"] = dist.astype(np.float32)

    if distance_bins_um:
        edges = [-np.inf] + sorted(distance_bins_um) + [np.inf]
        inner = [f"{a:g}..{b:g}" for a, b in zip(edges[1:-2], edges[2:-1], strict=True)]
        labels = [f"<{edges[1]:g}"] + inner + [f">{edges[-2]:g}"]
        cells["dist_bin"] = pd.cut(cells["dist_to_lesion_um"], bins=edges, labels=labels)
    return cells


def label_sections(tissue: np.ndarray, bin_um: float, *, min_area_um2: float = 2e5,
                   merge_um: float = 0.0, split_touching: bool = True, neck_depth_um: float = 200.0) -> np.ndarray:
    """Label separate tissue pieces (e.g. the ~10 spinal-cord cross-sections on one slide).

    ``merge_um`` closes gaps narrower than this (a section split by a tear stays
    one section). With ``split_touching`` pieces that touch are separated by a
    watershed on the distance transform: two blobs stay separate when the neck
    between them is at least ``neck_depth_um`` narrower than the blobs themselves
    (h-maxima seeds), so an irregular single section is not over-split.
    Pieces smaller than ``min_area_um2`` get section 0.
    """
    from .tissue import disk, remove_small

    r = int(round(merge_um / 2 / bin_um))
    m = ndi.binary_closing(tissue, structure=disk(r)) if r > 0 else tissue
    m = remove_small(m, int(min_area_um2 / bin_um ** 2))
    lab, n = ndi.label(m)
    lab = lab.astype(np.int32)
    if split_touching and n > 0:
        from skimage.morphology import h_maxima
        from skimage.segmentation import watershed

        edt = ndi.distance_transform_edt(m) * bin_um
        seeds = h_maxima(edt, neck_depth_um)
        markers, k = ndi.label(seeds)
        if k > n:
            ws = watershed(-edt, markers, mask=m)
            lab = ws.astype(np.int32)
            # drop tiny watershed fragments back into their neighbour
            small = remove_small(lab > 0, int(min_area_um2 / bin_um ** 2))
            lab[~small] = 0
            keep = np.unique(lab[lab > 0])
            remap = np.zeros(lab.max() + 1, np.int32)
            remap[keep] = np.arange(1, len(keep) + 1)
            lab = remap[lab]
            n = len(keep)
    # order sections top-to-bottom, left-to-right for stable ids
    if n > 1:
        cy, cx = np.array(ndi.center_of_mass(m, lab, np.arange(1, n + 1))).T
        order = np.lexsort((cx, np.round(cy / (500 / bin_um))))  # row bands of 500 µm
        remap = np.zeros(n + 1, np.int32)
        remap[np.asarray(order) + 1] = np.arange(1, n + 1)
        lab = remap[lab]
    # cells on tissue bins that were dropped as too small: nearest section
    if n > 0:
        _, inds = ndi.distance_transform_edt(lab == 0, return_indices=True)
        lab = np.where(tissue, lab[inds[0], inds[1]], 0).astype(np.int32)
    return lab


def assign_sections(cells: pd.DataFrame, lesions: pd.DataFrame, sections: np.ndarray, grid: Grid) -> tuple:
    gx, gy = grid.to_grid(cells["x_um"].to_numpy(), cells["y_um"].to_numpy())
    rows, cols = grid.shape
    coords = np.vstack([np.clip(gy, 0, rows - 1), np.clip(gx, 0, cols - 1)])
    cells = cells.copy()
    cells["section_id"] = ndi.map_coordinates(sections, coords, order=0, mode="nearest").astype(int)
    lesions = lesions.copy()
    if len(lesions):
        lx, ly = grid.to_grid(lesions["centroid_x_um"].to_numpy(), lesions["centroid_y_um"].to_numpy())
        lc = np.vstack([np.clip(ly, 0, rows - 1), np.clip(lx, 0, cols - 1)])
        lesions["section_id"] = ndi.map_coordinates(sections, lc, order=0, mode="nearest").astype(int)
    return cells, lesions


def section_summary(cells: pd.DataFrame, lesions: pd.DataFrame, sections: np.ndarray, bin_um: float,
                    pos_col: str = "pu1_pos") -> pd.DataFrame:
    """Per tissue section: area, cells, Pu.1+ cells, lesion count / area / fraction of section."""
    n = int(sections.max())
    area = np.bincount(sections.ravel(), minlength=n + 1)[1:] * bin_um ** 2 / 1e6
    g = cells[cells["section_id"] > 0].groupby("section_id")
    out = pd.DataFrame({"section_id": np.arange(1, n + 1), "area_mm2": area})
    out["n_cells"] = g.size().reindex(out.section_id).fillna(0).astype(int).to_numpy()
    out[f"n_{pos_col}"] = g[pos_col].sum().reindex(out.section_id).fillna(0).astype(int).to_numpy()
    out[f"frac_{pos_col}"] = out[f"n_{pos_col}"] / out["n_cells"].replace(0, np.nan)
    if len(lesions) and "section_id" in lesions:
        lg = lesions.groupby("section_id")
        out["n_lesions"] = lg.size().reindex(out.section_id).fillna(0).astype(int).to_numpy()
        out["lesion_area_mm2"] = lg["area_mm2"].sum().reindex(out.section_id).fillna(0).to_numpy()
    else:
        out["n_lesions"] = 0
        out["lesion_area_mm2"] = 0.0
    out["lesion_frac_of_section"] = out["lesion_area_mm2"] / out["area_mm2"]
    for z in ("core", "rim", "peri"):
        zc = cells[(cells["zone"] == z) & (cells["section_id"] > 0)].groupby("section_id")[pos_col].sum()
        out[f"n_{pos_col}_{z}"] = zc.reindex(out.section_id).fillna(0).astype(int).to_numpy()
    return out


def zone_summary(cells: pd.DataFrame, pos_col: str = "pu1_pos") -> pd.DataFrame:
    """Cells / Pu.1+ cells per zone (and per lesion)."""
    g = cells.groupby(["zone"], observed=True)
    out = pd.DataFrame({"n_cells": g.size(), f"n_{pos_col}": g[pos_col].sum()})
    out[f"frac_{pos_col}"] = out[f"n_{pos_col}"] / out["n_cells"]
    return out.reset_index()
