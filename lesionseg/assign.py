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
                                   categories=["background", "distal", "peri", "rim", "core", "deep"])
    cells["lesion_id"] = lid
    cells["dist_to_lesion_um"] = dist.astype(np.float32)

    if distance_bins_um:
        edges = [-np.inf] + sorted(distance_bins_um) + [np.inf]
        inner = [f"{a:g}..{b:g}" for a, b in zip(edges[1:-2], edges[2:-1], strict=True)]
        labels = [f"<{edges[1]:g}"] + inner + [f">{edges[-2]:g}"]
        cells["dist_bin"] = pd.cut(cells["dist_to_lesion_um"], bins=edges, labels=labels)
    return cells


def label_sections(tissue: np.ndarray, bin_um: float, *, min_area_um2: float = 2e5,
                   merge_um: float = 0.0, split_touching: bool = True, neck_depth_um: float = 200.0,
                   **_) -> np.ndarray:
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
    # distance to the edge of the own section (µm) – used for edge exclusion when picking LMD cells
    edt = ndi.distance_transform_edt(sections > 0) * grid.bin_um
    cells["dist_to_section_edge_um"] = ndi.map_coordinates(edt, coords, order=1, mode="nearest").astype(np.float32)
    lesions = lesions.copy()
    if len(lesions):
        lx, ly = grid.to_grid(lesions["centroid_x_um"].to_numpy(), lesions["centroid_y_um"].to_numpy())
        lc = np.vstack([np.clip(ly, 0, rows - 1), np.clip(lx, 0, cols - 1)])
        lesions["section_id"] = ndi.map_coordinates(sections, lc, order=0, mode="nearest").astype(int)
    return cells, lesions


def section_geometry(sections: np.ndarray, grid: Grid) -> dict[str, np.ndarray]:
    """Per-bin section geometry, size-normalised.

    * ``radius_um``  – equivalent radius sqrt(area/π) of the bin's section
    * ``d_edge_um``  – distance to the section boundary
    * ``d_center_um``– distance to the section centroid (≈ central canal in spinal cord)
    * ``rel_pos``    – d_center / (d_center + d_edge): 0 at the centre, 1 at the pial edge,
                       independent of section size and shape
    """
    n = int(sections.max())
    shape = sections.shape
    radius = np.zeros(shape, np.float32)
    d_center = np.zeros(shape, np.float32)
    d_edge = (ndi.distance_transform_edt(sections > 0) * grid.bin_um).astype(np.float32)
    if n > 0:
        ids = np.arange(1, n + 1)
        area = np.bincount(sections.ravel(), minlength=n + 1)[1:]
        cm = ndi.center_of_mass(sections > 0, sections, ids)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        for sid, (cy, cx), a in zip(ids, cm, area, strict=True):
            m = sections == sid
            radius[m] = np.sqrt(a / np.pi) * grid.bin_um
            d_center[m] = (np.hypot(xx[m] - cx, yy[m] - cy) * grid.bin_um).astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.where(sections > 0, d_center / np.maximum(d_center + d_edge, 1e-6), 0.0).astype(np.float32)
    return {"radius_um": radius, "d_edge_um": d_edge, "d_center_um": d_center, "rel_pos": rel}


def add_relative_geometry(cells: pd.DataFrame, geom: dict[str, np.ndarray], grid: Grid) -> pd.DataFrame:
    """Per-cell ``section_radius_um``, ``rel_pos`` (0 centre → 1 edge), ``dist_to_edge_rel`` and
    ``dist_to_lesion_rel`` (signed lesion distance / section radius)."""
    cells = cells.copy()
    gx, gy = grid.to_grid(cells["x_um"].to_numpy(), cells["y_um"].to_numpy())
    rows, cols = grid.shape
    coords = np.vstack([np.clip(gy, 0, rows - 1), np.clip(gx, 0, cols - 1)])
    cells["section_radius_um"] = ndi.map_coordinates(geom["radius_um"], coords, order=0, mode="nearest")
    cells["rel_pos"] = ndi.map_coordinates(geom["rel_pos"], coords, order=1, mode="nearest")
    r = cells["section_radius_um"].to_numpy(float)
    r = np.where(r > 0, r, np.nan)  # cells outside any section: relative quantities undefined
    if "dist_to_section_edge_um" in cells:
        cells["dist_to_edge_rel"] = cells["dist_to_section_edge_um"].to_numpy(float) / r
    if "dist_to_lesion_um" in cells:
        cells["dist_to_lesion_rel"] = cells["dist_to_lesion_um"].to_numpy(float) / r
    return cells


def add_signed_distance(cells: pd.DataFrame, mask: np.ndarray, grid: Grid, col: str) -> pd.DataFrame:
    """Sample the signed distance (µm, negative inside ``mask``) at each cell centroid."""
    cells = cells.copy()
    if not mask.any():
        cells[col] = np.inf
        return cells
    sd = np.where(mask, -ndi.distance_transform_edt(mask), ndi.distance_transform_edt(~mask)) * grid.bin_um
    gx, gy = grid.to_grid(cells["x_um"].to_numpy(), cells["y_um"].to_numpy())
    rows, cols = grid.shape
    coords = np.vstack([np.clip(gy, 0, rows - 1), np.clip(gx, 0, cols - 1)])
    cells[col] = ndi.map_coordinates(sd.astype(np.float32), coords, order=1, mode="nearest")
    return cells


def section_centroids_um(sections: np.ndarray, bin_um: float) -> dict[int, tuple[float, float]]:
    n = int(sections.max())
    if n == 0:
        return {}
    cm = ndi.center_of_mass(sections > 0, sections, np.arange(1, n + 1))
    return {i + 1: (cx * bin_um, cy * bin_um) for i, (cy, cx) in enumerate(cm)}


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
    for z in ("core", "rim", "peri", "deep"):
        zc = cells[(cells["zone"] == z) & (cells["section_id"] > 0)].groupby("section_id")[pos_col].sum()
        out[f"n_{pos_col}_{z}"] = zc.reindex(out.section_id).fillna(0).astype(int).to_numpy()
    return out


def zone_summary(cells: pd.DataFrame, pos_col: str = "pu1_pos") -> pd.DataFrame:
    """Cells / Pu.1+ cells per zone (and per lesion)."""
    g = cells.groupby(["zone"], observed=True)
    out = pd.DataFrame({"n_cells": g.size(), f"n_{pos_col}": g[pos_col].sum()})
    out[f"frac_{pos_col}"] = out[f"n_{pos_col}"] / out["n_cells"]
    return out.reset_index()
