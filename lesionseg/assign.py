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
        labels = [f"<{edges[1]:g}"] + [f"{a:g}..{b:g}" for a, b in zip(edges[1:-2], edges[2:-1])] + [f">{edges[-2]:g}"]
        cells["dist_bin"] = pd.cut(cells["dist_to_lesion_um"], bins=edges, labels=labels)
    return cells


def zone_summary(cells: pd.DataFrame, pos_col: str = "pu1_pos") -> pd.DataFrame:
    """Cells / Pu.1+ cells per zone (and per lesion)."""
    g = cells.groupby(["zone"], observed=True)
    out = pd.DataFrame({"n_cells": g.size(), f"n_{pos_col}": g[pos_col].sum()})
    out[f"frac_{pos_col}"] = out[f"n_{pos_col}"] / out["n_cells"]
    return out.reset_index()
