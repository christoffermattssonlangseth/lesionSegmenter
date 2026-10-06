"""Adaptive meninges boundary: meninges | buffer | parenchyma, placed per location.

A fixed band (``parenchyma.erode_um``) is too wide where the meninges are a single cell layer and too
narrow where they swell into a dense infiltrate next to lesions (40–80 µm). Here the boundary is
found on a fine raster (``scale`` of the scan, ~1.3 µm/px):

* **surface**    – outline through the outermost nuclei (cell centroids dilated by a nucleus radius,
  gaps closed) inside the section outlines; deeper than ``edge_um`` inside the grid surface everything
  is tissue (the cell-level outline breaks up in loose tissue);
* **meninges**   – the outer ``seed_um`` layer plus every patch of *compact* tissue (nuclear area
  fraction above the ``compact_q`` percentile of deep parenchyma: nuclei touching) connected to that
  layer within ``max_um`` of the surface. Dense lesion tissue that does not touch the surface layer is
  left alone. Thin flaps (opening with ``open_um``) are meninges too;
* **buffer**     – tissue within ``gap_um`` of the meninges, plus every cell within ``cell_gap_um`` of a
  meningeal cell: collected in **no** reaction, so meninges and lesion pools never touch;
* **parenchyma** – the rest.

**Manual lesion annotations are protected** (``protect``): the collaborator's CORE polygons are lesion by
definition, so nothing inside them is ever meninges or buffer, meninges cannot grow through them, and
where they meet the meninges the buffer is carved from the meningeal side. Calibration on the 4 scenes
(manual-core cells that would otherwise become meninges): seed layer + flaps only 0.9 %; compact growth at
the 99th percentile 23 %, at the 99.9th 7 % – nuclear packing alone also catches lymphocyte-rich
subpial lesion, hence the strict default ``compact_q = 99.9``.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

TIER_CODES = {"outside": 0, "parenchyma": 1, "buffer": 2, "meninges": 3}
DEFAULTS = {"scale": 0.25, "cell_r_um": 5.0, "close_um": 30.0, "edge_um": 20.0, "seed_um": 10.0, "max_um": 80.0,
            "compact_sigma_um": 8.0, "compact_q": 99.9, "gap_um": 10.0, "cell_gap_um": 6.0, "open_um": 150.0}


def _edt(m):
    return ndi.distance_transform_edt(m)


def _norm(a):
    v = a[a > 0]
    lo, hi = np.percentile(v, [1, 99.5]) if v.size else (0.0, 1.0)
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def _upsample_index(n_fine: int, res: float, bin_um: float, n_grid: int) -> np.ndarray:
    return (np.arange(n_fine) * res / bin_um).astype(int).clip(0, n_grid - 1)


def surface_layers(cells: pd.DataFrame, sections: np.ndarray, surface: np.ndarray, bin_um: float,
                   scene_shape_px: tuple[int, int], pixel_size_um: float, nuclear_overview: np.ndarray,
                   protect_shapes=None, **params) -> dict:
    """Fine-raster layers for one scene.

    ``sections`` / ``surface`` are the grid maps (``bin_um``); ``nuclear_overview`` is the nuclear
    channel read at ``params['scale']`` of the scene; ``protect_shapes`` are polygons in scene px (manual
    lesion cores) that must stay parenchyma. Returns ``{"res", "tissue", "depth", "meninges",
    "buffer", "protect", "tiers", "compact_threshold"}`` with ``tiers`` coded by :data:`TIER_CODES`.
    """
    p = {**DEFAULTS, **params}
    H = math.ceil(scene_shape_px[0] * p["scale"])
    W = math.ceil(scene_shape_px[1] * p["scale"])
    res = pixel_size_um / p["scale"]
    rows = _upsample_index(H, res, bin_um, sections.shape[0])
    cols = _upsample_index(W, res, bin_um, sections.shape[1])
    m = np.zeros((H, W), bool)
    m[(cells["y_um"] / res).astype(int).clip(0, H - 1), (cells["x_um"] / res).astype(int).clip(0, W - 1)] = True
    m = _edt(~m) * res <= p["cell_r_um"]
    m = _edt(_edt(~m) * res <= p["close_um"]) * res > p["close_um"]       # closing via distance transforms
    m |= _edt(surface.astype(bool)[rows][:, cols]) * res > p["edge_um"]
    m = ndi.binary_fill_holes(m)
    m &= sections[rows][:, cols] > 0
    depth = (_edt(m) * res).astype(np.float32)
    flap = m & ~(_edt(~(depth > p["open_um"])) * res <= p["open_um"])
    nuc = _norm(nuclear_overview)[:H, :W]
    if nuc.shape != (H, W):
        nuc = np.pad(nuc, ((0, H - nuc.shape[0]), (0, W - nuc.shape[1])))
    from skimage.filters import threshold_otsu

    nuc_mask = nuc > threshold_otsu(nuc[m]) if m.any() else np.zeros_like(m)
    frac = ndi.gaussian_filter(nuc_mask.astype(np.float32), p["compact_sigma_um"] / res)
    deep = m & (depth > 200)
    thr = float(np.percentile(frac[deep], p["compact_q"])) if deep.any() else 1.0
    protect = np.zeros((H, W), bool)
    if protect_shapes is not None and len(protect_shapes):
        from .density import Grid
        from .validate import rasterize

        protect = rasterize(protect_shapes, Grid(res, (H, W), pixel_size_um), pixel_size_um) & m
    compact = (frac > thr) & m & (depth <= p["max_um"]) & ~protect
    seed = m & (depth <= p["seed_um"])
    lab, _ = ndi.label(compact)
    keep = np.unique(lab[seed & compact])
    men = (np.isin(lab, keep[keep > 0]) | seed | flap) & m & ~protect
    buffer = m & ~men & ~protect & (_edt(~men) * res <= p["gap_um"])
    if protect.any():  # where meninges meet a manual core, the buffer comes out of the meninges
        near_core = men & (_edt(~protect) * res <= p["gap_um"])
        men &= ~near_core
        buffer |= near_core
    tiers = np.zeros((H, W), np.uint8)
    tiers[m] = TIER_CODES["parenchyma"]
    tiers[buffer] = TIER_CODES["buffer"]
    tiers[men] = TIER_CODES["meninges"]
    return {"res": res, "tissue": m, "depth": depth, "meninges": men, "buffer": buffer, "protect": protect,
            "tiers": tiers, "compact_threshold": thr, "params": p}


def excluded_grid(layers: dict, grid_shape: tuple[int, int], bin_um: float, frac: float = 0.5) -> np.ndarray:
    """Grid bins where at least ``frac`` of the tissue is meninges or buffer (they leave the parenchyma)."""
    res = layers["res"]
    H, W = layers["tiers"].shape
    rows = _upsample_index(H, res, bin_um, grid_shape[0]).astype(np.int32)
    cols = _upsample_index(W, res, bin_um, grid_shape[1]).astype(np.int32)
    idx = (rows[:, None] * grid_shape[1] + cols[None, :]).ravel()
    n = grid_shape[0] * grid_shape[1]
    excl = np.bincount(idx, weights=(layers["meninges"] | layers["buffer"]).ravel(), minlength=n)
    tis = np.bincount(idx, weights=layers["tissue"].ravel(), minlength=n)
    return ((excl >= frac * tis) & (tis > 0)).reshape(grid_shape)


def cell_tiers(cells: pd.DataFrame, layers: dict, cell_gap_um: float | None = None,
               protected: np.ndarray | None = None) -> pd.DataFrame:
    """``surface_tier`` (meninges | buffer | parenchyma | outside) and ``depth_um`` per cell.

    Cells on the ``protect`` raster or flagged in ``protected`` (bool per cell, e.g. inside a manual lesion
    core) are always parenchyma; meningeal cells within ``cell_gap_um`` of a protected cell become buffer instead."""
    gap = layers["params"]["cell_gap_um"] if cell_gap_um is None else cell_gap_um
    res = layers["res"]
    H, W = layers["tiers"].shape
    iy = (cells["y_um"] / res).astype(int).clip(0, H - 1).to_numpy()
    ix = (cells["x_um"] / res).astype(int).clip(0, W - 1).to_numpy()
    code = layers["tiers"][iy, ix]
    in_sec = cells["section_id"].to_numpy() > 0 if "section_id" in cells else np.ones(len(cells), bool)
    tier = np.full(len(cells), "outside", dtype=object)
    tier[in_sec] = "parenchyma"
    tier[in_sec & (code == TIER_CODES["buffer"])] = "buffer"
    tier[in_sec & ((code == TIER_CODES["meninges"]) | (code == TIER_CODES["outside"]))] = "meninges"
    prot = layers["protect"][iy, ix].copy()
    if protected is not None:  # exact polygon membership beats the raster at polygon edges
        prot |= np.asarray(protected, bool)
    prot &= in_sec
    tier[prot] = "parenchyma"
    xy = cells[["x_um", "y_um"]].to_numpy(float)
    if prot.any() and gap > 0:  # keep the gap on the meningeal side of protected cells
        m_idx = np.flatnonzero(tier == "meninges")
        if len(m_idx):
            dist, _ = cKDTree(xy[prot]).query(xy[m_idx], k=1)
            tier[m_idx[dist < gap]] = "buffer"
    men = tier == "meninges"
    if men.any() and gap > 0:
        p_idx = np.flatnonzero((tier == "parenchyma") & ~prot)
        if len(p_idx):
            dist, _ = cKDTree(xy[men]).query(xy[p_idx], k=1)
            tier[p_idx[dist < gap]] = "buffer"
    out = cells.copy()
    out["surface_tier"] = pd.Categorical(tier, categories=["outside", "parenchyma", "buffer", "meninges"])
    out["depth_um"] = np.where(code > 0, layers["depth"][iy, ix], np.nan).astype(np.float32)
    return out
