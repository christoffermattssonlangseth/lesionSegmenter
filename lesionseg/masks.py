"""Build the cell table from **pre-computed segmentation masks**.

This is the primary input route: a label image with every cell (nuclei or whole
cells) and a second label image with only the Pu.1+ cells (e.g. exported from
BIAS / Cellpose / QuPath). Both must be in the same full-resolution pixel frame
as the scan (and as the LMD calibration points).

Accepted formats: ``.tif/.tiff`` (single-page label image, any int dtype),
``.png`` (16-bit), ``.npy``, and Cellpose ``*_seg.npy`` (uses ``['masks']``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure
from tqdm import tqdm

from .io import SlideReader
from .segment import _contour_wkt


# --------------------------------------------------------------------------
def load_label_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".npy":
        arr = np.load(path, allow_pickle=True)
        if arr.dtype == object:  # cellpose *_seg.npy
            arr = arr.item()["masks"]
    elif suf in {".tif", ".tiff"}:
        import tifffile

        arr = tifffile.imread(path)
    elif suf == ".png":
        import imageio.v3 as iio

        arr = iio.imread(path)
    else:
        raise ValueError(f"unsupported mask format: {path}")
    arr = np.squeeze(arr)
    if arr.ndim == 3:  # e.g. RGB-encoded label or (1, Y, X)
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] in (3, 4):
            r, g, b = (arr[..., i].astype(np.int64) for i in range(3))
            arr = r + (g << 8) + (b << 16)
    if arr.ndim != 2:
        raise ValueError(f"label image must be 2-D, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.int32)
    return arr


def relabel_if_binary(labels: np.ndarray) -> np.ndarray:
    """A binary (0/1 or 0/255) mask is turned into connected-component labels."""
    u = np.unique(labels[::50, ::50])
    if len(u) <= 2 and labels.max() in (1, 255):
        lab, _ = ndi.label(labels > 0)
        return lab.astype(np.int32)
    return labels


# --------------------------------------------------------------------------
def cells_from_labels(labels: np.ndarray, pixel_size_um: float, *, contours: bool = True,
                      contour_only_for: np.ndarray | None = None, progress: bool = True,
                      offset_px: tuple[int, int] = (0, 0)) -> pd.DataFrame:
    """Per-object table (centroid, area, shape, contour) from a label image.

    ``contour_only_for``: boolean array aligned with the returned rows (or list
    of label ids) restricting contour extraction to a subset (e.g. Pu.1+ cells)
    – computing WKT for 500k nuclei takes minutes, for 20k a few seconds.
    ``offset_px`` = (x, y) shift if the mask covers only part of the scene.
    """
    props = measure.regionprops_table(labels, properties=("label", "centroid", "area", "bbox", "eccentricity",
                                                          "solidity"))
    df = pd.DataFrame(props)
    ox, oy = offset_px
    out = pd.DataFrame({
        "label": df["label"].to_numpy(),
        "x_px": df["centroid-1"].to_numpy() + ox,
        "y_px": df["centroid-0"].to_numpy() + oy,
    })
    out["x_um"] = out["x_px"] * pixel_size_um
    out["y_um"] = out["y_px"] * pixel_size_um
    out["area_um2"] = df["area"].to_numpy() * pixel_size_um ** 2
    out["eccentricity"] = df["eccentricity"].to_numpy()
    out["solidity"] = df["solidity"].to_numpy()
    if contours:
        bb = df[["bbox-0", "bbox-1", "bbox-2", "bbox-3"]].to_numpy().astype(int)
        labs = out["label"].to_numpy()
        if contour_only_for is None:
            sel = np.ones(len(labs), bool)
        elif np.asarray(contour_only_for).dtype == bool:
            sel = np.asarray(contour_only_for)
        else:
            sel = np.isin(labs, np.asarray(contour_only_for))
        wkts = np.full(len(labs), None, dtype=object)
        it = tqdm(np.nonzero(sel)[0], desc="contours", disable=not progress)
        for i in it:
            r0, c0, r1, c1 = bb[i]
            sub = labels[r0:r1, c0:c1] == labs[i]
            wkts[i] = _contour_wkt(sub, c0 + ox, r0 + oy, pixel_size_um)
        out["contour_wkt"] = wkts
    return out


def match_pu1_mask(cell_labels: np.ndarray, pu1_labels: np.ndarray, cells: pd.DataFrame, *,
                   min_overlap: float = 0.3) -> tuple[pd.DataFrame, dict]:
    """Flag cells that are covered by an object in the Pu.1+ mask.

    A Pu.1+ object is assigned to the cell label with which it shares most pixels,
    provided that overlap is ≥ ``min_overlap`` of the Pu.1+ object area. Pu.1+
    objects that hit no cell are appended as extra rows (``source='pu1_mask'``)
    so that no Pu.1+ cell is lost for DVP.
    """
    if cell_labels.shape != pu1_labels.shape:
        raise ValueError(f"mask shapes differ: cells {cell_labels.shape} vs pu1 {pu1_labels.shape}")
    sel = pu1_labels > 0
    p = pu1_labels[sel].astype(np.int64)
    c = cell_labels[sel].astype(np.int64)
    n_pu1_px = np.bincount(p)
    key = p * (int(cell_labels.max()) + 1) + c
    uk, cnt = np.unique(key, return_counts=True)
    kp = uk // (int(cell_labels.max()) + 1)
    kc = uk % (int(cell_labels.max()) + 1)
    # best cell per pu1 object (excluding background 0)
    fg = kc > 0
    best: dict[int, tuple[int, int]] = {}
    for pl, cl, n in zip(kp[fg], kc[fg], cnt[fg], strict=True):
        if pl not in best or n > best[pl][1]:
            best[pl] = (cl, n)
    pu1_ids = np.unique(p)
    matched_cells = set()
    unmatched = []
    for pl in pu1_ids:
        cl, n = best.get(pl, (0, 0))
        if cl > 0 and n / n_pu1_px[pl] >= min_overlap:
            matched_cells.add(int(cl))
        else:
            unmatched.append(int(pl))

    cells = cells.copy()
    cells["pu1_pos"] = cells["label"].isin(matched_cells).to_numpy()
    cells["source"] = "cells_mask"
    info = {"n_pu1_objects": len(pu1_ids), "n_matched_cells": len(matched_cells),
            "n_unmatched_pu1": len(unmatched), "min_overlap": min_overlap}
    if unmatched:
        extra = cells_from_labels(np.where(np.isin(pu1_labels, unmatched), pu1_labels, 0), 1.0,
                                  contours=False, progress=False)
        extra = extra.rename(columns={"label": "pu1_label"})
        extra["label"] = -extra["pu1_label"]  # negative = came from the Pu.1 mask only
        extra["pu1_pos"] = True
        extra["source"] = "pu1_mask"
        cells = pd.concat([cells, extra], ignore_index=True)
    return cells, info


# --------------------------------------------------------------------------
def measure_intensities(labels: np.ndarray, reader: SlideReader, scene: int, channels: list[int],
                        cells: pd.DataFrame, *, tile: int = 4096, annulus_px: int = 3,
                        progress: bool = True) -> pd.DataFrame:
    """Exact per-label mean (+ local annulus background) of image channels, tile by tile.

    Sums and pixel counts are accumulated over tiles, so objects crossing a tile
    edge are measured correctly.
    """
    from skimage.segmentation import expand_labels

    max_lab = int(labels.max()) + 1
    names = [reader.channel_names[c] for c in channels]
    sums = np.zeros((len(channels), max_lab), np.float64)
    cnts = np.zeros(max_lab, np.float64)
    bsums = np.zeros((len(channels), max_lab), np.float64)
    bcnts = np.zeros(max_lab, np.float64)
    H, W = labels.shape
    pad = annulus_px + 1
    tiles = [(y, x) for y in range(0, H, tile) for x in range(0, W, tile)]
    for y, x in tqdm(tiles, desc="intensities", disable=not progress):
        y0, x0 = max(y - pad, 0), max(x - pad, 0)
        y1, x1 = min(y + tile + pad, H), min(x + tile + pad, W)
        lab = labels[y0:y1, x0:x1]
        if lab.max() == 0:
            continue
        img = reader.read_region_stack(scene, x0, y0, x1 - x0, y1 - y0, channels).astype(np.float32)
        # inner box (without pad) for object pixels; annulus uses padded context
        iy0, ix0 = y - y0, x - x0
        iy1, ix1 = iy0 + min(tile, H - y), ix0 + min(tile, W - x)
        core_lab = lab[iy0:iy1, ix0:ix1]
        idx = core_lab.ravel()
        cnts += np.bincount(idx, minlength=max_lab)
        for k in range(len(channels)):
            sums[k] += np.bincount(idx, weights=img[k, iy0:iy1, ix0:ix1].ravel(), minlength=max_lab)
        ann = expand_labels(lab, annulus_px)
        ann[lab > 0] = 0
        ann_core = ann[iy0:iy1, ix0:ix1].ravel()
        bcnts += np.bincount(ann_core, minlength=max_lab)
        for k in range(len(channels)):
            bsums[k] += np.bincount(ann_core, weights=img[k, iy0:iy1, ix0:ix1].ravel(), minlength=max_lab)
    cells = cells.copy()
    lab_idx = cells["label"].to_numpy()
    valid = lab_idx > 0
    li = np.where(valid, lab_idx, 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        for k, nm in enumerate(names):
            m = np.where(cnts[li] > 0, sums[k][li] / cnts[li], np.nan)
            b = np.where(bcnts[li] > 0, bsums[k][li] / bcnts[li], np.nan)
            cells[f"{nm}_mean"] = np.where(valid, m, np.nan)
            cells[f"{nm}_bg"] = np.where(valid, b, np.nan)
    return cells


def cells_from_masks(cells_mask: str | Path, pu1_mask: str | Path | None, pixel_size_um: float, *,
                     contours: str = "pu1", min_overlap: float = 0.3, reader: SlideReader | None = None,
                     scene: int = 0, measure_channels: list[int] | None = None,
                     progress: bool = True) -> tuple[pd.DataFrame, dict]:
    """Full mask route: load → table → Pu.1 match → (optional) intensities.

    ``contours``: ``'pu1'`` (default; only Pu.1+ cells get WKT contours), ``'all'`` or ``'none'``.
    """
    labels = relabel_if_binary(load_label_image(cells_mask))
    info = {"cells_mask": str(cells_mask), "shape": list(labels.shape), "n_labels": int(len(np.unique(labels)) - 1)}
    cells = cells_from_labels(labels, pixel_size_um, contours=False, progress=progress)
    if pu1_mask is not None:
        pu1 = relabel_if_binary(load_label_image(pu1_mask))
        cells, minfo = match_pu1_mask(labels, pu1, cells, min_overlap=min_overlap)
        info["pu1_match"] = minfo
        # extra rows from the Pu.1 mask were measured in px – convert to µm and give them contours
        extra = cells["source"] == "pu1_mask"
        if extra.any():
            cells.loc[extra, ["x_um", "y_um"]] = cells.loc[extra, ["x_px", "y_px"]].to_numpy() * pixel_size_um
            cells.loc[extra, "area_um2"] = cells.loc[extra, "area_um2"] * pixel_size_um ** 2
    else:
        cells["pu1_pos"] = False
        cells["source"] = "cells_mask"

    if contours != "none":
        sel_cells = cells["source"] == "cells_mask"
        want = cells["pu1_pos"] if contours == "pu1" else pd.Series(True, index=cells.index)
        tab = cells_from_labels(labels, pixel_size_um, contours=True,
                                contour_only_for=cells.loc[sel_cells, "label"][want[sel_cells]].to_numpy(),
                                progress=progress)
        wk = dict(zip(tab["label"], tab["contour_wkt"], strict=True))
        cells["contour_wkt"] = [wk.get(lab) for lab in cells["label"]]
        if pu1_mask is not None and (cells["source"] == "pu1_mask").any():
            ex = cells["source"] == "pu1_mask"
            tab2 = cells_from_labels(np.where(np.isin(pu1, (-cells.loc[ex, "label"]).to_numpy()), pu1, 0),
                                     pixel_size_um, contours=True, progress=False)
            wk2 = dict(zip(-tab2["label"], tab2["contour_wkt"], strict=True))
            cells.loc[ex, "contour_wkt"] = [wk2.get(lab) for lab in cells.loc[ex, "label"]]

    if reader is not None and measure_channels:
        cells = measure_intensities(labels, reader, scene, measure_channels, cells, progress=progress)
    cells.insert(0, "cell_id", np.arange(len(cells)))
    cells["scene"] = scene
    return cells, info
