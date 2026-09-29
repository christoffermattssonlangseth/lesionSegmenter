"""Nucleus segmentation (per tile) + tiled whole-scene runner.

Three interchangeable segmenters:

* ``classical`` – Gaussian / background subtraction / Otsu / watershed. Fast on
  CPU, good enough for centroids + density; contours are rougher.
* ``cellpose``  – Cellpose (v3 ``nuclei`` model or v4 ``cpsam``). Best contours
  for LMD export; uses MPS on Apple Silicon when available.
* ``stardist``  – StarDist ``2D_versatile_fluo``.

Every segmenter takes a 2-D nuclear image (any dtype) and returns an int32
label image with 0 = background.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import feature, filters, measure, segmentation
from tqdm import tqdm

from .io import SlideReader, Tile
from .tissue import remove_small


# --------------------------------------------------------------------------
# segmenters
# --------------------------------------------------------------------------
def segment_classical(img: np.ndarray, pixel_size_um: float, *, nucleus_diameter_um: float = 7.0,
                      min_area_um2: float = 8.0, threshold: str | float = "otsu",
                      bg_sigma_factor: float = 5.0) -> np.ndarray:
    """Threshold + distance-transform watershed nucleus segmentation."""
    x = img.astype(np.float32)
    d_px = max(nucleus_diameter_um / pixel_size_um, 2.0)
    # background subtraction (large-scale gaussian) and light denoise
    bg = filters.gaussian(x, sigma=d_px * bg_sigma_factor, preserve_range=True)
    x = np.clip(x - bg, 0, None)
    x = filters.gaussian(x, sigma=max(d_px / 8, 0.5), preserve_range=True)
    if x.max() <= 0:
        return np.zeros(img.shape, np.int32)
    if threshold == "otsu":
        thr = filters.threshold_otsu(x)
    elif threshold == "li":
        thr = filters.threshold_li(x)
    else:
        thr = float(threshold)
    fg = x > thr
    fg = remove_small(fg, int(min_area_um2 / pixel_size_um ** 2))
    fg = ndi.binary_fill_holes(fg)
    if not fg.any():
        return np.zeros(img.shape, np.int32)
    dist = ndi.distance_transform_edt(fg)
    min_dist = max(int(d_px * 0.4), 2)
    peaks = feature.peak_local_max(dist, min_distance=min_dist, labels=measure.label(fg),
                                   exclude_border=False)
    markers = np.zeros(fg.shape, np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = segmentation.watershed(-dist, markers, mask=fg)
    return labels.astype(np.int32)


class CellposeSegmenter:
    """Lazy wrapper around cellpose (v3 or v4 API)."""

    def __init__(self, model_type: str = "nuclei", gpu: bool = True, pretrained_model: str | None = None,
                 flow_threshold: float = 0.4, cellprob_threshold: float = 0.0):
        from cellpose import models  # noqa: WPS433

        self.flow_threshold = flow_threshold
        self.cellprob_threshold = cellprob_threshold
        device = None
        try:
            import torch

            if gpu and torch.backends.mps.is_available():
                device = torch.device("mps")
        except Exception:
            pass
        kwargs = {"gpu": gpu}
        if device is not None:
            kwargs["device"] = device
        if pretrained_model:
            kwargs["pretrained_model"] = pretrained_model
        elif hasattr(models, "CellposeModel"):
            # v4 ignores model_type (cpsam); v3 accepts it
            try:
                self.model = models.CellposeModel(model_type=model_type, **kwargs)
                return
            except TypeError:
                pass
        self.model = models.CellposeModel(**kwargs)

    def __call__(self, img: np.ndarray, pixel_size_um: float, *, nucleus_diameter_um: float = 7.0,
                 **_) -> np.ndarray:
        diam = nucleus_diameter_um / pixel_size_um
        try:
            masks, *_ = self.model.eval(img, diameter=diam, channels=[0, 0],
                                        flow_threshold=self.flow_threshold,
                                        cellprob_threshold=self.cellprob_threshold)
        except TypeError:  # cellpose >= 4 dropped `channels`
            masks, *_ = self.model.eval(img, diameter=diam, flow_threshold=self.flow_threshold,
                                        cellprob_threshold=self.cellprob_threshold)
        return np.asarray(masks).astype(np.int32)


class StarDistSegmenter:
    def __init__(self, model_name: str = "2D_versatile_fluo", prob_thresh=None, nms_thresh=None):
        from stardist.models import StarDist2D

        self.model = StarDist2D.from_pretrained(model_name)
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    def __call__(self, img: np.ndarray, pixel_size_um: float, **_) -> np.ndarray:
        from csbdeep.utils import normalize

        x = normalize(img.astype(np.float32), 1, 99.8)
        labels, _ = self.model.predict_instances(x, prob_thresh=self.prob_thresh,
                                                 nms_thresh=self.nms_thresh)
        return labels.astype(np.int32)


def build_segmenter(cfg: dict) -> Callable[..., np.ndarray]:
    """Return ``fn(img, pixel_size_um, **params) -> labels`` from a config dict."""
    method = cfg.get("method", "classical")
    if method == "classical":
        return segment_classical
    if method == "cellpose":
        return CellposeSegmenter(model_type=cfg.get("model_type", "nuclei"), gpu=cfg.get("gpu", True),
                                 pretrained_model=cfg.get("pretrained_model"),
                                 flow_threshold=cfg.get("flow_threshold", 0.4),
                                 cellprob_threshold=cfg.get("cellprob_threshold", 0.0))
    if method == "stardist":
        return StarDistSegmenter(cfg.get("model_name", "2D_versatile_fluo"), cfg.get("prob_thresh"),
                                 cfg.get("nms_thresh"))
    raise ValueError(f"unknown segmentation method {method!r}")


# --------------------------------------------------------------------------
# per-tile measurement
# --------------------------------------------------------------------------
@dataclass
class TileResult:
    cells: pd.DataFrame
    labels: np.ndarray | None = None


def _contour_wkt(mask: np.ndarray, x_off: float, y_off: float, scale: float) -> str | None:
    """Outer contour of a boolean object mask as WKT POLYGON (x, y) in scaled units."""
    from shapely.geometry import Polygon

    padded = np.pad(mask, 1)
    contours = measure.find_contours(padded.astype(np.float32), 0.5)
    if not contours:
        return None
    c = max(contours, key=len)
    if len(c) < 4:
        return None
    # (row, col) -> (x, y); remove padding offset, add tile/bbox offset
    xy = np.column_stack([(c[:, 1] - 1 + x_off) * scale, (c[:, 0] - 1 + y_off) * scale])
    poly = Polygon(xy)
    if not poly.is_valid:
        poly = poly.buffer(0)
    if poly.is_empty:
        return None
    return poly.simplify(0.3 * scale).wkt


def measure_tile(labels: np.ndarray, intensity: np.ndarray, channel_names: list[str], tile: Tile,
                 pixel_size_um: float, *, contours: bool = True, annulus_px: int = 3) -> pd.DataFrame:
    """Per-object table for one tile.

    Keeps only objects whose centroid falls inside ``tile.core`` (stitching rule).
    Columns: x_px, y_px (scene px), x_um, y_um, area_um2, eccentricity, solidity,
    ``<ch>_mean``, ``<ch>_bg`` (local annulus mean, same nucleus only), and
    ``contour_wkt`` in µm.
    """
    if labels.max() == 0:
        return pd.DataFrame()
    props = measure.regionprops_table(
        labels, properties=("label", "centroid", "area", "bbox", "eccentricity", "solidity"))
    df = pd.DataFrame(props)
    cy = df["centroid-0"].to_numpy()
    cx = df["centroid-1"].to_numpy()
    gx = cx + tile.x
    gy = cy + tile.y
    x0, y0, x1, y1 = tile.core
    keep = (gx >= x0) & (gx < x1) & (gy >= y0) & (gy < y1)
    df = df.loc[keep].reset_index(drop=True)
    if df.empty:
        return df
    lab_keep = df["label"].to_numpy()

    # intensity features
    ann = segmentation.expand_labels(labels, distance=annulus_px)
    ann[labels > 0] = 0
    idx = lab_keep
    for c, name in enumerate(channel_names):
        ch = intensity[c].astype(np.float32)
        df[f"{name}_mean"] = ndi.mean(ch, labels, idx)
        df[f"{name}_bg"] = np.nan_to_num(ndi.mean(ch, ann, idx), nan=0.0)

    out = pd.DataFrame({
        "x_px": gx[keep],
        "y_px": gy[keep],
        "x_um": gx[keep] * pixel_size_um,
        "y_um": gy[keep] * pixel_size_um,
        "area_um2": df["area"].to_numpy() * pixel_size_um ** 2,
        "eccentricity": df["eccentricity"].to_numpy(),
        "solidity": df["solidity"].to_numpy(),
    })
    for name in channel_names:
        out[f"{name}_mean"] = df[f"{name}_mean"].to_numpy()
        out[f"{name}_bg"] = df[f"{name}_bg"].to_numpy()

    if contours:
        wkts = []
        bb = df[["bbox-0", "bbox-1", "bbox-2", "bbox-3"]].to_numpy().astype(int)
        for lab, (r0, c0, r1, c1) in zip(lab_keep, bb):
            sub = labels[r0:r1, c0:c1] == lab
            wkts.append(_contour_wkt(sub, c0 + tile.x, r0 + tile.y, pixel_size_um))
        out["contour_wkt"] = wkts
    out["tile"] = f"{tile.index[0]}_{tile.index[1]}"
    return out


# --------------------------------------------------------------------------
# whole-scene runner
# --------------------------------------------------------------------------
def segment_scene(reader: SlideReader, scene: int, *, nuclear_channel: int, measure_channels: Iterable[int],
                  segmenter: Callable[..., np.ndarray], seg_params: dict | None = None,
                  tile: int = 2048, overlap: int = 128, tissue_mask: np.ndarray | None = None,
                  tissue_scale: float | None = None, contours: bool = True, scale: float = 1.0,
                  progress: bool = True) -> pd.DataFrame:
    """Segment all nuclei in a scene tile by tile and return one cell table.

    ``scale`` < 1 processes a downscaled version of the scene (quick looks).
    Coordinates in the returned table are always in **full-resolution** scene px / µm.
    ``tissue_mask`` (+ its ``tissue_scale`` relative to full res) lets the runner
    skip tiles containing no tissue.
    """
    seg_params = dict(seg_params or {})
    measure_channels = list(measure_channels)
    if nuclear_channel not in measure_channels:
        measure_channels = [nuclear_channel] + measure_channels
    ch_names = [reader.channel_names[c] for c in measure_channels]
    px = reader.pixel_size_um

    if scale != 1.0:
        # downscaled path: read whole scene once at `scale`, then tile the array
        stack = reader.read_overview_stack(scene, scale, measure_channels)
        sub = ArrayReaderShim(stack, ch_names, px / scale)
        tiles = list(sub.iter_tiles(0, tile, overlap))
        read = lambda t: sub.read_region_stack(0, t.x, t.y, t.w, t.h, range(len(measure_channels)))
        eff_px = px / scale
    else:
        tiles = list(reader.iter_tiles(scene, tile, overlap))
        read = lambda t: reader.read_region_stack(scene, t.x, t.y, t.w, t.h, measure_channels)
        eff_px = px

    nuc_i = measure_channels.index(nuclear_channel)
    frames = []
    it = tqdm(tiles, desc=f"segment scene {scene}", disable=not progress)
    for t in it:
        if tissue_mask is not None and tissue_scale is not None:
            # tile box in tissue-mask px; tissue_scale is relative to full res
            f = tissue_scale / scale
            ty0, ty1 = int(t.y * f), int(np.ceil((t.y + t.h) * f))
            tx0, tx1 = int(t.x * f), int(np.ceil((t.x + t.w) * f))
            if not tissue_mask[ty0:ty1, tx0:tx1].any():
                continue
        stack = read(t)
        labels = segmenter(stack[nuc_i], eff_px, **seg_params)
        df = measure_tile(labels, stack, ch_names, t, eff_px, contours=contours)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    cells = pd.concat(frames, ignore_index=True)
    if scale != 1.0:  # back to full-res px
        cells["x_px"] /= scale
        cells["y_px"] /= scale
    cells.insert(0, "cell_id", np.arange(len(cells)))
    cells["scene"] = scene
    return cells


class ArrayReaderShim(SlideReader):
    """Tiny in-memory reader used by :func:`segment_scene` for the downscaled path."""

    def __init__(self, stack, names, pixel_size_um):
        from .io import SceneInfo

        self._arr = stack
        self.channel_names = list(names)
        self.pixel_size_um = pixel_size_um
        self.scenes = [SceneInfo(0, "s", 0, 0, stack.shape[2], stack.shape[1])]
        self.path = None

    def read_region(self, scene, x, y, w, h, channel):
        return self._arr[channel, y:y + h, x:x + w]
