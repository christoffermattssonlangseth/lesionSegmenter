"""Readers for whole-slide images.

Two backends share one small interface (:class:`SlideReader`):

* :class:`CziReader` – Zeiss CZI mosaics via ``aicspylibczi``. Reads downscaled
  overviews and full-resolution regions without loading the whole scan.
* :class:`ArrayReader` – any in-memory ``(C, Y, X)`` array (TIFF / synthetic
  data / tests).

Coordinates
-----------
All region coordinates handed to/returned by a reader are **scene pixel
coordinates**: ``(0, 0)`` is the top-left of the scene bounding box at full
resolution. The reader takes care of the (often negative) absolute mosaic
offsets that CZI files use internally.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class SceneInfo:
    index: int
    name: str
    x0: int  # absolute mosaic offset (CZI); 0 for arrays
    y0: int
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width


@dataclass
class Tile:
    """A full-resolution tile in scene pixel coordinates.

    ``(x, y, w, h)`` is the region read from disk (including overlap);
    ``core`` is the inner box whose objects this tile "owns" when stitching.
    """

    x: int
    y: int
    w: int
    h: int
    core: tuple[int, int, int, int]  # x0, y0, x1, y1 (exclusive) in scene px
    index: tuple[int, int] = field(default=(0, 0))


class SlideReader:
    """Minimal interface implemented by both backends."""

    path: Path
    channel_names: list[str]
    pixel_size_um: float
    scenes: list[SceneInfo]

    def channel_index(self, name_or_index: str | int) -> int:
        if isinstance(name_or_index, int):
            return name_or_index
        if name_or_index in self.channel_names:
            return self.channel_names.index(name_or_index)
        # allow case-insensitive / substring matches ("pu1" in "Pu.1-AF647")
        low = [c.lower() for c in self.channel_names]
        key = str(name_or_index).lower()
        for i, c in enumerate(low):
            if key == c or key in c:
                return i
        raise KeyError(f"channel {name_or_index!r} not in {self.channel_names}")

    # -- to implement ------------------------------------------------------
    def read_region(self, scene: int, x: int, y: int, w: int, h: int, channel: int) -> np.ndarray:
        raise NotImplementedError

    def read_overview(self, scene: int, scale: float, channel: int) -> np.ndarray:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    def read_region_stack(self, scene, x, y, w, h, channels: Sequence[int]) -> np.ndarray:
        return np.stack([self.read_region(scene, x, y, w, h, c) for c in channels], axis=0)

    def read_overview_stack(self, scene, scale, channels: Sequence[int]) -> np.ndarray:
        return np.stack([self.read_overview(scene, scale, c) for c in channels], axis=0)

    def iter_tiles(self, scene: int, tile: int = 2048, overlap: int = 128) -> Iterator[Tile]:
        """Yield overlapping tiles covering the scene.

        Neighbouring tiles overlap by ``overlap`` px on each side; each tile's
        ``core`` box is the tile shrunk by ``overlap/2`` (clamped at the scene
        border), so cores partition the scene exactly.
        """
        info = self.scenes[scene]
        H, W = info.height, info.width
        step = tile - overlap
        half = overlap // 2
        ny = int(np.ceil(max(H - overlap, 1) / step))
        nx = int(np.ceil(max(W - overlap, 1) / step))
        for j in range(ny):
            for i in range(nx):
                x = i * step
                y = j * step
                w = min(tile, W - x)
                h = min(tile, H - y)
                if w <= 0 or h <= 0:
                    continue
                cx0 = 0 if i == 0 else x + half
                cy0 = 0 if j == 0 else y + half
                cx1 = W if i == nx - 1 else x + w - half
                cy1 = H if j == ny - 1 else y + h - half
                yield Tile(x, y, w, h, (cx0, cy0, cx1, cy1), (j, i))

    def n_tiles(self, scene: int, tile: int = 2048, overlap: int = 128) -> int:
        return sum(1 for _ in self.iter_tiles(scene, tile, overlap))


# --------------------------------------------------------------------------
# CZI
# --------------------------------------------------------------------------
class CziReader(SlideReader):
    def __init__(self, path: str | Path):
        import aicspylibczi  # local import: optional at import time of the package

        self.path = Path(path)
        self._czi = aicspylibczi.CziFile(str(self.path))
        if not self._czi.is_mosaic():
            raise ValueError("Only mosaic CZI files are supported (tiled slide scans).")
        self.channel_names = _czi_channel_names(self._czi)
        self.pixel_size_um = _czi_pixel_size_um(self._czi)
        self.scenes = self._scene_infos()

    # -- metadata ----------------------------------------------------------
    def _scene_infos(self) -> list[SceneInfo]:
        infos = []
        try:
            bboxes = self._czi.get_all_scene_bounding_boxes()
        except Exception:  # single-scene / old library
            bb = self._czi.get_mosaic_bounding_box()
            bboxes = {0: bb}
        names = _czi_scene_names(self._czi)
        for idx in sorted(bboxes):
            bb = bboxes[idx]
            infos.append(
                SceneInfo(
                    index=idx,
                    name=names.get(idx, f"scene{idx}"),
                    x0=bb.x,
                    y0=bb.y,
                    width=bb.w,
                    height=bb.h,
                )
            )
        return infos

    def info(self) -> dict:
        return {
            "path": str(self.path),
            "channels": self.channel_names,
            "pixel_size_um": self.pixel_size_um,
            "scenes": [
                {"index": s.index, "name": s.name, "width_px": s.width, "height_px": s.height,
                 "width_mm": round(s.width * self.pixel_size_um / 1000, 2),
                 "height_mm": round(s.height * self.pixel_size_um / 1000, 2)}
                for s in self.scenes
            ],
        }

    # -- pixels ------------------------------------------------------------
    def read_region(self, scene, x, y, w, h, channel):
        info = self.scenes[scene]
        # clamp to scene
        x = max(0, x)
        y = max(0, y)
        w = min(w, info.width - x)
        h = min(h, info.height - y)
        region = (info.x0 + x, info.y0 + y, w, h)
        arr = self._czi.read_mosaic(region=region, scale_factor=1.0, C=channel)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        return arr

    def read_overview(self, scene, scale, channel):
        info = self.scenes[scene]
        region = (info.x0, info.y0, info.width, info.height)
        arr = self._czi.read_mosaic(region=region, scale_factor=float(scale), C=channel)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[-2], arr.shape[-1])
        return arr


def _czi_channel_names(czi) -> list[str]:
    try:
        root = czi.meta
        chans = root.findall(".//Metadata/Information/Image/Dimensions/Channels/Channel")
        names = []
        for c in chans:
            n = c.get("Name")
            if not n:
                f = c.find("Fluor")
                n = f.text if f is not None else c.get("Id", "C")
            names.append(n)
        if names:
            return names
    except Exception:
        pass
    dims = czi.get_dims_shape()[0]
    n = dims["C"][1] - dims["C"][0]
    return [f"C{i}" for i in range(n)]


def _czi_pixel_size_um(czi) -> float:
    try:
        root: ET.Element = czi.meta
        for d in root.findall(".//Metadata/Scaling/Items/Distance"):
            if d.get("Id") == "X":
                v = d.find("Value")
                return float(v.text) * 1e6  # metres -> µm
    except Exception:
        pass
    raise ValueError("Could not read pixel size from CZI metadata; set pixel_size_um in config.")


def _czi_scene_names(czi) -> dict[int, str]:
    out = {}
    try:
        root = czi.meta
        for s in root.findall(".//Metadata/Information/Image/Dimensions/S/Scenes/Scene"):
            idx = int(s.get("Index"))
            out[idx] = s.get("Name", f"scene{idx}")
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------
# In-memory arrays (TIFF, synthetic)
# --------------------------------------------------------------------------
class ArrayReader(SlideReader):
    """Wrap a ``(C, Y, X)`` array (or ``(Y, X)`` for single channel)."""

    def __init__(self, array: np.ndarray, channel_names: Sequence[str] | None,
                 pixel_size_um: float, path: str | Path = "<array>"):
        arr = np.asarray(array)
        if arr.ndim == 2:
            arr = arr[None]
        if arr.ndim != 3:
            raise ValueError("array must be (C, Y, X)")
        self._arr = arr
        self.path = Path(path)
        self.channel_names = list(channel_names) if channel_names else [f"C{i}" for i in range(arr.shape[0])]
        self.pixel_size_um = float(pixel_size_um)
        self.scenes = [SceneInfo(0, "scene0", 0, 0, arr.shape[2], arr.shape[1])]

    @classmethod
    def from_tiff(cls, path, channel_names=None, pixel_size_um=None):
        import tifffile

        with tifffile.TiffFile(path) as tf:
            arr = tf.asarray()
            if pixel_size_um is None:
                try:
                    tags = tf.pages[0].tags
                    xres = tags["XResolution"].value
                    unit = tags.get("ResolutionUnit")
                    px = xres[1] / xres[0]
                    if unit is not None and unit.value == 3:  # centimetre
                        px *= 1e4
                    elif unit is not None and unit.value == 2:  # inch
                        px *= 25400
                    pixel_size_um = px
                except Exception as e:  # pragma: no cover
                    raise ValueError("pixel_size_um required (not found in TIFF tags)") from e
        return cls(arr, channel_names, pixel_size_um, path)

    def read_region(self, scene, x, y, w, h, channel):
        return self._arr[channel, y:y + h, x:x + w]

    def read_overview(self, scene, scale, channel):
        from skimage.transform import rescale

        img = self._arr[channel]
        if scale == 1.0:
            return img
        out = rescale(img.astype(np.float32), scale, anti_aliasing=True, preserve_range=True)
        return out.astype(img.dtype)


def open_slide(path: str | Path, channel_names=None, pixel_size_um=None) -> SlideReader:
    path = Path(path)
    if path.suffix.lower() == ".czi":
        return CziReader(path)
    if path.suffix.lower() in {".tif", ".tiff", ".ome.tif"}:
        return ArrayReader.from_tiff(path, channel_names, pixel_size_um)
    if path.suffix.lower() == ".npy":
        if pixel_size_um is None:
            raise ValueError("pixel_size_um required for .npy input")
        return ArrayReader(np.load(path), channel_names, pixel_size_um, path)
    raise ValueError(f"unsupported input: {path}")
