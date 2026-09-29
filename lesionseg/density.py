"""Cell-density maps on a regular µm grid."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .tissue import resample_mask


@dataclass
class Grid:
    """Regular grid over a scene. ``bin_um`` µm per grid pixel; origin at scene (0, 0)."""

    bin_um: float
    shape: tuple[int, int]  # (rows, cols)
    pixel_size_um: float  # of the full-res image

    @property
    def scale(self) -> float:
        """grid px per full-res image px."""
        return self.pixel_size_um / self.bin_um

    def to_grid(self, x_um, y_um):
        return np.asarray(x_um) / self.bin_um, np.asarray(y_um) / self.bin_um

    def bin_area_mm2(self) -> float:
        return (self.bin_um / 1000.0) ** 2

    @classmethod
    def for_scene(cls, width_px: int, height_px: int, pixel_size_um: float, bin_um: float) -> Grid:
        rows = int(np.ceil(height_px * pixel_size_um / bin_um))
        cols = int(np.ceil(width_px * pixel_size_um / bin_um))
        return cls(bin_um, (rows, cols), pixel_size_um)


@dataclass
class DensityMaps:
    grid: Grid
    tissue: np.ndarray                         # bool, grid shape
    nuclei: np.ndarray                         # cells / mm² (smoothed)
    pu1: np.ndarray                            # Pu.1+ cells / mm² (smoothed)
    pu1_fraction: np.ndarray                   # pu1 / nuclei (0 where no nuclei)
    extra: dict = field(default_factory=dict)  # named additional maps (z-scores, score …)

    def as_dict(self) -> dict[str, np.ndarray]:
        d = {"tissue": self.tissue.astype(np.uint8), "nuclei_density": self.nuclei,
             "pu1_density": self.pu1, "pu1_fraction": self.pu1_fraction}
        d.update(self.extra)
        return d


def count_map(x_um, y_um, grid: Grid) -> np.ndarray:
    gx, gy = grid.to_grid(x_um, y_um)
    rows, cols = grid.shape
    ix = np.clip(np.floor(gx).astype(int), 0, cols - 1)
    iy = np.clip(np.floor(gy).astype(int), 0, rows - 1)
    counts = np.zeros(grid.shape, np.float32)
    np.add.at(counts, (iy, ix), 1)
    return counts


def density_map(counts: np.ndarray, grid: Grid, sigma_um: float, tissue: np.ndarray | None = None) -> np.ndarray:
    """Gaussian-smoothed density in cells/mm².

    Inside tissue the kernel is renormalised by the smoothed tissue mask so that
    bins at the tissue edge are not diluted by empty space outside the section.
    """
    sigma_px = sigma_um / grid.bin_um
    dens = ndi.gaussian_filter(counts, sigma_px, mode="constant") / grid.bin_area_mm2()
    if tissue is not None:
        w = ndi.gaussian_filter(tissue.astype(np.float32), sigma_px, mode="constant")
        with np.errstate(divide="ignore", invalid="ignore"):
            dens = np.where(w > 0.05, dens / np.clip(w, 0.05, 1), 0.0)
        dens[~tissue] = 0.0
    return dens.astype(np.float32)


def compute_density_maps(cells: pd.DataFrame, grid: Grid, *, tissue_mask_lowres: np.ndarray | None = None,
                         sigma_um: float = 40.0, pos_col: str = "pu1_pos", tissue_source: str = "cells",
                         tissue_params: dict | None = None) -> DensityMaps:
    """Density maps + grid tissue mask.

    ``tissue_source``: ``'cells'`` (from cell positions, default), ``'image'``
    (resample ``tissue_mask_lowres``) or ``'both'`` (union).
    """
    from .tissue import tissue_from_counts

    n_counts = count_map(cells["x_um"], cells["y_um"], grid)
    tp = dict(tissue_params or {})
    if tissue_source in ("cells", "both") or tissue_mask_lowres is None:
        tissue = tissue_from_counts(n_counts, grid.bin_um, **tp)
        if tissue_source == "both" and tissue_mask_lowres is not None:
            tissue |= resample_mask(tissue_mask_lowres, grid.shape)
    else:
        tissue = resample_mask(tissue_mask_lowres, grid.shape)
        if tissue.shape != grid.shape:  # pad if rounding made it short
            t = np.zeros(grid.shape, bool)
            t[: tissue.shape[0], : tissue.shape[1]] = tissue
            tissue = t
    pos = cells[pos_col].to_numpy(bool) if pos_col in cells else np.zeros(len(cells), bool)
    p_counts = count_map(cells["x_um"][pos], cells["y_um"][pos], grid)
    nuc = density_map(n_counts, grid, sigma_um, tissue)
    pu1 = density_map(p_counts, grid, sigma_um, tissue)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(nuc > 0, pu1 / nuc, 0.0).astype(np.float32)
    return DensityMaps(grid, tissue, nuc, pu1, frac)


def robust_z(values: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(z, median, MAD-scaled sigma) using only ``mask`` bins as the reference population."""
    ref = values[mask]
    med = float(np.median(ref)) if ref.size else 0.0
    mad = float(np.median(np.abs(ref - med))) * 1.4826 if ref.size else 1.0
    mad = mad if mad > 1e-9 else float(np.std(ref) + 1e-9)
    z = (values - med) / mad
    z[~mask] = 0.0
    return z.astype(np.float32), med, mad
