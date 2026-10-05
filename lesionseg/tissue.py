"""Tissue mask from a low-resolution nuclear-channel overview."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage import filters


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    """Drop connected components with fewer than ``min_px`` pixels (version-stable)."""
    if min_px <= 1 or not mask.any():
        return mask
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def fill_small_holes(mask: np.ndarray, max_px: int) -> np.ndarray:
    """Fill background components smaller than ``max_px`` pixels."""
    if max_px <= 1:
        return mask
    holes = ~mask
    small = holes & ~remove_small(holes, max_px)
    return mask | small


def disk(r: int) -> np.ndarray:
    r = int(max(r, 1))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (xx ** 2 + yy ** 2) <= r ** 2


def tissue_mask(overview: np.ndarray, pixel_size_um: float, *, sigma_um: float = 20.0,
                min_area_um2: float = 5e4, hole_area_um2: float = 2e5,
                threshold: float | str | None = None, dilate_um: float = 0.0, **_) -> np.ndarray:
    """Boolean tissue mask at the overview's resolution.

    Parameters
    ----------
    overview : 2-D array, nuclear channel at low resolution.
    pixel_size_um : µm per overview pixel.
    sigma_um : Gaussian smoothing before thresholding.
    min_area_um2 : drop tissue fragments smaller than this.
    hole_area_um2 : fill holes (e.g. central canal, vessels) smaller than this.
    threshold : fixed intensity threshold, ``'otsu'`` / ``'triangle'``, or None (= triangle,
        which is lenient on sparse tissue such as white matter).
    dilate_um : grow the final mask (safety margin when it is used to skip tiles).
    """
    img = overview.astype(np.float32)
    sigma_px = max(sigma_um / pixel_size_um, 0.5)
    sm = filters.gaussian(img, sigma=sigma_px, preserve_range=True)
    if threshold is None or threshold == "triangle":
        thr = filters.threshold_triangle(sm)
    elif threshold == "otsu":
        thr = filters.threshold_otsu(sm)
    else:
        thr = float(threshold)
    mask = sm > thr
    px_area = pixel_size_um ** 2
    mask = remove_small(mask, int(min_area_um2 / px_area))
    mask = fill_small_holes(mask, int(hole_area_um2 / px_area))
    mask = ndi.binary_closing(mask, iterations=2)
    if dilate_um > 0:
        mask = ndi.binary_dilation(mask, structure=disk(int(round(dilate_um / pixel_size_um))))
    return mask


def tissue_from_counts(counts: np.ndarray, bin_um: float, *, sigma_um: float = 30.0,
                       min_cells_per_mm2: float = 100.0, min_area_um2: float = 5e4,
                       hole_area_um2: float = 2e5, dilate_um: float = 20.0, **_) -> np.ndarray:
    """Tissue mask on the density grid derived from cell positions alone.

    Works when only segmentation masks (no image) are available and is robust to
    sparse regions (white matter): any bin with a smoothed density above
    ``min_cells_per_mm2`` counts as tissue; holes (vessels, canal) are filled.
    Keep ``sigma_um`` small (≈30 µm): a wide kernel bleeds past the section edge
    and merges neighbouring cross-sections on the slide into one piece.
    """
    sigma_px = sigma_um / bin_um
    dens = ndi.gaussian_filter(counts.astype(np.float32), sigma_px, mode="constant") / (bin_um / 1000.0) ** 2
    mask = dens > min_cells_per_mm2
    px_area = bin_um ** 2
    mask = remove_small(mask, int(min_area_um2 / px_area))
    mask = fill_small_holes(mask, int(hole_area_um2 / px_area))
    if dilate_um > 0:
        mask = ndi.binary_dilation(mask, structure=disk(int(round(dilate_um / bin_um))))
    return mask


def resample_mask(mask: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resample of a boolean mask to ``out_shape``."""
    zy = out_shape[0] / mask.shape[0]
    zx = out_shape[1] / mask.shape[1]
    return ndi.zoom(mask.astype(np.uint8), (zy, zx), order=0).astype(bool)[: out_shape[0], : out_shape[1]]


def parenchyma_mask(sections: np.ndarray, bin_um: float, *, open_um: float = 150.0, erode_um: float = 20.0,
                    min_area_um2: float = 2e5) -> np.ndarray:
    """Parenchyma = each section opened with a disk of radius ``open_um`` (removes meninges, nerve
    roots and other thin flaps attached to the surface) and eroded by ``erode_um`` (pial margin).

    Returns a bool grid mask. Everything in a section but outside this mask is treated as
    meninges / surface and is neither zoned nor collected.
    """
    out = np.zeros(sections.shape, bool)
    r_open = int(round(open_um / bin_um))
    r_er = int(round(erode_um / bin_um))
    for sid in range(1, int(sections.max()) + 1):
        m = sections == sid
        if not m.any():
            continue
        if r_open > 0:
            m = ndi.binary_opening(m, structure=disk(r_open))
        if r_er > 0:
            m = ndi.binary_erosion(m, structure=disk(r_er))
        m = remove_small(m, int(min_area_um2 / bin_um ** 2))
        out |= m
    return out


def surface_from_counts(counts: np.ndarray, bin_um: float, *, close_um: float = 60.0) -> np.ndarray:
    """Tight tissue surface: bins that contain cells, with gaps up to ``close_um`` closed and all
    interior holes filled – no smoothing or dilation, so the boundary runs through the outermost cells
    (sparse white matter stays solid because every enclosed hole is filled)."""
    m = counts > 0
    r = int(round(close_um / bin_um))
    if r > 0:
        m = ndi.binary_closing(np.pad(m, r), structure=disk(r))[r:-r, r:-r]
    return ndi.binary_fill_holes(m)
