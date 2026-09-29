import numpy as np
import pytest

from lesionseg.io import ArrayReader


def make_synthetic(seed=0, size=1500, px_um=0.5, lesion_center=(750, 750), lesion_r=200):
    """3-channel synthetic slide: nuclei everywhere, dense Pu.1+ cluster in a 'lesion'."""
    rng = np.random.default_rng(seed)
    nuc = np.zeros((size, size), np.float32)
    pu1 = np.zeros((size, size), np.float32)
    yy, xx = np.mgrid[:size, :size]
    tissue = ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) < (size * 0.45) ** 2
    lesion = ((xx - lesion_center[0]) ** 2 + (yy - lesion_center[1]) ** 2) < lesion_r ** 2

    def add_nuclei(mask, n, pos_frac, spacing):
        # jittered grid -> nuclei never overlap (spacing > 2*r + jitter)
        g = np.arange(spacing // 2, size, spacing)
        gy, gx = np.meshgrid(g, g, indexing="ij")
        gy = gy.ravel() + rng.integers(-2, 3, gy.size)
        gx = gx.ravel() + rng.integers(-2, 3, gx.size)
        ok = mask[np.clip(gy, 0, size - 1), np.clip(gx, 0, size - 1)]
        ys, xs = gy[ok], gx[ok]
        n = min(n, len(ys))
        idx = rng.choice(len(ys), n, replace=False)
        pos = rng.random(n) < pos_frac
        r = 5
        for y, x, p in zip(ys[idx], xs[idx], pos, strict=True):
            y0, y1, x0, x1 = max(y - r, 0), min(y + r + 1, size), max(x - r, 0), min(x + r + 1, size)
            sub = ((xx[y0:y1, x0:x1] - x) ** 2 + (yy[y0:y1, x0:x1] - y) ** 2) <= (r - 1) ** 2
            nuc[y0:y1, x0:x1][sub] = 180
            if p:
                pu1[y0:y1, x0:x1][sub] = 150
        return pos.sum()

    # erode masks so grid points near the edge do not straddle the boundary
    from scipy import ndimage as ndi
    lesion_in = ndi.binary_erosion(lesion, iterations=8)
    outside = tissue & ~ndi.binary_dilation(lesion, iterations=8)
    # dense but Pu.1-negative blob = "central canal" (must never be called a lesion)
    canal_center = (350, 1100)
    canal = ((xx - canal_center[0]) ** 2 + (yy - canal_center[1]) ** 2) < 90 ** 2
    canal_in = ndi.binary_erosion(canal, iterations=8)
    outside = outside & ~ndi.binary_dilation(canal, iterations=8)
    n_out = add_nuclei(outside, 1500, 0.1, spacing=30)
    n_in = add_nuclei(lesion_in, 600, 0.8, spacing=15)
    add_nuclei(canal_in, 200, 0.0, spacing=13)
    n_total = 1500 + 600 + 200
    nuc += rng.normal(10, 3, nuc.shape)
    pu1 += rng.normal(8, 3, pu1.shape) + 5 * tissue
    stack = np.clip(np.stack([nuc, pu1, np.zeros_like(nuc)]), 0, 255).astype(np.uint8)
    return stack, px_um, {"lesion_center_um": (lesion_center[0] * px_um, lesion_center[1] * px_um),
                          "lesion_r_um": lesion_r * px_um, "n_pos_out": n_out, "n_pos_in": n_in, "n_total": n_total,
                          "canal_center_um": (canal_center[0] * px_um, canal_center[1] * px_um)}


@pytest.fixture(scope="session")
def synthetic():
    stack, px, truth = make_synthetic()
    return ArrayReader(stack, ["SYTOG", "AF647", "AF555"], px), truth
