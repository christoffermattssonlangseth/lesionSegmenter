import numpy as np
import pandas as pd
from shapely.geometry import box

from lesionseg import meninges

PX = 0.65        # µm per scene px
SIZE_UM = 1400.0
CENTER, R = 700.0, 600.0


def _tissue():
    """Round section with sparse nuclei, a compact band at the surface (east side), a compact patch
    inside the tissue that does not touch the surface, and nuclear image + grid maps to match."""
    rng = np.random.default_rng(0)

    def lattice(spacing, keep):
        g = np.arange(0, SIZE_UM, spacing)
        x, y = (a.ravel() + rng.uniform(-1, 1, a.size) for a in np.meshgrid(g, g))
        return x[keep(x, y)], y[keep(x, y)]

    def r_of(x, y):
        return np.hypot(x - CENTER, y - CENTER)

    sparse = lattice(16.0, lambda x, y: r_of(x, y) < R)
    band = lattice(6.0, lambda x, y: (r_of(x, y) < R) & (r_of(x, y) > R - 40) & (np.abs(y - CENTER) < 150)
                   & (x > CENTER))
    patch = lattice(6.0, lambda x, y: np.hypot(x - CENTER, y - (CENTER - R + 90)) < 30)
    x = np.concatenate([sparse[0], band[0], patch[0]])
    y = np.concatenate([sparse[1], band[1], patch[1]])
    kind = np.repeat(["sparse", "band", "patch"], [len(sparse[0]), len(band[0]), len(patch[0])])
    cells = pd.DataFrame({"x_um": x, "y_um": y, "kind": kind, "section_id": 1})
    cells = cells.drop_duplicates(subset=["x_um", "y_um"]).reset_index(drop=True)
    # nuclear channel at scale 0.5 of the scene: a 3 µm disk per nucleus
    scale = 0.5
    res = PX / scale
    n = int(np.ceil(SIZE_UM / PX * scale))
    img = np.zeros((n, n), np.float32)
    yy, xx = np.mgrid[-3:4, -3:4]
    disk = (xx ** 2 + yy ** 2) <= (3.0 / res) ** 2
    for cx, cy in zip((cells.x_um / res).astype(int), (cells.y_um / res).astype(int), strict=True):
        if 3 <= cx < n - 3 and 3 <= cy < n - 3:
            img[cy - 3:cy + 4, cx - 3:cx + 4] = np.maximum(img[cy - 3:cy + 4, cx - 3:cx + 4], disk)
    img += rng.uniform(0, 0.05, img.shape).astype(np.float32)
    bin_um = 10.0
    g = int(np.ceil(SIZE_UM / bin_um))
    gy, gx = (np.mgrid[:g, :g] + 0.5) * bin_um
    sections = (np.hypot(gx - CENTER, gy - CENTER) < R).astype(np.int32)
    shape_px = (int(np.ceil(SIZE_UM / PX)),) * 2
    return cells, img, sections, bin_um, shape_px, scale


def _run(protect=None):
    cells, img, sections, bin_um, shape_px, scale = _tissue()
    lay = meninges.surface_layers(cells, sections, sections > 0, bin_um, shape_px, PX, img,
                                  protect_shapes=protect, scale=scale, compact_q=99.0)
    return meninges.cell_tiers(cells, lay), lay


def test_compact_surface_band_is_meninges_but_inner_patch_is_not():
    c, lay = _run()
    band = c[c.kind == "band"]
    assert (band.surface_tier == "meninges").mean() > 0.9
    patch = c[c.kind == "patch"]
    assert (patch.surface_tier == "parenchyma").mean() > 0.9   # compact but not touching the surface
    deep = c[(c.kind == "sparse") & (c.depth_um > 90)]
    assert (deep.surface_tier == "parenchyma").all()
    # a buffer separates meninges from parenchyma
    assert (c.surface_tier == "buffer").sum() > 0
    assert lay["meninges"].sum() > 0 and not (lay["meninges"] & lay["buffer"]).any()


def test_manual_cores_are_never_meninges():
    # a manual core covering the outer half of the band, in scene px
    core = box((CENTER + R - 60) / PX, (CENTER - 60) / PX, (CENTER + R + 20) / PX, (CENTER + 60) / PX)
    c, lay = _run(protect=[core])
    inside = c.x_um.between(CENTER + R - 60, CENTER + R + 20) & c.y_um.between(CENTER - 60, CENTER + 60)
    assert inside.sum() > 0
    assert (c.loc[inside, "surface_tier"] == "parenchyma").all()
    assert not (lay["meninges"] & lay["protect"]).any() and not (lay["buffer"] & lay["protect"]).any()
    # the band outside the core is still meninges, and buffer cells sit between it and the core
    rest = c[(c.kind == "band") & ~inside]
    assert (rest.surface_tier.isin(["meninges", "buffer"])).all()
    assert (rest.surface_tier == "meninges").sum() > 0
