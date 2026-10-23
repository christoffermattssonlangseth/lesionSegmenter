import numpy as np

from lesionseg import xenium_meninges as X


def _disk_section(r_um=600.0, men_um=20.0, res=5.0, seed=0):
    """A round section: uniform transcripts, meningeal genes in the outer ``men_um``, glia limitans below it."""
    rng = np.random.default_rng(seed)
    n = int(2 * (r_um + 150) / res)
    yy, xx = (np.mgrid[:n, :n] + 0.5) * res - (r_um + 150)
    rad = np.hypot(xx, yy)
    tis = rad <= r_um
    total = rng.poisson(np.where(tis, 200, 1))
    crop = {k: np.zeros_like(total) for k in X.GROUPS}
    crop["total"] = total
    crop["meningeal"] = rng.poisson(np.where(tis & (rad > r_um - men_um), 6, 0.05 * tis))
    crop["glia_limitans"] = rng.poisson(np.where(tis & (rad <= r_um - men_um), 8, 0.5 * tis))
    return crop, res


def test_tissue_mask_follows_the_edge():
    crop, res = _disk_section()
    tis = X.tissue_mask(crop["total"], res)
    truth = crop["total"] > 50
    assert (tis != truth).mean() < 0.01


def test_pia_depth_finds_the_meningeal_layer():
    crop, res = _disk_section(men_um=20.0)
    tis = X.tissue_mask(crop["total"], res)
    depth, seg = X.depth_and_segments(tis, res, (0.0, 0.0))
    prof = X.section_profiles(crop, tis, depth, seg).assign(section="s")
    pia = X.pia_depth(prof, X.section_refs(prof))
    assert len(pia) > 50
    assert abs(pia["pia_depth_um"].median() - 20) <= 5


def test_stretch_without_parenchyma_has_no_pia():
    crop, res = _disk_section(men_um=20.0)
    crop["glia_limitans"][:] = 0                   # a meningeal flap: nothing parenchymal underneath
    crop["glia_limitans"][0, 0] = 1
    tis = X.tissue_mask(crop["total"], res)
    depth, seg = X.depth_and_segments(tis, res, (0.0, 0.0))
    prof = X.section_profiles(crop, tis, depth, seg).assign(section="s")
    ref = X.section_refs(prof)
    ref["s"]["gl_deep"] = 1.0                      # deep level the flap never reaches
    assert X.pia_depth(prof, ref).empty
