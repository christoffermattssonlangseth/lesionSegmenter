import numpy as np

from lesionseg import orientation as O


def _ellipse(lab, cx, cy, a, b, theta, value):
    yy, xx = np.mgrid[: lab.shape[0], : lab.shape[1]]
    u = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    v = -(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta)
    lab[(u / a) ** 2 + (v / b) ** 2 <= 1] = value


def test_label_moments_recovers_angle_and_eccentricity_across_chunks():
    lab = np.zeros((200, 200), np.int32)
    angles = {1: 0.0, 2: np.pi / 4, 3: -np.pi / 3, 4: np.pi / 2}
    centres = {1: (40, 40), 2: (100, 64), 3: (60, 150), 4: (150, 150)}   # label 2 straddles the chunk border
    for k, th in angles.items():
        _ellipse(lab, *centres[k], 12, 4, th, k)
    ref = np.zeros((5, 2))
    for k, (x, y) in centres.items():
        ref[k] = (x + 1.5, y - 2.0)   # reference points need not be the exact centroid
    m = O.label_moments(lab, ref, chunk=64).set_index("label")
    for k, th in angles.items():
        d = (m.loc[k, "orient_rad"] - th + np.pi / 2) % np.pi - np.pi / 2   # axes are mod π
        assert abs(d) < 0.03
        assert abs(m.loc[k, "ecc_mom"] - np.sqrt(1 - (4 / 12) ** 2)) < 0.03


def test_surface_angle_tangential_vs_radial_on_a_disk():
    n, R = 400, 180.0
    yy, xx = np.mgrid[:n, :n]
    tissue = np.hypot(xx - n / 2, yy - n / 2) < R
    nx, ny = O.surface_normal(tissue, res_um=1.0, sigma_um=3.0)
    phi = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    px = (n / 2 + (R - 10) * np.cos(phi)).astype(int)
    py = (n / 2 + (R - 10) * np.sin(phi)).astype(int)
    tang = O.surface_angle(phi + np.pi / 2, nx[py, px], ny[py, px])
    rad = O.surface_angle(phi, nx[py, px], ny[py, px])
    assert tang.max() < 8 and rad.min() > 82
    assert O.order_parameter(tang) > 0.95 and O.order_parameter(rad) < -0.95
    assert abs(O.order_parameter(np.random.default_rng(0).uniform(0, 90, 20000))) < 0.03
