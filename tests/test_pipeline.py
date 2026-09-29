import numpy as np
import pandas as pd

from lesionseg import ZONE_CODES
from lesionseg.config import deep_update, load_config
from lesionseg.pipeline import run_sample


def test_tiles_partition(synthetic):
    reader, _ = synthetic
    tiles = list(reader.iter_tiles(0, tile=512, overlap=64))
    H, W = reader.scenes[0].shape
    cover = np.zeros((H, W), np.int32)
    for t in tiles:
        x0, y0, x1, y1 = t.core
        cover[y0:y1, x0:x1] += 1
        assert t.x + t.w <= W and t.y + t.h <= H
    assert cover.min() == 1 and cover.max() == 1  # cores partition the scene exactly


def test_end_to_end(synthetic, tmp_path):
    reader, truth = synthetic
    cfg = load_config(None)
    cfg = deep_update(cfg, {
        "segmentation": {"tile": 512, "overlap": 64, "params": {"nucleus_diameter_um": 6.0}},
        "pu1": {"method": "gmm"},
        "tissue": {"min_area_um2": 1e3, "hole_area_um2": 1e4, "sigma_um": 40},
        "density": {"bin_um": 5.0, "sigma_um": 20.0},
        "lesion": {"min_area_um2": 2000, "rim_width_um": 20, "peri_width_um": 40, "smooth_um": 10},
    })
    log = run_sample(cfg, name="synthetic", scene=0, out_dir=tmp_path, reader=reader, progress=False)
    cells = pd.read_parquet(tmp_path / "cells.parquet")

    # segmentation recovered (almost) every synthetic nucleus
    n_true = truth["n_total"]
    assert abs(len(cells) - n_true) / n_true < 0.1, (len(cells), n_true)
    # Pu.1 classification roughly right
    n_pos_true = truth["n_pos_out"] + truth["n_pos_in"]
    assert abs(log["pu1"]["n_pos"] - n_pos_true) / n_pos_true < 0.25
    # exactly one lesion, centred where we put it
    assert log["lesion"]["n_lesions"] == 1
    les = pd.read_csv(tmp_path / "lesions.csv").iloc[0]
    cx, cy = truth["lesion_center_um"]
    assert abs(les.centroid_x_um - cx) < 30
    assert abs(les.centroid_y_um - cy) < 30
    assert 0.6 < les.equiv_diameter_um / (2 * truth["lesion_r_um"]) < 1.4
    # the dense-but-Pu.1-negative "canal" was flagged, not called a lesion
    assert log["lesion"]["n_dense_nonmyeloid"] >= 1
    dense = pd.read_csv(tmp_path / "dense_nonmyeloid_regions.csv")
    kx, ky = truth["canal_center_um"]
    assert ((dense.centroid_x_um - kx).abs() < 40).any() and ((dense.centroid_y_um - ky).abs() < 40).any()
    # zones ordered by distance and Pu.1 fraction highest in core
    # (the synthetic lesion has a hard edge, so the thin rim band is nearly empty – only test core vs distal)
    zs = cells.groupby("zone", observed=True)["pu1_pos"].mean()
    assert zs["core"] > 0.7 and zs["core"] > zs["distal"]
    d = cells.groupby("zone", observed=True)["dist_to_lesion_um"].mean()
    assert d["core"] < d["rim"] < 0 < d["peri"] < d["distal"]
    # one tissue piece on the synthetic slide, everything attributed to it
    assert log["n_sections"] == 1
    assert (cells["section_id"] == 1).mean() > 0.99
    sec = pd.read_csv(tmp_path / "section_summary.csv")
    assert len(sec) == 1 and sec.n_lesions.iloc[0] == 1
    # outputs exist
    for f in ["overview.png", "section_summary.csv", "zones_px.geojson", "cells_pu1_px.geojson",
              "maps/zones.tif", "run_log.json",
              "zone_summary.csv", "distance_profile.csv"]:
        assert (tmp_path / f).exists(), f


def test_from_cells_reruns_fast(synthetic, tmp_path):
    reader, _ = synthetic
    cfg = deep_update(load_config(None), {
        "segmentation": {"tile": 512, "overlap": 64},
        "pu1": {"method": "gmm"},
        "tissue": {"min_area_um2": 1e3, "hole_area_um2": 1e4, "sigma_um": 40},
        "density": {"bin_um": 5.0, "sigma_um": 20.0},
        "lesion": {"min_area_um2": 2000, "rim_width_um": 20, "peri_width_um": 40, "smooth_um": 10},
    })
    run_sample(cfg, name="s", scene=0, out_dir=tmp_path, reader=reader, progress=False)
    cfg2 = deep_update(cfg, {"lesion": {"rim_width_um": 40}})
    log = run_sample(cfg2, name="s", scene=0, out_dir=tmp_path, reader=reader, from_cells=tmp_path / "cells.parquet",
                     progress=False)
    assert "from_cells" in log["cells"]
    assert log["lesion"]["rim_width_um"] == 40


def test_zone_codes_consistent():
    assert ZONE_CODES["core"] > ZONE_CODES["rim"] > ZONE_CODES["peri"] > ZONE_CODES["distal"]
