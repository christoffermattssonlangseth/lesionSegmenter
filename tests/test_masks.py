"""Mask route: label images for all cells + Pu.1+ cells (the expected real input)."""
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi

from lesionseg.config import deep_update, load_config
from lesionseg.pipeline import run_sample


def _masks_from_synthetic(reader):
    stack = reader._arr
    cells, _ = ndi.label(stack[0] > 100)
    # Pu.1 mask = cells whose mean Pu.1 intensity is high (independent segmentation: slightly shrunk)
    idx = np.arange(1, cells.max() + 1)
    means = ndi.mean(stack[1], cells, idx)
    pos_labels = idx[means > 60]
    pu1 = np.where(np.isin(cells, pos_labels), cells, 0)
    pu1 = ndi.grey_erosion(pu1, size=3)  # different outline than the cell mask
    # relabel pu1 independently (as a separate segmentation would)
    pu1, _ = ndi.label(pu1 > 0)
    return cells.astype(np.int32), pu1.astype(np.int32), len(pos_labels)


def test_mask_route(synthetic, tmp_path):
    reader, truth = synthetic
    cells_lab, pu1_lab, n_pos = _masks_from_synthetic(reader)
    tifffile.imwrite(tmp_path / "cells.tif", cells_lab)
    tifffile.imwrite(tmp_path / "pu1.tif", pu1_lab)
    cfg = deep_update(load_config(None), {
        "tissue": {"min_area_um2": 1e3, "hole_area_um2": 1e4, "sigma_um": 40},
        "density": {"bin_um": 5.0, "sigma_um": 20.0},
        "lesion": {"min_area_um2": 2000, "rim_width_um": 20, "peri_width_um": 40, "smooth_um": 10},
    })
    out = tmp_path / "out"
    # 1) masks only (no image) ---------------------------------------------
    log = run_sample(cfg, name="masks", out_dir=out, cells_mask=tmp_path / "cells.tif",
                     pu1_mask=tmp_path / "pu1.tif", pixel_size_um=reader.pixel_size_um, progress=False)
    cells = pd.read_parquet(out / "cells.parquet")
    assert len(cells) == cells_lab.max()
    assert log["pu1"]["n_pos"] == n_pos
    assert log["cells"]["pu1_match"]["n_unmatched_pu1"] == 0
    assert log["lesion"]["n_lesions"] == 1
    zs = cells.groupby("zone", observed=True)["pu1_pos"].mean()
    assert zs["core"] > 0.7 and zs["core"] > zs["distal"]
    les = pd.read_csv(out / "lesions.csv").iloc[0]
    assert les.mean_pu1_fraction > 0.5
    assert abs(les.centroid_x_um - truth["lesion_center_um"][0]) < 30
    # the Pu.1-negative dense "canal" is flagged, not a lesion
    assert log["lesion"]["n_lesions"] == 1 and log["lesion"]["n_dense_nonmyeloid"] >= 1
    # contours only for Pu.1+ cells by default
    assert cells.loc[cells.pu1_pos, "contour_wkt"].notna().all()
    assert cells.loc[~cells.pu1_pos, "contour_wkt"].isna().all()
    assert (out / "cells_pu1_px.geojson").exists()

    # 2) masks + image: intensities get measured -----------------------------
    out2 = tmp_path / "out2"
    log2 = run_sample(cfg, name="masks_img", out_dir=out2, reader=reader, cells_mask=tmp_path / "cells.tif",
                      pu1_mask=tmp_path / "pu1.tif", progress=False)
    cells2 = pd.read_parquet(out2 / "cells.parquet")
    assert "AF647_mean" in cells2
    assert cells2.loc[cells2.pu1_pos, "AF647_mean"].mean() > 3 * cells2.loc[~cells2.pu1_pos, "AF647_mean"].mean()
    assert log2["lesion"]["n_lesions"] == 1


def test_unmatched_pu1_objects_are_kept(tmp_path):
    from lesionseg.masks import cells_from_labels, match_pu1_mask

    cells = np.zeros((100, 100), np.int32)
    cells[10:20, 10:20] = 1
    cells[50:60, 50:60] = 2
    pu1 = np.zeros((100, 100), np.int32)
    pu1[12:18, 12:18] = 7      # inside cell 1
    pu1[80:90, 80:90] = 8      # no cell here
    tab = cells_from_labels(cells, 1.0, contours=False, progress=False)
    out, info = match_pu1_mask(cells, pu1, tab)
    assert info["n_matched_cells"] == 1 and info["n_unmatched_pu1"] == 1
    assert out["pu1_pos"].sum() == 2 and (out["source"] == "pu1_mask").sum() == 1
    assert out.loc[out.source == "pu1_mask", "label"].iloc[0] == -8
