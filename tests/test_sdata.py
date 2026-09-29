"""SpatialData route: zarr v3 labels + AnnData table + manual shapes (the collaborator's export format)."""
import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi

from lesionseg.config import deep_update, load_config
from lesionseg.pipeline import run_sample

zarr = pytest.importorskip("zarr")
ad = pytest.importorskip("anndata")


def _build_store(reader, truth, root):
    from shapely import wkb
    from shapely.geometry import Point, box

    stack = reader._arr
    labels, _ = ndi.label(stack[0] > 100)
    labels = labels.astype(np.int32)
    z = root / "slide.zarr"
    arr = zarr.create_array(str(z / "labels" / "slide_masks_filtered" / "s0"), shape=labels.shape, dtype="int32",
                            chunks=(512, 512))
    arr[:] = labels
    # table
    idx = np.arange(1, labels.max() + 1)
    cy, cx = np.array(ndi.center_of_mass(labels > 0, labels, idx)).T
    area = np.bincount(labels.ravel())[1:]
    pu1 = ndi.mean(stack[1], labels, idx)
    obs = pd.DataFrame({"cell_id": idx, "slide_name": "slide", "centroid_x": cx, "centroid_y": cy, "area": area,
                        "eccentricity": 0.5, "solidity": 0.9,
                        "Pu1_class": np.where(pu1 > 60, "Pu1_positive", "Pu1_negative")},
                       index=[f"slide_{i}" for i in idx])
    X = np.column_stack([ndi.mean(stack[c], labels, idx) for c in range(3)])
    a = ad.AnnData(X=X.astype(np.float32), obs=obs)
    a.var_names = ["SytoG", "Pu1", "Iba1"]
    a.write_h5ad(root / "table.h5ad")
    # manual shapes: CORE = true lesion disc (px), Sample_category = whole tissue, VBO = small box
    px = reader.pixel_size_um
    lc = np.array(truth["lesion_center_um"]) / px
    core = Point(*lc).buffer(truth["lesion_r_um"] / px * 0.6)
    tissue = Point(labels.shape[1] / 2, labels.shape[0] / 2).buffer(labels.shape[0] * 0.45)
    vbo = box(100, 100, 200, 200)
    for name, geoms, extra in (("CORE", [core], {}), ("Sample_category", [tissue], {"sample_category": ["T1"]}),
                               ("VBO", [vbo], {})):
        d = z / "shapes" / name
        d.mkdir(parents=True)
        df = pd.DataFrame({**extra, "geometry": [wkb.dumps(g) for g in geoms]})
        df.to_parquet(d / "shapes.parquet")
    return z, root / "table.h5ad", int((obs["Pu1_class"] == "Pu1_positive").sum())


def test_spatialdata_route(synthetic, tmp_path):
    reader, truth = synthetic
    z, table, n_pos = _build_store(reader, truth, tmp_path)
    cfg = deep_update(load_config(None), {
        "tissue": {"min_area_um2": 1e3, "hole_area_um2": 1e4, "sigma_um": 40},
        "density": {"bin_um": 5.0, "sigma_um": 20.0},
        "sections": {"lesion_min_area_mm2": 0.01},
        "lesion": {"min_area_um2": 2000, "rim_width_um": 20, "peri_width_um": 40, "smooth_um": 10},
        "wells": {"target_area_um2": 300.0, "edge_exclusion_um": 20.0,
                  "groups": [{"name": "core", "zone": ["core"]}, {"name": "ring", "dist_um": [0, 60]},
                             {"name": "mc_ring", "manual_dist_um": [0, 60]}]},
    })
    out = tmp_path / "out"
    log = run_sample(cfg, name="slide", out_dir=out, sdata_path=z, table_path=table, slide_name="slide",
                     pixel_size_um=reader.pixel_size_um, progress=False)
    cells = pd.read_parquet(out / "cells.parquet")
    assert log["cells"]["route"] == "spatialdata"
    assert log["pu1"]["n_pos"] == n_pos and log["lesion"]["n_lesions"] == 1
    # contours traced from labels for Pu.1+ only (no per-cell shapes layer in this store)
    assert cells.loc[cells.pu1_pos, "contour_wkt"].notna().all()
    # sections come from the Sample_category polygon; manual core validated
    assert log["sections_from"] == "Sample_category polygons"
    assert (cells["section_name"] == "T1").mean() > 0.95
    assert log["validation"]["core_vs_lesion"]["manual_covered_by_auto"] > 0.9
    assert (cells["manual_core_id"] > 0).sum() > 50
    assert (cells["in_vbo"]).sum() < 10
    assert log["section_focus"]["mode"] == "auto" and log["section_focus"]["lesion_sections"] == [1]
    # wells: core + ring groups filled, only Pu.1+ cells, none inside VBO or near the edge
    w = pd.read_csv(out / "wells.csv")
    assert set(w.loc[w.well_group > 0, "group"]) == {"core", "ring", "mc_ring"}
    sel = cells[cells.well_group > 0]
    assert sel["pu1_pos"].all() and not sel["in_vbo"].any()
    assert (sel["dist_to_section_edge_um"] >= 20).all()
    assert abs(w.loc[w.group == "core", "area_um2"].iloc[0] - 300) < 60
    # manual-core rings sit inside our lesion (manual core is smaller than the auto lesion)
    mc = sel[sel.well_name.str.endswith("mc_ring")]
    assert (mc["dist_to_lesion_um"] < 0).mean() > 0.8
