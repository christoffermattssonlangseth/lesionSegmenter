import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
from shapely.geometry import box, mapping

from lesionseg import meninges_model as MM


def _grid_cells(n=20, step=10.0):
    g = np.arange(n) * step
    x, y = (a.ravel() for a in np.meshgrid(g, g))
    return pd.DataFrame({"x_px": x, "y_px": y})


def test_cell_labels_roi_meninges_unsure():
    c = _grid_cells()
    annot = {MM.ROI_LAYER: [box(-1, -1, 101, 101)], MM.ANNOT_LAYER: [box(-1, -1, 31, 101)],
             MM.UNSURE_LAYER: [box(25, -1, 41, 101)]}
    lab = MM.cell_labels(c, annot)
    inside = (c.x_px <= 100) & (c.y_px <= 100)
    assert (lab.y[~inside] == -1).all() and (lab.roi[~inside] == -1).all()
    assert (lab.y[inside & (c.x_px <= 20)] == 1).all()
    assert (lab.y[inside & c.x_px.between(30, 40)] == -1).all()          # unsure wins
    assert (lab.y[inside & (c.x_px >= 50)] == 0).all()


def test_load_annotations_from_geojson(tmp_path):
    run = SimpleNamespace(name="S scene 0", log={"cells": {"slide_name": "S"}})
    for layer, g in ((MM.ROI_LAYER, box(0, 0, 10, 10)), (MM.ANNOT_LAYER, box(0, 0, 3, 10))):
        fc = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {}, "geometry": mapping(g)}]}
        (tmp_path / f"S_{layer}.geojson").write_text(json.dumps(fc))
    a = MM.load_annotations(run, tmp_path)
    assert len(a[MM.ROI_LAYER]) == 1 and len(a[MM.ANNOT_LAYER]) == 1 and a[MM.UNSURE_LAYER] == []
    assert a[MM.ANNOT_LAYER][0].area == 30


def test_neighbour_features_alignment_and_edge():
    rng = np.random.default_rng(0)
    n = 400
    xy = np.c_[rng.uniform(0, 400, n), rng.uniform(0, 400, n)]
    ecc = np.full(n, 0.95)
    layer = xy[:, 1] < 40                                    # an aligned layer along the bottom edge
    orient = np.where(layer, 0.0, rng.uniform(-np.pi / 2, np.pi / 2, n))
    f = MM.neighbour_features(xy, ecc, orient, np.zeros(n, bool), radii=(30,))
    assert f.loc[layer & (xy[:, 1] < 15), "shape_align_30"].median() > 0.9
    assert f.loc[~layer & (xy[:, 1] > 100), "shape_align_30"].median() < 0.6
    assert f.loc[xy[:, 1] < 5, "geometry_offset_30"].median() > f.loc[xy[:, 1] > 100, "geometry_offset_30"].median()


def test_evaluate_reports_models_and_rule():
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.integers(0, 2, n)
    tab = pd.DataFrame({"shape_a": y + rng.normal(0, 0.3, n), "markers_b": rng.normal(size=n),
                        "density_c": rng.normal(size=n), "geometry_d": rng.normal(size=n), "y": y,
                        "rule": y, "animal": np.repeat(["A", "B", "C"], n // 3),
                        "section": np.repeat(["A_1", "B_1", "C_1"], n // 3)})
    rep, oof = MM.evaluate(tab)
    assert rep.loc[rep.model == "shape", "auc"].iloc[0] > 0.9
    assert rep.loc[rep.model == "rule (adaptive boundary)", "f1"].iloc[0] == 1.0
    assert oof.notna().all().all() and rep.held_out_by.iloc[0] == "animal"
