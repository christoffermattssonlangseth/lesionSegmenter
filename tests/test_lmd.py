import numpy as np
import pandas as pd
import pytest

from lesionseg.export import export_lmd


def test_export_lmd(tmp_path):
    pytest.importorskip("lmd")
    cells = pd.DataFrame({
        "cell_id": [0, 1, 2],
        "zone": ["core", "rim", "distal"],
        "pu1_pos": [True, True, True],
        "contour_wkt": ["POLYGON ((0 0, 5 0, 5 5, 0 5, 0 0))", "POLYGON ((10 10, 15 10, 15 15, 10 15, 10 10))",
                        "POLYGON ((20 20, 25 20, 25 25, 20 25, 20 20))"],
    })
    calib = np.array([[0, 0], [1000, 0], [0, 1000]])
    counts = export_lmd(cells, tmp_path / "shapes.xml", calibration_points_px=calib,
                        wells={"core": "A1", "rim": "A2"}, pixel_size_um=0.5)
    assert counts == {"core": 1, "rim": 1}
    assert (tmp_path / "shapes.xml").exists()
