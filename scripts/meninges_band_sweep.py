"""How many Pu.1+ cells (and how many manual-core cells) fall within a given distance of the true
tissue surface – the trade-off behind ``parenchyma.erode_um``.

    python scripts/meninges_band_sweep.py --run-dir outputs/sdata
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from lesionseg.report import load_runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--bands", default="0,25,30,40,50,60,75,100")
    a = ap.parse_args()
    bands = [float(b) for b in a.bands.split(",")]
    rows = []
    for r in load_runs(a.run_dir):
        if not (r.dir / "maps" / "surface.tif").exists():
            continue
        surf = r.map("surface").astype(bool)
        g = r.grid
        d_surf = ndi.distance_transform_edt(surf) * g.bin_um
        c = r.cells
        gx, gy = g.to_grid(c.x_um.to_numpy(), c.y_um.to_numpy())
        nr, nc = g.shape
        d = ndi.map_coordinates(d_surf, np.vstack([np.clip(gy, 0, nr - 1), np.clip(gx, 0, nc - 1)]), order=1,
                                mode="nearest")
        pos = c.pu1_pos.to_numpy(bool) & c.section_has_lesion.to_numpy(bool)
        mc = pos & (c["manual_core_id"].to_numpy() > 0) if "manual_core_id" in c else np.zeros(len(c), bool)
        for w in bands:
            band = d < w
            rows.append({"scene": r.name, "band_um": w, "n_pu1_in_band": int((band & pos).sum()),
                         "pct_pu1_lesion_sections": round(100 * (band & pos).sum() / max(pos.sum(), 1), 1),
                         "n_manual_core_pu1_in_band": int((band & mc).sum()),
                         "pct_manual_core_lost": round(100 * (band & mc).sum() / max(mc.sum(), 1), 1)})
    df = pd.DataFrame(rows)
    out = Path(a.run_dir) / "cohort" / "meninges_band_sweep.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False)
    print(df.pivot_table(index="band_um", columns="scene", values="pct_manual_core_lost").round(1).to_string())


if __name__ == "__main__":
    main()
