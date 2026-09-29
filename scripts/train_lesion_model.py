"""Train the supervised lesion model from a finished run (maps + cells) and the manual CORE polygons.

    python scripts/train_lesion_model.py --run-dir outputs/sdata_unrestricted --out models

Writes one leave-one-scene-out model per scene (``lesion_hgb_loso_excl_<scene>.joblib``) plus a
model trained on all scenes (``lesion_hgb_all.joblib``) and ``training_report.csv``.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from lesionseg import model as M
from lesionseg import sdata, validate
from lesionseg.density import Grid


def load_scene(logf: Path):
    log = json.load(open(logf))
    d = logf.parent
    meta = json.load(open(d / "maps/maps.json"))
    bin_um, px = meta["bin_um"], meta["pixel_size_um"]
    tissue = tifffile.imread(d / "maps/tissue.tif").astype(bool)
    sections = tifffile.imread(d / "maps/sections.tif")
    grid = Grid(bin_um, tissue.shape, px)
    cells = pd.read_parquet(d / "cells.parquet")
    manual = sdata.load_manual_annotations(log["cells"]["zarr"])
    core = validate.rasterize(manual["CORE"]["geometry"], grid, px)
    core_secs = validate.sections_with_cores(manual["CORE"]["geometry"], sections, grid, px)
    sf = M.build_features(cells, grid, tissue, sections)
    sf.name = f"{log['sample']}_scene{log['scene']}"
    lab = M.make_labels(core, sections, core_secs, tissue, grid)
    return dict(name=sf.name, sf=sf, lab=lab, core=core, manual=manual, grid=grid, px=px)


def xy(scenes):
    X = np.concatenate([s["sf"].X[s["lab"].ravel()[s["sf"].idx] >= 0] for s in scenes])
    y = np.concatenate([s["lab"].ravel()[s["sf"].idx][s["lab"].ravel()[s["sf"].idx] >= 0] for s in scenes])
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata_unrestricted")
    ap.add_argument("--out", default="models")
    ap.add_argument("--p-low", type=float, default=0.25)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    scenes = [load_scene(Path(f)) for f in sorted(glob.glob(f"{args.run_dir}/*/scene*/run_log.json"))]
    from sklearn.metrics import average_precision_score, roc_auc_score

    rows = []
    for held in scenes:
        X, y = xy([s for s in scenes if s is not held])
        clf = M.train(X, y)
        single = held["name"] in ("CML_1_scene0", "CML_2_rescan_scene0")
        name = held["name"].replace("_scene0", "") if single else held["name"]
        M.save_model(clf, held["sf"].names, out / f"lesion_hgb_loso_excl_{name}.joblib",
                     meta={"excluded": held["name"], "trained_on": [s["name"] for s in scenes if s is not held]})
        p = M.predict_map(clf, held["sf"])
        yl = held["lab"].ravel()[held["sf"].idx]
        pv = p.ravel()[held["sf"].idx]
        ok = yl >= 0
        m = p > args.p_low
        det = [(validate.rasterize([g], held["grid"], held["px"]) & m).sum()
               / max(validate.rasterize([g], held["grid"], held["px"]).sum(), 1) > 0.5
               for g in held["manual"]["CORE"]["geometry"]]
        rows.append({"held_out": held["name"], "auc": roc_auc_score(yl[ok], pv[ok]),
                     "avg_precision": average_precision_score(yl[ok], pv[ok]), "n_cores": len(det),
                     "cores_detected_frac": float(np.mean(det))})
    X, y = xy(scenes)
    M.save_model(M.train(X, y), scenes[0]["sf"].names, out / "lesion_hgb_all.joblib",
                 meta={"trained_on": [s["name"] for s in scenes]})
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "training_report.csv", index=False)
    print(rep.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
