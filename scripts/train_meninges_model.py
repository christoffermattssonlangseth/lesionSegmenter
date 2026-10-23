"""Train and evaluate the meninges model from manual MENINGES annotations (``lesionseg.meninges_model``).

    python scripts/train_meninges_model.py --run-dir outputs/sdata [--annot-dir annotations/meninges]

Annotations come from each slide's SpatialData store (shapes layers MENINGES_ROI / MENINGES /
MENINGES_UNSURE) or from ``<slide>_<LAYER>.geojson`` in ``--annot-dir``. Writes to ``--report``:
``labels.csv`` (labelled cells per section / box), ``evaluation.csv`` (held-out metrics per feature set and the
rule), ``by_kind.csv`` (per box kind from ``targets.csv``: where the rule and the model disagree with the
annotation); the model trained on all annotations to ``--model``; ``meninges_pred.parquet`` (p per cell) in
every scene directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lesionseg import meninges_model as MM
from lesionseg import report as R


def box_kinds(tab: pd.DataFrame, targets: Path) -> pd.Series:
    """Kind of the suggested box each annotated box sits on (``other`` for boxes the annotator added)."""
    if not targets.exists():
        return pd.Series("other", index=tab.index)
    t = pd.read_csv(targets)
    kind = pd.Series("other", index=tab.index, dtype=object)
    for r in t.itertuples():
        hit = ((tab["scene"] == r.scene) & ((tab["roi_x_um"] - r.x_um).abs() < 30)
               & ((tab["roi_y_um"] - r.y_um).abs() < 30))
        kind[hit] = r.kind
    return kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--annot-dir", default=None)
    ap.add_argument("--targets", default="results/meninges_annotation/targets.csv")
    ap.add_argument("--report", default="results/meninges_model")
    ap.add_argument("--model", default="models/meninges_hgb.joblib")
    ap.add_argument("--no-predict", action="store_true")
    a = ap.parse_args()
    runs = R.load_runs(a.run_dir)
    tab = MM.training_table(runs, a.annot_dir)
    if tab.empty:
        raise SystemExit("no meninges annotations found: need MENINGES_ROI + MENINGES per slide, in the SpatialData "
                         "store or as <slide>_<LAYER>.geojson in --annot-dir (docs/meninges_annotation.md)")
    out = Path(a.report)
    out.mkdir(parents=True, exist_ok=True)
    tab["kind"] = box_kinds(tab, Path(a.targets))
    lab = (tab.groupby(["scene", "section", "roi", "kind"]).agg(cells=("y", "size"), meninges=("y", "sum"))
           .reset_index())
    lab.to_csv(out / "labels.csv", index=False)
    print(f"{len(tab):,} labelled cells ({int(tab.y.sum()):,} meninges) in {lab.groupby(['scene', 'roi']).ngroups} "
          f"boxes, {tab.section.nunique()} sections, {tab.animal.nunique()} animals")
    rep, oof = MM.evaluate(tab)
    rep.to_csv(out / "evaluation.csv", index=False)
    print(rep.round(3).to_string(index=False))
    best = rep.loc[rep["model"].isin(MM.FEATURE_SETS)].sort_values("avg_precision", ascending=False)["model"].iloc[0]
    rows = []
    for k, g in tab.assign(p=oof[best]).groupby("kind"):
        ok = g["p"].notna()
        rows.append({"kind": k, "cells": len(g), "meninges_share": g.y.mean(),
                     **{f"rule_{m}": v for m, v in MM._binary_metrics(g.y, g.rule.astype(bool)).items()},
                     **{f"model_{m}": v for m, v in MM._binary_metrics(g.y[ok], g.p[ok] >= 0.5).items()}})
    by_kind = pd.DataFrame(rows)
    by_kind.to_csv(out / "by_kind.csv", index=False)
    print(f"\nper box kind (model = '{best}', held out):")
    print(by_kind.round(3).to_string(index=False))
    cols = MM.columns(tab, MM.FEATURE_SETS[best])
    clf = MM.fit(tab[cols], tab["y"])
    Path(a.model).parent.mkdir(parents=True, exist_ok=True)
    MM.save(clf, cols, Path(a.model), meta={"feature_set": best, "sections": sorted(tab.section.unique()),
                                            "cells": len(tab), "annot_dir": a.annot_dir})
    (out / "model.json").write_text(json.dumps({"model": a.model, "feature_set": best, "features": cols}, indent=1))
    if not a.no_predict:
        for r in runs:
            pr = MM.predict(r, Path(a.model))
            pr.to_parquet(r.dir / "meninges_pred.parquet")
            print(f"{r.name}: {np.mean(pr.p_meninges >= 0.5):.1%} of cells predicted meninges")


if __name__ == "__main__":
    main()
