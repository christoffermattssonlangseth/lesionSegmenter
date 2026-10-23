"""Gene-defined meninges in the RRMAP2 Xenium 5k EAE spinal cord atlas: where is the pia, and how deep is meningeal
vs subpial infiltrate? (notebook 14)

    # on the remote, where the raw Xenium output is (env sq):
    python scripts/xenium_transcript_rasters.py --regions capture_regions.csv --out rasters -j 4
    # here:
    python scripts/xenium_meninges.py --rasters outputs/xenium_rasters --out results/xenium_meninges

Inputs: the 5 µm transcript rasters, the curated atlas (``$RRMAP2_H5AD``; only ``obs`` is read and cached as
``outputs/xenium_meninges/atlas_cells.parquet``) and the RRMAP2 sample sheet (``sections.csv``).
Writes small summary tables to ``--out`` and the per-cell / per-stretch tables to ``outputs/xenium_meninges/``.
"""
from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from lesionseg import xenium_meninges as X

ATLAS = os.environ.get("RRMAP2_H5AD", os.path.expanduser(
    "~/Downloads/RRMAP2_xenium_all_samples.cellcharter.companion.ready.with_metadata."
    "rerun.with_AnnoL1Curated_with_Region_Anno2to4Updated.h5ad"))
SHEET = Path(__file__).resolve().parents[2] / "ImmunoTransformer/docs/sample_sheet/sections.csv"
OBS = ["cell_id", "xenium_output", "meta_sample_id", "x_centroid", "y_centroid", "Anno_L1_curated", "Anno_L2",
       "Global_niche_group", "condition", "stage"]
EXCLUDED = ["C2_G3_Mid_1"]          # dropped in most RRMAP2 analyses
DEPTH_BINS = [0, 10, 20, 30, 50, 80, 150]


def atlas_cells(cache: Path) -> pd.DataFrame:
    if not cache.exists():
        import anndata as ad

        a = ad.read_h5ad(ATLAS, backed="r")
        a.obs[OBS].to_parquet(cache)
    return pd.read_parquet(cache).rename(columns={"meta_sample_id": "section"})


def gene_specificity(out: Path, n: int = 3000, seed: int = 0) -> pd.DataFrame:
    """Detection rate (fraction of cells with ≥ 1 count) per gene in meningeal fibroblasts, lesion (EAE-associated)
    fibroblasts, perineural fibroblasts, CAMs, VSMCs and a sample of all other cells – the basis of the gene groups."""
    f = out / "fb_gene_detection.csv"
    if f.exists():
        return pd.read_csv(f, index_col=0)
    import anndata as ad

    a = ad.read_h5ad(ATLAS, backed="r")
    lab = a.obs["Anno_L2"].astype(str).to_numpy()
    rng = np.random.default_rng(seed)
    groups = {"meningeal_fb": X.MENINGEAL_FB, "eae_associated_fb": ["EAE - Associated FBs"],
              "perineural_fb": ["Perineural FBs"], "cam": ["CAM"], "vsmc": ["VSMC"]}
    pick = {k: rng.choice(np.flatnonzero(np.isin(lab, v)), n, replace=False) for k, v in groups.items()}
    rest = np.flatnonzero(~np.isin(lab, sum(groups.values(), [])))
    pick["all_other"] = rng.choice(rest, 20000, replace=False)
    det = {k: np.asarray((a.X[np.sort(ix)] > 0).mean(0)).ravel() for k, ix in pick.items()}
    d = pd.DataFrame(det, index=a.var_names)
    d["vs_other"] = d["meningeal_fb"] / (d["all_other"] + 0.01)
    d["vs_eae_fb"] = d["meningeal_fb"] / (d["eae_associated_fb"] + 0.01)
    d.to_csv(f)
    return d


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--rasters", default="outputs/xenium_rasters")
    ap.add_argument("--sheet", default=str(SHEET))
    ap.add_argument("--out", default="results/xenium_meninges")
    ap.add_argument("--work", default="outputs/xenium_meninges")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    sh = pd.read_csv(a.sheet)
    sh = sh[~sh["section"].isin(EXCLUDED)]
    sh["group"] = np.where(sh["condition"] == "CONTROL", "CFA control", sh["phase"])
    cells = atlas_cells(work / "atlas_cells.parquet")
    gene_specificity(out)
    prof, cl = X.run(a.rasters, cells, sh)
    prof = prof.merge(sh[["section", "group"]], on="section")
    pia = X.pia_depth(prof, X.section_refs(prof)).merge(sh[["section", "group", "animal"]], on="section")
    cl = cl.merge(sh[["section", "group", "animal"]], on="section")
    prof.to_parquet(work / "profiles.parquet", index=False)
    pia.to_parquet(work / "pia.parquet", index=False)
    cl.to_parquet(work / "cells_depth.parquet", index=False)

    # transcript shares vs depth (per 1000 transcripts)
    g = prof.groupby(["group", "depth"], as_index=False)[X.GROUPS + ["pixels"]].sum()
    for k in X.GROUPS[1:]:
        g[f"{k}_per_1000"] = 1000 * g[k] / g["total"].clip(lower=1)
    g.to_csv(out / "depth_profiles.csv", index=False)
    # pia depth per stretch
    n_edge = prof[prof["depth"] == prof["depth"].min()].groupby("section")["seg"].nunique()
    cov = (pia.groupby("section").size() / n_edge).rename("stretches_with_pia").reset_index()
    cov.to_csv(out / "section_coverage.csv", index=False)
    summ = pia.groupby("group")["pia_depth_um"].describe(percentiles=[.1, .25, .5, .75, .9])
    summ["no_meningeal_layer"] = pia.groupby("group")["pia_depth_um"].apply(lambda v: (v == 0).mean())
    summ.reindex(X.PHASES).to_csv(out / "pia_depth_summary.csv")
    # cells above / below the pia, both definitions
    c = cl.merge(pia[["section", "seg", "pia_depth_um", "pia_glia_um"]], on=["section", "seg"], how="inner")
    c["above_balance"] = c["depth"] <= c["pia_depth_um"]
    c["above_glia"] = c["depth"] <= c["pia_glia_um"]
    # the clean question: where a meningeal layer exists, which side of it is the infiltrate on?
    c["above_present"] = c["above_balance"].where(c["pia_depth_um"] > 0)
    c["cell_class"] = np.select([c["Anno_L2"].isin(X.MENINGEAL_FB), c["Anno_L1_curated"].isin(X.IMMUNE_L1)],
                                ["meningeal FB", "immune"], "other")
    c = c[c["depth"] <= DEPTH_BINS[-1]]
    c["depth_bin"] = pd.cut(c["depth"], DEPTH_BINS).astype(str)
    long = c.melt(id_vars=["section", "group", "animal", "cell_class", "depth_bin"],
                  value_vars=["above_balance", "above_glia", "above_present"], var_name="definition",
                  value_name="above_pia")
    long["definition"] = long["definition"].str.replace("above_", "")
    keys = ["definition", "group", "cell_class", "depth_bin"]
    long = long.dropna(subset=["above_pia"])
    ab = long.groupby(keys)["above_pia"].agg(["mean", "size"]).rename(columns={"mean": "above_pia", "size": "cells"})
    ab.reset_index().to_csv(out / "above_pia_by_depth.csv", index=False)
    # per section (the unit for any test) and per animal (for our animals' Xenium twins), immune cells only
    imm = long[long["cell_class"] == "immune"]
    for unit in ("section", "animal"):
        (imm.groupby(["definition", unit, "group", "depth_bin"])["above_pia"].agg(["mean", "size"])
         .rename(columns={"mean": "above_pia", "size": "cells"}).reset_index()
         .to_csv(out / f"immune_above_pia_by_{unit}.csv", index=False))
    # sensitivity of the balance definition to MIN_MENINGEAL
    ref, rows, keep = X.section_refs(prof), [], X.MIN_MENINGEAL
    for t in (0.0, 0.05, 0.1, 0.2, 0.3):
        X.MIN_MENINGEAL = t
        q = X.pia_depth(prof, ref)
        e = cl.merge(q[["section", "seg", "pia_depth_um"]], on=["section", "seg"])
        e = e[e["depth"] <= DEPTH_BINS[-1]]
        e["above"] = e["depth"] <= e["pia_depth_um"]
        e["depth_bin"] = pd.cut(e["depth"], DEPTH_BINS).astype(str)
        fb, im = e[e["Anno_L2"].isin(X.MENINGEAL_FB)], e[e["Anno_L1_curated"].isin(X.IMMUNE_L1)
                                                          & (e["group"] != "CFA control")]
        rows += [{"min_meningeal": t, "cells": "meningeal FB", **fb.groupby("depth_bin")["above"].mean()},
                 {"min_meningeal": t, "cells": "immune (EAE)", **im.groupby("depth_bin")["above"].mean()}]
    X.MIN_MENINGEAL = keep
    pd.DataFrame(rows).to_csv(out / "sensitivity_min_meningeal.csv", index=False)
    print(f"{prof.section.nunique()} sections, {len(pia):,} stretches with a pia, {len(c):,} cells placed")
    print(ab.xs("immune", level="cell_class")["above_pia"].unstack("group").reindex(columns=X.PHASES).round(2)
          .to_string())

if __name__ == "__main__":
    main()
