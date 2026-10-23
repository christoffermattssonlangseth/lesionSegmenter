"""Meninges model without annotations: confident labels, position-free features, and a virtual-cut leak test.

    python scripts/meninges_weak_labels.py --run-dir outputs/sdata --out results/meninges_weak

Until the meninges are annotated (docs/meninges_annotation.md) the only labels are the rule's. Here only the
*confident* ones are used: 1 = outer ≤ 10 µm of the surface (meninges by any definition), 0 = parenchyma ≥ 150 µm
deep outside manual lesion cores; everything in between is left out of training. Models see no position, so
what they say in between is their own opinion. Checks, each animal predicted by a model trained on the others:

* ``outer vs deep``   – AUC on the confident labels (easy; mostly a sanity check);
* ``outer vs cut``    – **virtual cut**: in deep parenchyma (≥ 300 µm from any surface) all cells on one side of a
  random line are deleted and the features recomputed as if that were a surface. A model that learned "edge"
  calls these cells meninges; AUC of real outer meninges vs cells ≤ 10 µm from a cut;
* ``outer vs core``   – manual lesion-core cells ≤ 30 µm from the surface (the collaborator's annotation, the only
  independent negatives at the surface): meninges vs subpial lesion, the case that decides what is collected.

Writes ``evaluation.csv``, ``virtual_cut_profile.csv`` (p vs distance to the cut), ``depth_profile.csv``
(p of the chosen model vs depth, by lesion context) and ``cell_predictions.parquet`` (held-out p of every feature
set per cell; read by notebook 13).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from sklearn.metrics import roc_auc_score

from lesionseg import meninges_model as MM
from lesionseg import orientation as O
from lesionseg import report as R

SETS = {
    "shape": ("shape",),
    "shape + markers": ("shape", "markers"),
    "shape + markers + image": ("shape", "markers", "image"),
    "shape + markers + density": ("shape", "markers", "density"),
    "shape + markers + density + image": ("shape", "markers", "density", "image"),
}
CUT_WIN_UM, CUT_EVAL_UM = 260.0, 150.0   # cells kept within the window; scored within the inner radius


def scene_tables(r, n_cuts: int, rng) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(all in-section cells with features + bookkeeping, cells near virtual cuts with recomputed features)."""
    from lesionseg.report import surface_tiers

    c = r.cells
    f = MM.cell_features(r)
    tier = c["surface_tier"].astype(str).to_numpy()
    d = c["depth_um"].to_numpy(float)
    core = (c["manual_core_id"] > 0).to_numpy() if "manual_core_id" in c else np.zeros(len(c), bool)
    sec = c["section_name"].astype(str)
    y = np.full(len(c), -1)
    y[(tier == "meninges") & (d <= 10)] = 1
    y[(tier == "parenchyma") & (d >= 150) & ~core] = 0
    keep = (c["section_id"] > 0).to_numpy()
    dles = c["dist_to_lesion_um"].replace(np.inf, np.nan).to_numpy(float)
    ctx = np.where(~c["section_has_lesion"].to_numpy(bool), "control section",
                   np.where(dles <= 100, "≤ 100 µm from lesion", "far from lesion"))
    D = f[keep].assign(label=c["label"].to_numpy()[keep], x_um=c["x_um"].to_numpy()[keep],
                       y_um=c["y_um"].to_numpy()[keep], y=y[keep], tier=tier[keep], depth=d[keep], core=core[keep],
                       ctx=ctx[keep], scene=r.name,
                       section=sec[keep], animal=sec[keep].str.rsplit("_", n=1).str[0])
    tiers, res = surface_tiers(r)
    tissue = (tiers > 0).astype(np.float32)
    orient = c[["label"]].merge(O.nucleus_orientation(r), on="label", how="left")["orient_rad"].to_numpy()
    xy = c[["x_um", "y_um"]].to_numpy(float)
    cand = np.flatnonzero((tier == "parenchyma") & (d >= 300) & ~core)
    cuts = []
    m = int((CUT_WIN_UM + 70) / res)
    for k in rng.choice(cand, min(n_cuts, len(cand)), replace=False):
        cen = xy[k]
        th = rng.uniform(0, np.pi)
        nrm = np.array([np.cos(th), np.sin(th)])
        kept = np.flatnonzero((np.hypot(*(xy - cen).T) < CUT_WIN_UM) & ((xy - cen) @ nrm >= 0))
        r0, c0 = int(cen[1] / res) - m, int(cen[0] / res) - m
        if r0 < 0 or c0 < 0 or r0 + 2 * m > tissue.shape[0] or c0 + 2 * m > tissue.shape[1]:
            continue
        nf = MM.neighbour_features(xy[kept], c["eccentricity"].to_numpy(float)[kept], orient[kept],
                                   c["pu1_pos"].to_numpy(bool)[kept])
        yy, xx = np.mgrid[r0:r0 + 2 * m, c0:c0 + 2 * m]
        t = tissue[r0:r0 + 2 * m, c0:c0 + 2 * m] * (
            (((xx + .5) * res - cen[0]) * nrm[0] + ((yy + .5) * res - cen[1]) * nrm[1]) >= 0)
        iy = ((xy[kept, 1] / res).astype(int) - r0).clip(0, t.shape[0] - 1)
        ix = ((xy[kept, 0] / res).astype(int) - c0).clip(0, t.shape[1] - 1)
        g = f.iloc[kept].copy()
        for col in nf:
            g[col] = nf[col].to_numpy()
        for rr in MM.RADII_UM:
            frac = fftconvolve(t, MM._disk(rr / res), mode="same")[iy, ix].clip(0.05, 1)
            g[f"density_{rr}"] = nf[f"geometry_n_{rr}"].to_numpy() / (np.pi * rr ** 2 * 1e-6 * frac)
        # image features over the cut tissue mask (same definitions as MM.image_features, on this crop)
        g = _cut_image_features(r, g, t, r0, c0, iy, ix, res)
        inner = np.hypot(*(xy[kept] - cen).T) < CUT_EVAL_UM
        cuts.append(g[inner].assign(dist_cut=((xy[kept] - cen) @ nrm)[inner], scene=r.name,
                                    animal=sec.iloc[kept][inner].str.rsplit("_", n=1).str[0].to_numpy(),
                                    cut=f"{r.name}:{k}"))
    return D, (pd.concat(cuts, ignore_index=True) if cuts else pd.DataFrame())


_IMG: dict = {}


def _cut_image_features(r, g, t, r0, c0, iy, ix, res):
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu

    from lesionseg.report import surface_tiers

    if r.name not in _IMG:   # full overview channels + the scene-wide constants used by MM.image_features
        tiers, _ = surface_tiers(r)
        tissue = tiers > 0
        H, W = tissue.shape
        rd = r.open_image()

        def fit(a):
            a = a[:H, :W].astype(np.float32)
            return np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])))

        imgs = {k: fit(rd.read_overview(r.scene, r.px / res, rd.channel_index(v)))
                for k, v in MM.IMAGE_CHANNELS.items() if v in rd.channel_names}
        thr = threshold_otsu(imgs["nuc"][tissue])
        bgm = tissue & ~(imgs["nuc"] > thr)
        _IMG[r.name] = (imgs, thr, {k: float(np.median(im[bgm])) or 1.0 for k, im in imgs.items()})
    imgs, thr, ref = _IMG[r.name]
    sl = (slice(r0, r0 + t.shape[0]), slice(c0, c0 + t.shape[1]))
    nuc = (imgs["nuc"][sl] > thr) & (t > 0)
    bg = ((t > 0) & ~nuc).astype(np.float32)
    for s in MM.IMAGE_SIGMAS_UM:
        gf = lambda a, s=s: ndi.gaussian_filter(a, s / res)[iy, ix]   # noqa: E731
        wt, wb = np.maximum(gf(t), 1e-6), np.maximum(gf(bg), 1e-6)
        g[f"image_nucfrac_{s:g}"] = gf(nuc.astype(np.float32)) / wt
        m1, m2 = gf(imgs["nuc"][sl] * t) / wt, gf(imgs["nuc"][sl] ** 2 * t) / wt
        g[f"image_nuc_sd_{s:g}"] = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
        for k, im in imgs.items():
            g[f"image_{k}_bg_{s:g}"] = gf(im[sl] * bg) / wb / ref[k]
    return g


def loao(D, C, cols, seed=0):
    """Leave-one-animal-out probabilities for all cells of D and of C (negatives subsampled to 5× positives)."""
    pD, pC = np.full(len(D), np.nan), np.full(len(C), np.nan)
    for a in D["animal"].unique():
        tr = D[(D["animal"] != a) & (D["y"] >= 0)]
        neg = tr[tr["y"] == 0]
        tr = pd.concat([tr[tr["y"] == 1], neg.sample(min(len(neg), 5 * int((tr["y"] == 1).sum())), random_state=seed)])
        m = MM.fit(tr[cols], tr["y"])
        tD, tC = (D["animal"] == a).to_numpy(), (C["animal"] == a).to_numpy()
        pD[tD] = m.predict_proba(D.loc[tD, cols])[:, 1]
        if tC.any():
            pC[tC] = m.predict_proba(C.loc[tC, cols])[:, 1]
    return pD, pC


def auc(pos, neg):
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    return roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg])


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--out", default="results/meninges_weak")
    ap.add_argument("--cuts", type=int, default=25, help="virtual cuts per scene")
    ap.add_argument("--model", default="shape + markers", help="feature set for the depth profile")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    parts = [scene_tables(r, a.cuts, rng) for r in R.load_runs(a.run_dir)]
    D = pd.concat([p[0] for p in parts], ignore_index=True)
    C = pd.concat([p[1] for p in parts], ignore_index=True)
    outer = (D["y"] == 1).to_numpy()
    core_s = (D["core"] & (D["depth"] <= 30)).to_numpy()
    cut10 = (C["dist_cut"] <= 10).to_numpy()
    print(f"{len(D):,} cells; confident {outer.sum():,} outer / {(D.y == 0).sum():,} deep; "
          f"{C.cut.nunique()} virtual cuts ({cut10.sum()} cells ≤ 10 µm); {core_s.sum():,} core cells ≤ 30 µm deep")
    rows, prof, keep_p = [], [], None
    preds = D[["scene", "section", "animal", "label", "x_um", "y_um", "depth", "tier", "core", "ctx", "y"]].copy()
    bins = [0, 10, 20, 30, 40, 60, 80, 150]
    for name, groups in SETS.items():
        cols = MM.columns(D, groups)
        pD, pC = loao(D, C, cols)
        conf = (D["y"] >= 0).to_numpy()
        rows.append({"features": name, "outer_vs_deep_auc": roc_auc_score(D["y"][conf], pD[conf]),
                     "outer_vs_cut_edge_auc": auc(pD[outer], pC[cut10]),
                     "outer_vs_core_at_surface_auc": auc(pD[outer], pD[core_s]),
                     "cut_edge_called_pct": 100 * np.nanmean(pC[cut10] >= 0.5),
                     "outer_called_pct": 100 * np.nanmean(pD[outer] >= 0.5),
                     "core_at_surface_called_pct": 100 * np.nanmean(pD[core_s] >= 0.5)})
        prof.append(C.assign(p=pC, features=name, db=pd.cut(C["dist_cut"], bins, include_lowest=True))
                    .groupby(["features", "db"], observed=True)["p"].agg(["mean", "size"]).reset_index())
        preds[f"p_{name}"] = pD
        if name == a.model:
            keep_p = pD
    preds.to_parquet(out / "cell_predictions.parquet", index=False)
    rep = pd.DataFrame(rows)
    rep.to_csv(out / "evaluation.csv", index=False)
    print(rep.round(3).to_string(index=False))
    cp = pd.concat(prof)
    cp["db"] = cp["db"].astype(str)
    cp.to_csv(out / "virtual_cut_profile.csv", index=False)
    print("\nmean p vs distance to a virtual cut:")
    print(cp.pivot_table(index="db", columns="features", values="mean", sort=False).round(2).to_string())
    dbins = [0, 10, 20, 30, 40, 60, 80, 120, 160, 250, 400]
    dp = (D.assign(p=keep_p, db=pd.cut(D["depth"], dbins, include_lowest=True),
                   group=np.where(D["core"], "manual lesion core", D["ctx"]))
          .groupby(["group", "db"], observed=True)["p"].agg(["mean", "size"]).reset_index())
    dp["db"] = dp["db"].astype(str)
    dp.to_csv(out / "depth_profile.csv", index=False)
    print(f"\nmean p ('{a.model}') vs depth below the surface:")
    print(dp.pivot_table(index="db", columns="group", values="mean", sort=False).round(2).to_string())


if __name__ == "__main__":
    main()
