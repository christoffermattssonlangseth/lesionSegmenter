"""Which meningeal regions hold the immune cells? One row per 50 µm surface stretch of the RRMAP2 Xenium atlas
(notebook 15). Needs the outputs of ``scripts/xenium_meninges.py`` (``outputs/xenium_meninges/``).

    python scripts/xenium_meningeal_regions.py --out results/xenium_meningeal_regions

Unit choices: stretch-level tables are descriptive; every test is on sections (paired within section) or
clustered by animal (GEE), never on stretches or cells as if independent.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

from lesionseg import xenium_meninges as X

SHEET = Path(__file__).resolve().parents[2] / "ImmunoTransformer/docs/sample_sheet/sections.csv"
REGIONS = ["dorsal", "lateral", "ventral"]
MODEL_REGIONS = REGIONS + ["root zone"]
MIN_SURFACE_UM = 300.0     # a section × region needs this much traced surface to enter a comparison


def per_section_region(t: pd.DataFrame) -> pd.DataFrame:
    """Per section × region: surface, meningeal / subpial immune cells per 100 µm (all and per immune group),
    share of stretches with a meningeal layer, mean pia depth."""
    cols = {f"men_{g}": (f"men_{g}", "sum") for g in X.IMMUNE_GROUPS if f"men_{g}" in t}
    g = t.groupby(["section", "group", "animal", "spinal_level", "region"]).agg(
        surface_um=("surface_um", "sum"), men_immune=("men_immune", "sum"), sub_immune=("sub_immune", "sum"),
        men_cells=("men_cells", "sum"), layer=("layer", "mean"), pia_um=("pia_depth_um", "mean"),
        stretches=("seg", "size"), **cols).reset_index()
    g["men_immune_share"] = g["men_immune"] / g["men_cells"].where(g["men_cells"] >= 10)
    for k in ["men_immune", "sub_immune"] + list(cols):
        g[f"{k}_per100"] = 100 * g[k] / g["surface_um"]
    return g


def paired_regions(g: pd.DataFrame, value: str = "men_immune_per100", eps: float = 0.1) -> pd.DataFrame:
    """Region pairs within EAE sections: log2 ratio (value + eps), sections and animals (animal = mean of its
    sections) higher, Wilcoxon on sections."""
    e = g[(g["group"] != "CFA control") & (g["surface_um"] >= MIN_SURFACE_UM)]
    w = e.pivot_table(index=["section", "animal"], columns="region", values=value)
    rows = []
    for a, b in [("dorsal", "ventral"), ("dorsal", "lateral"), ("lateral", "ventral")]:
        d = w[[a, b]].dropna()
        lr = np.log2(d[a] + eps) - np.log2(d[b] + eps)
        an = lr.groupby(level="animal").mean()
        rows.append({"value": value, "pair": f"{a} vs {b}", "sections": len(lr), "median_log2_ratio": lr.median(),
                     "sections_higher": int((lr > 0).sum()), "animals": len(an), "animals_higher": int((an > 0).sum()),
                     "wilcoxon_p": wilcoxon(lr).pvalue if len(lr) >= 6 and (lr != 0).any() else np.nan})
    return pd.DataFrame(rows)


def composition(t: pd.DataFrame) -> pd.DataFrame:
    """Share of each immune group among the immune cells of the meninges (``men``) and of the subpial band
    (``sub``), per group (phase) × region (cells pooled within a section, sections averaged; sections with ≥ 10
    immune cells in that compartment and region)."""
    out = []
    for comp in ("men", "sub"):
        cols = [f"{comp}_{g}" for g in X.IMMUNE_GROUPS if f"{comp}_{g}" in t]
        s = t.groupby(["section", "group", "region"])[cols].sum()
        s = s[s.sum(1) >= 10]
        s = s.div(s.sum(1), axis=0)
        s.columns = [c[len(comp) + 1:] for c in s.columns]
        o = s.groupby(["group", "region"]).mean()
        o["sections"] = s.groupby(["group", "region"]).size()
        out.append(o.reset_index().assign(compartment=comp))
    return pd.concat(out, ignore_index=True)


def neighbours(t: pd.DataFrame) -> dict:
    """Adjacent stretches (8-neighbourhood of the 50 µm grid) within each section: {section: (index, i, j)}."""
    out = {}
    for sec, d in t.groupby("section"):
        sx, sy = (d["seg"] % 100000).to_numpy(), (d["seg"] // 100000).to_numpy()
        pos = {(x, y): k for k, (x, y) in enumerate(zip(sx, sy, strict=True))}
        i, j = [], []
        for k, (x, y) in enumerate(zip(sx, sy, strict=True)):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    m = pos.get((x + dx, y + dy))
                    if m is not None and m != k:
                        i.append(k)
                        j.append(m)
        out[sec] = (d.index.to_numpy(), np.array(i, int), np.array(j, int))
    return out


def morans_i(x: np.ndarray, i: np.ndarray, j: np.ndarray) -> float:
    z = x - x.mean()
    den = (z ** 2).sum()
    return np.nan if den == 0 or len(i) == 0 else len(x) / len(i) * (z[i] * z[j]).sum() / den


def clustering(t: pd.DataFrame, nb: dict, n_perm: int = 499, seed: int = 0) -> pd.DataFrame:
    """Per EAE section: Moran's I of meningeal immune cells per 100 µm along the surface vs a permutation null
    (values shuffled over the section's stretches), and the share of meningeal immune cells in the top 10 % of
    stretches."""
    rng = np.random.default_rng(seed)
    rows = []
    for sec, (idx, i, j) in nb.items():
        d = t.loc[idx]
        if d["group"].iloc[0] == "CFA control" or d["men_immune"].sum() < 20 or len(d) < 30:
            continue
        x = (100 * d["men_immune"] / d["surface_um"].clip(lower=10)).to_numpy(float)
        obs = morans_i(x, i, j)
        null = np.array([morans_i(rng.permutation(x), i, j) for _ in range(n_perm)])
        top = d["men_immune"].sort_values(ascending=False)
        rows.append({"section": sec, "group": d["group"].iloc[0], "stretches": len(d),
                     "men_immune": int(d["men_immune"].sum()), "morans_i": obs, "null_mean": null.mean(),
                     "p_perm": (1 + (null >= obs).sum()) / (1 + n_perm),
                     "top10pct_share": top.iloc[:max(1, len(top) // 10)].sum() / top.sum()})
    return pd.DataFrame(rows)


def coupling(t: pd.DataFrame, nb: dict) -> pd.DataFrame:
    """Per EAE section: Spearman between meningeal and subpial immune cells per 100 µm over stretches (same stretch),
    and the share of all near-surface immune cells (meninges + subpial band) that are meningeal."""
    rows = []
    for sec, (idx, _, _) in nb.items():
        d = t.loc[idx]
        if d["group"].iloc[0] == "CFA control" or len(d) < 30:
            continue
        a = d["men_immune"] / d["surface_um"].clip(lower=10)
        b = d["sub_immune"] / d["surface_um"].clip(lower=10)
        rho = spearmanr(a, b).statistic if a.std() > 0 and b.std() > 0 else np.nan
        tot = d["men_immune"].sum() + d["sub_immune"].sum()
        rows.append({"section": sec, "group": d["group"].iloc[0], "spearman_men_sub": rho,
                     "meningeal_share": d["men_immune"].sum() / tot if tot else np.nan,
                     "near_surface_immune": int(tot)})
    return pd.DataFrame(rows)


def gee(t: pd.DataFrame, response: str = "men_immune") -> pd.DataFrame:
    """Poisson GEE with robust (sandwich) errors – valid under overdispersion; a negative-binomial GEE diverged –
    (log link, offset log surface, exchangeable within animal) on EAE stretches with a
    region (dorsal / lateral / ventral / root zone): rate ratios for region, phase, spinal level and vessels in
    the meninges; the meningeal model uses stretches with a meningeal layer, the subpial one adds layer."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    e = t[(t["group"] != "CFA control") & t["region"].isin(MODEL_REGIONS)].copy()
    if response == "men_immune":       # no layer → no meningeal cells by definition: model the layered stretches
        e = e[e["layer"]]
    e = e.dropna(subset=["surface_um"])
    e["vessel"] = (e.get("men_endothelial", 0) + e.get("men_VSMC", 0)) > 0
    e["log_surface"] = np.log(e["surface_um"].clip(lower=10))
    e["phase"] = pd.Categorical(e["group"], [p for p in X.PHASES if p in set(e["group"]) and p != "CFA control"])
    f = (f"{response} ~ C(region, Treatment('ventral')) + C(phase, Treatment('peak')) + C(spinal_level) + vessel"
         + ("" if response == "men_immune" else " + layer"))
    m = smf.gee(f, groups="animal", data=e, offset=e["log_surface"], family=sm.families.Poisson(),
                cov_struct=sm.cov_struct.Exchangeable()).fit()
    ci = m.conf_int()
    return pd.DataFrame({"term": m.params.index, "rate_ratio": np.exp(m.params), "ci_low": np.exp(ci[0]),
                         "ci_high": np.exp(ci[1]), "p": m.pvalues, "response": response, "stretches": len(e),
                         "animals": e["animal"].nunique()}).reset_index(drop=True)


def sectors(t: pd.DataFrame, cells: pd.DataFrame, work: Path, out: Path) -> pd.DataFrame:
    """Geometric dorsal / lateral / ventral sectors (``X.sector_axes``: axis from ventral- to dorsal-niche cells),
    independent of the niche of the cells under a stretch (lesion cells are often 'Mixed'). Per section × sector:
    meningeal, subpial (0–50 µm) and deeper (50–200 µm) immune cells per 100 µm of surface, and over the whole
    sector the share of lesion-niche and of immune cells; paired comparisons within EAE sections."""
    ax = X.sector_axes(cells).set_index("section")
    ax.reset_index().to_csv(out / "sector_axes.csv", index=False)
    t = t[t["section"].isin(ax.index)].copy()
    t["sector"] = ""
    for s, d in t.groupby("section"):
        t.loc[d.index, "sector"] = X.sector((d["seg"] % 100000 + .5) * X.SEG_UM, (d["seg"] // 100000 + .5) * X.SEG_UM,
                                            ax.loc[s])
    t["deep_immune"] = t[[f"deep_{g}" for g in X.IMMUNE_GROUPS if f"deep_{g}" in t]].sum(1)
    t.to_parquet(work / "stretches_sector.parquet", index=False)
    c = cells[cells["section"].isin(ax.index)].copy()
    c["sector"] = ""
    for s, d in c.groupby("section", observed=True):
        c.loc[d.index, "sector"] = X.sector(d["x_centroid"], d["y_centroid"], ax.loc[s])
    c["lesion"] = c["Global_niche_group"].astype(str).str.startswith("Lesion")
    c["immune"] = c["Anno_L1_curated"].isin(X.IMMUNE_L1)
    sc = c.groupby(["section", "sector"], observed=True).agg(cells=("lesion", "size"), lesion=("lesion", "sum"),
                                                             immune=("immune", "sum")).reset_index()
    sc["lesion_frac"], sc["immune_frac"] = sc["lesion"] / sc["cells"], sc["immune"] / sc["cells"]
    g = t.groupby(["section", "group", "animal", "sector"]).agg(
        surface_um=("surface_um", "sum"), men=("men_immune", "sum"), sub=("sub_immune", "sum"),
        deep=("deep_immune", "sum"), pia_um=("pia_depth_um", "mean"), layer=("layer", "mean")).reset_index()
    for k in ("men", "sub", "deep"):
        g[f"{k}_per100"] = 100 * g[k] / g["surface_um"]
    g = g.merge(sc, on=["section", "sector"])
    g.to_csv(out / "section_sector.csv", index=False)
    e = g[(g["group"] != "CFA control") & (g["surface_um"] >= MIN_SURFACE_UM)]
    rows = []
    for v in ("lesion_frac", "immune_frac", "deep_per100", "sub_per100", "men_per100", "pia_um", "layer"):
        w = e.pivot_table(index=["section", "animal"], columns="sector", values=v)
        for a, b in [("ventral", "dorsal"), ("ventral", "lateral"), ("lateral", "dorsal")]:
            d = w[[a, b]].dropna()
            lr = np.log2(d[a] + 0.01) - np.log2(d[b] + 0.01)
            an = lr.groupby(level="animal").mean()
            rows.append({"value": v, "pair": f"{a} vs {b}", "sections": len(lr), "fold": 2 ** lr.median(),
                         "sections_higher": int((lr > 0).sum()), "animals": len(an),
                         "animals_higher": int((an > 0).sum()), "wilcoxon_p": wilcoxon(lr).pvalue})
    pr = pd.DataFrame(rows)
    pr.to_csv(out / "paired_sectors.csv", index=False)
    return pr


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="outputs/xenium_meninges")
    ap.add_argument("--sheet", default=str(SHEET))
    ap.add_argument("--out", default="results/xenium_meningeal_regions")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    sh = pd.read_csv(a.sheet)
    cl = pd.read_parquet(work / "cells_depth.parquet")
    pia = pd.read_parquet(work / "pia.parquet").drop(columns=["animal"], errors="ignore")
    prof = pd.read_parquet(work / "profiles.parquet")
    t = X.stretch_table(cl, pia, prof).merge(
        sh[["section", "animal", "spinal_level", "model", "day_of_sacrifice", "score_at_sacrifice", "sex"]],
        on="section")
    t["region"] = np.where(t["root_zone"], "root zone", t["anatomy"])
    t.to_parquet(work / "stretches.parquet", index=False)
    g = per_section_region(t)
    g.to_csv(out / "section_region.csv", index=False)
    values = ("men_immune_per100", "men_immune_share", "sub_immune_per100", "layer", "pia_um")
    pd.concat([paired_regions(g, v) for v in values]).to_csv(out / "paired_regions.csv", index=False)
    composition(t).to_csv(out / "composition.csv", index=False)
    cells = pd.read_parquet(work / "atlas_cells.parquet").rename(columns={"meta_sample_id": "section"})
    print(sectors(t, cells, work, out).round(3).to_string(index=False))
    nb = neighbours(t)
    cl_ = clustering(t, nb)
    cl_.to_csv(out / "clustering.csv", index=False)
    coupling(t, nb).to_csv(out / "coupling.csv", index=False)
    pd.concat([gee(t, "men_immune"), gee(t, "sub_immune")]).to_csv(out / "gee.csv", index=False)
    print(f"{len(t):,} stretches, {t.section.nunique()} sections; regions:", t.region.value_counts().to_dict())
    print(pd.read_csv(out / "paired_regions.csv").round(3).to_string(index=False))
    print(f"\nclustering: {len(cl_)} EAE sections, Moran's I median {cl_.morans_i.median():.2f} "
          f"(null {cl_.null_mean.median():.2f}), p < 0.05 in {(cl_.p_perm < 0.05).mean():.0%}; "
          f"top 10 % of stretches hold {cl_.top10pct_share.median():.0%} of meningeal immune cells")
    print(pd.read_csv(out / "gee.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
