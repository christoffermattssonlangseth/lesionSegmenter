"""Per-cell marker expression by lesion proximity (exploratory).

Question: do Pu.1⁺ cells in the lesion core express more Pu.1 (and Iba1, smaller / rounder nuclei) than
cells in the rim, the peri-lesion area and further out?

Pitfalls this module is built around:

* **the unit is the section, not the cell** – tens of thousands of cells make any difference "significant";
  comparisons use per-section medians, paired within a section, replicate slides averaged
  (:func:`section_medians`, :func:`paired_tests`);
* **scan / staining differences** – intensities are compared as log2 ratios to the same section's distal
  Pu.1⁺ cells (paired), or to the scene's control-section Pu.1⁺ cells (``*_norm`` columns);
* **crowding** – in the packed core, signal from touching neighbours can raise a nucleus' mean intensity;
  ``n_nb_15um`` (nuclei within 15 µm) lets zones be compared at matched density (:func:`density_matched`);
* **truncation** – Pu.1⁺ cells were called by a threshold, so their intensities are a truncated distribution;
  the Pu.1⁺ fraction per zone is reported alongside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wilcoxon

LESION_ORDER = ["core", "rim", "peri", "deep", "distal"]
ORDER = LESION_ORDER + ["meninges", "GM", "WM"]
MARKERS = {"Pu1_mean": "Pu.1 (nuclear mean)", "Iba1_mean": "Iba1 (nuclear mean)",
           "area_um2": "nuclear area (µm²)", "eccentricity": "nuclear eccentricity"}


def compartment(cells: pd.DataFrame) -> pd.Series:
    """core / rim / peri / deep / distal in lesion sections, GM / WM (manual) in control sections, meninges
    anywhere; buffer cells and the rest are None."""
    zone = cells["zone"].astype(str)
    blank = pd.Series("", index=cells.index)
    tier = cells["surface_tier"].astype(str) if "surface_tier" in cells else blank.replace("", "parenchyma")
    les = cells["section_has_lesion"].to_numpy(bool)
    manual = cells["manual_annotation"].astype(str) if "manual_annotation" in cells else blank
    par = (tier == "parenchyma").to_numpy()
    out = pd.Series(None, index=cells.index, dtype=object)
    for z in LESION_ORDER:
        out[les & par & (zone == z).to_numpy()] = z
    for m in ("GM", "WM"):
        out[~les & par & (manual == m).to_numpy()] = m
    out[(tier == "meninges").to_numpy()] = "meninges"
    if "in_vbo" in cells:
        out[cells["in_vbo"].to_numpy(bool)] = None
    return out


def expression_table(runs, radius_um: float = 15.0) -> pd.DataFrame:
    """One row per cell with a compartment: markers, normalised Pu.1 / Iba1, distances, local density."""
    frames = []
    for r in runs:
        c = r.cells
        comp = compartment(c)
        xy = c[["x_um", "y_um"]].to_numpy(float)
        tree = cKDTree(xy)
        n_nb = tree.query_ball_point(xy, r=radius_um, return_length=True) - 1
        pos = c["pu1_pos"].to_numpy(bool)
        n_pos_nb = cKDTree(xy[pos]).query_ball_point(xy, r=radius_um, return_length=True) - pos.astype(int)
        d = pd.DataFrame({
            "scene": r.name, "section": c["section_name"].astype(str), "compartment": comp,
            "pu1_pos": pos, "x_um": c["x_um"], "y_um": c["y_um"],
            "dist_to_lesion_um": c["dist_to_lesion_um"].replace(np.inf, np.nan),
            "dist_to_lesion_rel": (c["dist_to_lesion_rel"].replace(np.inf, np.nan) if "dist_to_lesion_rel" in c
                                   else np.nan),
            f"n_nb_{radius_um:g}um": n_nb, f"n_pu1_nb_{radius_um:g}um": n_pos_nb,
            **{k: c[k].to_numpy(float) for k in MARKERS if k in c},
        })
        d = d[d["compartment"].notna()]
        # scene baseline: Pu.1⁺ cells of the control sections (GM + WM)
        base = d[d["pu1_pos"] & d["compartment"].isin(["GM", "WM"])]
        for k in ("Pu1_mean", "Iba1_mean"):
            if k in d and len(base):
                d[f"{k}_norm"] = d[k] / base[k].median()
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out["compartment"] = pd.Categorical(out["compartment"], categories=ORDER, ordered=True)
    return out


def section_medians(df: pd.DataFrame, value: str, pu1_only: bool = True, min_cells: int = 20,
                    ref: str = "distal") -> pd.DataFrame:
    """Median ``value`` per scene × section × compartment, plus ``log2_vs_ref`` = log2(median / median of
    the same section's ``ref`` compartment). Compartments with fewer than ``min_cells`` cells are dropped."""
    d = df[df["pu1_pos"]] if pu1_only else df
    g = (d.groupby(["scene", "section", "compartment"], observed=True)[value]
         .agg(median="median", n="size").reset_index())
    g = g[g["n"] >= min_cells]
    refs = g[g["compartment"] == ref].set_index(["scene", "section"])["median"]
    g["log2_vs_ref"] = np.log2(g["median"] / g.set_index(["scene", "section"]).index.map(refs).to_numpy())
    return g


def paired_tests(med: pd.DataFrame, ref: str = "distal") -> pd.DataFrame:
    """Per compartment vs ``ref``: replicate slides of a section averaged (unit = section), then a
    two-sided Wilcoxon signed-rank test of log2(compartment / ref) against 0."""
    per_sec = (med.dropna(subset=["log2_vs_ref"]).groupby(["section", "compartment"], observed=True)["log2_vs_ref"]
               .mean().reset_index())
    rows = []
    for comp, s in per_sec.groupby("compartment", observed=True):
        if comp == ref:
            continue
        v = s["log2_vs_ref"].to_numpy()
        p = wilcoxon(v).pvalue if len(v) >= 5 and np.any(v != 0) else np.nan
        rows.append({"compartment": comp, "n_sections": len(v), "median_log2_vs_" + ref: np.median(v),
                     "fold_change": 2 ** np.median(v), "sections_higher": int((v > 0).sum()), "wilcoxon_p": p})
    return pd.DataFrame(rows)


def gradient(df: pd.DataFrame, value: str, edges_um=(-300, -200, -150, -100, -50, -25, 0, 25, 50, 100, 150,
                                                       200, 300, 500, 800), pu1_only: bool = True) -> pd.DataFrame:
    """Median ``value`` per scene × signed-distance bin (negative = inside the lesion)."""
    d = df[df["pu1_pos"] & df["dist_to_lesion_um"].notna()] if pu1_only else df[df["dist_to_lesion_um"].notna()]
    d = d[d["compartment"].isin(LESION_ORDER)]
    b = pd.cut(d["dist_to_lesion_um"], list(edges_um))
    g = d.groupby(["scene", b], observed=True)[value].agg(median="median", q25=lambda x: x.quantile(0.25),
                                                         q75=lambda x: x.quantile(0.75), n="size").reset_index()
    g["mid_um"] = g["dist_to_lesion_um"].map(lambda iv: iv.mid).astype(float)
    return g[g["n"] >= 20]


def density_matched(df: pd.DataFrame, value: str, density_col: str = "n_nb_15um", n_bins: int = 4,
                    compartments=("core", "rim", "peri", "distal")) -> pd.DataFrame:
    """Median ``value`` of Pu.1⁺ cells per compartment within quantile bins of local nuclear density
    (bins from all Pu.1⁺ lesion-section cells): does the core stay higher at matched crowding?"""
    d = df[df["pu1_pos"] & df["compartment"].isin(compartments)].copy()
    d["density_bin"] = pd.qcut(d[density_col], n_bins, duplicates="drop")
    return (d.groupby(["density_bin", "compartment"], observed=True)[value].median().unstack("compartment")
            .reindex(columns=[c for c in compartments]))


# --------------------------------------------------------------------------
# figures (report notebook 07)
# --------------------------------------------------------------------------
COLORS = {"core": "#e66767", "rim": "#c98500", "peri": "#199e70", "deep": "#9085e9", "distal": "#898781",
          "meninges": "#c3c2b7", "GM": "#3987e5", "WM": "#52514e"}
SCENE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def plot_paired(med: pd.DataFrame, ax, title: str, comps=("core", "rim", "peri", "deep", "distal", "meninges")):
    """log2(compartment / distal) per section (replicate slides averaged): one grey line per section, dots
    coloured by compartment, black bar = median over sections."""
    per = (med.groupby(["section", "compartment"], observed=True)["log2_vs_ref"].mean().unstack("compartment")
           .reindex(columns=[c for c in comps]))
    x = np.arange(len(comps))
    for _, row in per.iterrows():
        ax.plot(x, row.to_numpy(float), color="#c3c2b7", lw=1, zorder=1)
    for i, c in enumerate(comps):
        v = per[c].dropna()
        ax.scatter(np.full(len(v), i), v, s=36, color=COLORS[c], edgecolors="white", linewidths=0.8, zorder=3)
        if len(v):
            ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#0b0b0b", lw=2, zorder=4)
    ax.axhline(0, color="#898781", lw=0.8, ls=":")
    ax.set_xticks(x, comps)
    ax.set_ylabel("log2 (section median / distal)")
    ax.set_title(title, loc="left")
    return per


def plot_gradient(g: pd.DataFrame, ax, title: str, ylabel: str):
    """Median per distance bin, one line per scene; shaded = inside the lesion."""
    lo = g["mid_um"].min()
    ax.axvspan(lo - 50, 0, color="#e66767", alpha=0.08, lw=0)
    for (scene, d), col in zip(g.groupby("scene", sort=True), SCENE_COLORS, strict=False):
        d = d.sort_values("mid_um")
        ax.plot(d["mid_um"], d["median"], color=col, lw=2, marker="o", ms=4, label=scene)
    ax.axvline(0, color="#0b0b0b", lw=0.8)
    ax.text(0, ax.get_ylim()[1], " lesion edge", va="top", fontsize=8, color="#52514e")
    ax.set_xlabel("signed distance to the lesion edge (µm; negative = inside)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, frameon=False)


def plot_density_matched(tab: pd.DataFrame, ax, title: str, xlabel: str):
    x = np.arange(len(tab))
    for c in tab.columns:
        ax.plot(x, tab[c].to_numpy(float), color=COLORS[c], lw=2, marker="o", ms=6, label=c)
    ax.set_xticks(x, [str(i) for i in tab.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("median Pu.1 of Pu.1⁺ cells")
    ax.set_title(title, loc="left")
    ax.legend(fontsize=8, frameon=False)


def plot_section_map(run, df: pd.DataFrame, section: str, ax, value: str = "Pu1_mean_norm", vmax_q: float = 0.98):
    """Pu.1⁺ cells of one section on one slide coloured by ``value`` (single-hue ramp, light = high) over
    the dimmed image, with the lesion outline (white)."""
    from matplotlib.colors import LinearSegmentedColormap

    from .report import read_crop_um, section_bbox_um

    sid = int(run.sections.set_index("section_name").loc[section, "section_id"])
    x0, x1, y1, y0 = section_bbox_um(run, sid)
    cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    rgb, ext = read_crop_um(run, cx, cy, size, scale=0.2)
    if rgb is not None:
        ax.imshow(rgb.mean(axis=2) * 0.35, extent=ext, cmap="gray", vmin=0, vmax=1)
    d = df[(df["scene"] == run.name) & (df["section"] == section) & df["pu1_pos"]].sort_values(value)
    cmap = LinearSegmentedColormap.from_list("pu1", ["#3a2a1e", "#c98500", "#ffe6b3"])
    lo, hi = d[value].quantile([0.02, vmax_q])
    sc = ax.scatter(d["x_um"], d["y_um"], c=d[value], s=5, cmap=cmap, vmin=lo, vmax=hi, linewidths=0)
    # lesion outline in white: the usual yellow disappears into the Pu.1 ramp
    ax.contour(run.map("lesion_mask").astype(float), levels=[0.5], colors="white", linewidths=0.8,
               extent=run.extent_um, origin="upper")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{run.name} – {section}: Pu.1⁺ cells by Pu.1 level", loc="left", fontsize=10)
    return sc
