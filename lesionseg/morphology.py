"""Nuclear elongation: segmented nucleus outlines in representative zoom-ins (report notebook 09).

``eccentricity`` (from the Cellpose cell table) is that of the ellipse with the same second moments as
the nucleus mask: √(1 − (b/a)²), 0 = circle, → 1 = line. Classes used here:

* **round**      eccentricity < 0.6   (long : short axis < 1.25)
* **elongated**  eccentricity ≥ 0.85  (long : short axis ≥ 1.9)
* intermediate in between.

Outlines come from the label image of the SpatialData store (mask value == the cell table's ``label``,
scene px), read crop by crop, so every nucleus is shown – not only the Pu.1⁺ cells whose contours are in the cell table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

ROUND_MAX, ELONG_MIN = 0.6, 0.85
CLASS_ORDER = ["round", "intermediate", "elongated"]
CLASS_COLORS = {"round": "#2a78d6", "intermediate": "#c3c2b7", "elongated": "#eb6834"}


def ecc_class(ecc) -> np.ndarray:
    e = np.asarray(ecc, float)
    return np.where(e < ROUND_MAX, "round", np.where(e >= ELONG_MIN, "elongated", "intermediate"))


def axis_ratio(ecc) -> np.ndarray:
    """long : short axis of the moment ellipse."""
    e = np.clip(np.asarray(ecc, float), 0, 0.999)
    return 1 / np.sqrt(1 - e ** 2)


def label_crop(run, x_um: float, y_um: float, size_um: float):
    """(labels, extent µm) of a square crop of the label image centred at (x, y) µm."""
    import zarr

    px = run.px
    arr = zarr.open(str(run.log["cells"]["labels"]), mode="r")
    H, W = arr.shape
    half = size_um / 2 / px
    x0, y0 = int(max(x_um / px - half, 0)), int(max(y_um / px - half, 0))
    x1, y1 = int(min(x_um / px + half, W)), int(min(y_um / px + half, H))
    lab = np.asarray(arr[y0:y1, x0:x1])
    return lab, (x0 * px, x1 * px, y1 * px, y0 * px)


def nucleus_polygons(lab: np.ndarray, extent) -> dict[int, np.ndarray]:
    """Outline (N × 2, µm) per label in the crop: the longest iso-contour of each label's mask."""
    from scipy import ndimage as ndi
    from skimage.measure import find_contours

    x0, x1, y1, y0 = extent
    sy = (y1 - y0) / lab.shape[0]
    sx = (x1 - x0) / lab.shape[1]
    out = {}
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        m = np.pad(lab[sl] == i, 1)
        cs = find_contours(m.astype(float), 0.5)
        if not cs:
            continue
        c = max(cs, key=len) - 1  # undo the pad
        out[i] = np.c_[x0 + (c[:, 1] + sl[1].start + 0.5) * sx, y0 + (c[:, 0] + sl[0].start + 0.5) * sy]
    return out


def window_compartments(cells: pd.DataFrame, comp: pd.Series, max_meninges_depth_um: float = 80.0) -> pd.Series:
    """Compartments for picking windows: meninges deeper than ``max_meninges_depth_um`` (thin flaps, roots,
    often sparse or artefact-ridden) become ``flap`` so they are never shown as typical meninges."""
    out = comp.copy()
    if "depth_um" in cells:
        flap = (comp == "meninges").to_numpy() & ~(cells["depth_um"].to_numpy(float) <= max_meninges_depth_um)
        out[flap] = "flap"
    return out


def representative_windows(cells: pd.DataFrame, comp: pd.Series, target: str, *, size_um: float = 120.0,
                           n: int = 3, min_frac: float = 0.8, min_cells: int = 40, seed: int = 0,
                           n_candidates: int = 400, min_sep_um: float = 600.0) -> list[tuple[float, float]]:
    """Window centres where ``target`` dominates (≥ ``min_frac`` of the cells) and the median eccentricity
    is closest to the compartment's own median – typical, not extreme, views. Spread over sections."""
    xy = cells[["x_um", "y_um"]].to_numpy(float)
    ecc = cells["eccentricity"].to_numpy(float)
    is_t = (comp == target).to_numpy()
    if not is_t.any():
        return []
    med = np.median(ecc[is_t])
    tree = cKDTree(xy)
    rng = np.random.default_rng(seed)
    cand = rng.choice(np.flatnonzero(is_t), min(n_candidates, int(is_t.sum())), replace=False)
    rows = []
    for i in cand:
        nb = tree.query_ball_point(xy[i], r=size_um / 2)
        if len(nb) < min_cells:
            continue
        frac = is_t[nb].mean()
        if frac < min_frac:
            continue
        rows.append((abs(np.median(ecc[nb][is_t[nb]]) - med), -frac, i))
    rows.sort()
    picks: list[tuple[float, float]] = []
    secs: list = []
    for _, _, i in rows:
        p = xy[i]
        sec = cells["section_id"].iat[i] if "section_id" in cells else None
        if any(np.hypot(*(p - q)) < min_sep_um for q in picks):
            continue
        if sec in secs and len(set(secs)) < n:  # prefer different sections first
            continue
        picks.append(p)
        secs.append(sec)
        if len(picks) == n:
            break
    return [(float(x), float(y)) for x, y in picks]


def plot_nuclei(ax, run, cells: pd.DataFrame, x_um: float, y_um: float, size_um: float = 120.0, *,
                raw: bool = False, alpha: float = 0.6, comp: pd.Series | None = None, focus: str | None = None,
                surface: bool = True) -> dict:
    """Raw image (``raw``) or every segmented nucleus as a polygon filled by elongation class; Pu.1⁺
    nuclei get a white edge. With ``comp`` (compartment per cell, aligned to ``cells``) and ``focus``,
    nuclei of the focus compartment are drawn at full strength and all others faded. ``surface`` draws
    the inner edges of the meninges (solid grey) and buffer (dashed white).

    Returns counts for the window: nuclei in / outside the focus compartment, and the elongated and
    Pu.1⁺ shares of the focus nuclei."""
    from matplotlib.collections import PolyCollection

    from .report import draw_surface, read_crop_um

    rgb, ext = read_crop_um(run, x_um, y_um, size_um, scale=1.0)
    if rgb is not None:
        ax.imshow(np.clip(rgb * (1.3 if raw else 0.55), 0, 1), extent=ext)
    half = size_um / 2
    stats: dict = {}
    if not raw:
        lab, lext = label_crop(run, x_um, y_um, size_um)
        polys = nucleus_polygons(lab, lext)
        c = cells.set_index("label")  # mask label (``cell_id`` is the row number)
        ids = [i for i in polys if i in c.index]
        cls = ecc_class(c.loc[ids, "eccentricity"])
        pos = c.loc[ids, "pu1_pos"].to_numpy(bool)
        if comp is not None and focus is not None:
            cmap = pd.Series(np.asarray(comp, dtype=object), index=cells["label"].to_numpy())
            infocus = (cmap.reindex(ids).to_numpy() == focus)
        else:
            infocus = np.ones(len(ids), bool)
        for k in CLASS_ORDER:
            sel = cls == k
            for edge, lw, m, a in (("#0b0b0b", 0.4, sel & ~pos & infocus, alpha),
                                   ("#ffffff", 1.3, sel & pos & infocus, alpha),
                                   ("#52514e", 0.3, sel & ~infocus, 0.15)):
                verts = [polys[i] for i, mm in zip(ids, m, strict=True) if mm]
                if verts:
                    ax.add_collection(PolyCollection(verts, facecolors=CLASS_COLORS[k], edgecolors=edge,
                                                     linewidths=lw, alpha=a))
        n_f = int(infocus.sum())
        stats = {"n_focus": n_f, "n_other": int((~infocus).sum()),
                 "elongated_focus": float((cls[infocus] == "elongated").mean()) if n_f else np.nan,
                 "pu1_focus": float(pos[infocus].mean()) if n_f else np.nan}
        if surface:
            draw_surface(ax, run, (x_um - half, x_um + half, y_um - half, y_um + half), lw=1.2)
    ax.set_xlim(x_um - half, x_um + half)
    ax.set_ylim(y_um + half, y_um - half)
    ax.set_xticks([])
    ax.set_yticks([])
    return stats


def stats_line(focus: str, st: dict) -> str:
    """'62 meninges nuclei (31 % elongated, 18 % Pu.1⁺) · 40 other (faded)'."""
    if not st:
        return ""
    return (f"{st['n_focus']} {focus} nuclei ({100 * st['elongated_focus']:.0f} % elongated, "
            f"{100 * st['pu1_focus']:.0f} % Pu.1⁺) · {st['n_other']} other (faded)")


def class_legend(ax, loc="lower right", surface: bool = True):
    from matplotlib.patches import Patch

    h = [Patch(facecolor=CLASS_COLORS["round"], edgecolor="#0b0b0b", label=f"round (ecc < {ROUND_MAX})"),
         Patch(facecolor=CLASS_COLORS["intermediate"], edgecolor="#0b0b0b", label="intermediate"),
         Patch(facecolor=CLASS_COLORS["elongated"], edgecolor="#0b0b0b", label=f"elongated (ecc ≥ {ELONG_MIN})"),
         Patch(facecolor="none", edgecolor="#ffffff", linewidth=1.3, label="white edge = Pu.1⁺"),
         Patch(facecolor="#c3c2b7", alpha=0.15, edgecolor="#52514e", label="faded = other compartment")]
    if surface:
        from matplotlib.lines import Line2D

        h += [Line2D([], [], color="#c3c2b7", lw=1.2, label="inner edge of meninges"),
              Line2D([], [], color="#ffffff", lw=1.2, ls="--", label="inner edge of buffer")]
    ax.legend(handles=h, loc=loc, fontsize=7, framealpha=0.85)


def class_shares(df: pd.DataFrame, pu1: str = "all") -> pd.DataFrame:
    """Share of round / intermediate / elongated nuclei per slide × section × compartment."""
    d = df if pu1 == "all" else df[df["pu1_pos"] == (pu1 == "pos")]
    d = d.assign(ecc_class=ecc_class(d["eccentricity"]))
    t = d.groupby(["scene", "section", "compartment", "ecc_class"], observed=True).size().unstack("ecc_class",
                                                                                                 fill_value=0)
    t = t.reindex(columns=CLASS_ORDER, fill_value=0)
    return t.div(t.sum(axis=1), axis=0).reset_index()


# --------------------------------------------------------------------------
# notebook 10: is the meningeal elongation reliable?
# --------------------------------------------------------------------------
def random_windows(cells: pd.DataFrame, comp: pd.Series, target: str, n: int, *, size_um: float = 120.0,
                   min_target: int = 15, seed: int = 0, min_sep_um: float = 300.0,
                   max_tries: int = 5000) -> list[tuple[float, float]]:
    """``n`` window centres drawn at random among ``target`` nuclei (no selection on elongation): a draw is
    kept if the window holds ≥ ``min_target`` target nuclei and is ≥ ``min_sep_um`` from earlier draws."""
    xy = cells[["x_um", "y_um"]].to_numpy(float)
    is_t = (comp == target).to_numpy()
    idx = np.flatnonzero(is_t)
    if not len(idx):
        return []
    tree = cKDTree(xy[is_t])
    rng = np.random.default_rng(seed)
    picks: list[tuple[float, float]] = []
    for i in rng.permutation(idx)[:max_tries]:
        p = xy[i]
        if any(np.hypot(p[0] - q[0], p[1] - q[1]) < min_sep_um for q in picks):
            continue
        if len(tree.query_ball_point(p, r=size_um / 2)) < min_target:
            continue
        picks.append((float(p[0]), float(p[1])))
        if len(picks) == n:
            break
    return picks


def reference_compartment(df: pd.DataFrame, near_surface_um: tuple[float, float] = (20.0, 80.0)) -> pd.Series:
    """Per cell: ``meninges`` (surface meninges, depth ≤ 80 µm), ``parenchyma`` (the section's own
    reference: distal in lesion sections, GM / WM in control sections) and ``near_surface`` (parenchyma
    20–80 µm below the surface – the edge-artefact control); everything else None."""
    comp = df["compartment"].astype(str)
    depth = df["depth_um"].to_numpy(float) if "depth_um" in df else np.full(len(df), np.nan)
    out = pd.Series(None, index=df.index, dtype=object)
    out[(comp == "meninges").to_numpy() & (depth <= 80)] = "meninges"
    ref = comp.isin(["distal", "GM", "WM"]).to_numpy()
    out[ref] = "parenchyma"
    par = comp.isin(["core", "rim", "peri", "deep", "distal", "GM", "WM"]).to_numpy()
    lo, hi = near_surface_um
    out[par & (depth >= lo) & (depth <= hi)] = "near_surface"
    return out


def meninges_effect(df: pd.DataFrame, ref: str = "parenchyma", metric: str = "elongated", thr: float = ELONG_MIN,
                    subset: pd.Series | None = None, min_cells: int = 20) -> pd.DataFrame:
    """Per section (replicate slides averaged): meninges − ``ref`` for ``metric`` = ``elongated`` (share
    with eccentricity ≥ ``thr``, in percentage points) or ``median`` (median eccentricity). ``subset``
    restricts the nuclei (e.g. solidity ≥ 0.9). Adds ``animal``."""
    d = df.assign(group=reference_compartment(df))
    if subset is not None:
        d = d[subset.reindex(d.index).fillna(False).to_numpy(bool)]
    d = d[d["group"].isin(["meninges", ref])]
    if metric == "elongated":
        d = d.assign(v=(d["eccentricity"] >= thr).astype(float) * 100)
        agg = "mean"
    else:
        d = d.assign(v=d["eccentricity"])
        agg = "median"
    g = d.groupby(["scene", "section", "group"])["v"].agg([agg, "size"]).reset_index()
    g = g[g["size"] >= min_cells].pivot_table(index=["scene", "section"], columns="group", values=agg)
    if not {"meninges", ref} <= set(g.columns):
        return pd.DataFrame(columns=["section", "meninges", ref, "delta", "animal"])
    g = g.dropna(subset=["meninges", ref])
    g["delta"] = g["meninges"] - g[ref]
    per = g.groupby("section")[["meninges", ref, "delta"]].mean().reset_index()
    per["animal"] = per["section"].str.rsplit("_", n=1).str[0]
    return per


def hierarchical_bootstrap(per: pd.DataFrame, col: str = "delta", n_boot: int = 4000, seed: int = 0):
    """95 % CI of the mean ``col`` resampling animals, then sections within each drawn animal."""
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy(float) for _, g in per.groupby("animal")]
    if not groups:
        return np.nan, np.nan
    means = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        vals = np.concatenate([rng.choice(groups[i], len(groups[i]), replace=True) for i in pick])
        means[b] = vals.mean()
    return tuple(np.percentile(means, [2.5, 97.5]))


def robustness_table(df: pd.DataFrame) -> pd.DataFrame:
    """The meninges − parenchyma effect under every check, one row each: sections, animals, median Δ,
    mean Δ with hierarchical-bootstrap 95 % CI, sections higher, sign-test and Wilcoxon p."""
    from scipy.stats import binomtest, wilcoxon

    pos = df["pu1_pos"].astype(bool)
    solid = df["solidity"] >= 0.9 if "solidity" in df else pd.Series(True, index=df.index)
    area = df["area_um2"].between(15, 45)
    checks = [
        ("elongated share (ecc ≥ 0.85), all nuclei", dict()),
        ("elongated share, ecc ≥ 0.80", dict(thr=0.80)),
        ("elongated share, ecc ≥ 0.90", dict(thr=0.90)),
        ("median eccentricity (×100)", dict(metric="median")),
        ("well-segmented nuclei only (solidity ≥ 0.9)", dict(subset=solid)),
        ("size-matched nuclei (15–45 µm²)", dict(subset=area)),
        ("Pu.1⁻ nuclei only", dict(subset=~pos)),
        ("Pu.1⁺ nuclei only", dict(subset=pos)),
        ("vs parenchyma 20–80 µm under the surface (edge control)", dict(ref="near_surface")),
    ]
    rows = []
    for name, kw in checks:
        per = meninges_effect(df, **kw)
        if not len(per):
            continue
        v = per["delta"].to_numpy(float) * (100 if kw.get("metric") == "median" else 1)
        per = per.assign(delta=v)
        lo, hi = hierarchical_bootstrap(per)
        k = int((v > 0).sum())
        rows.append({"check": name, "sections": len(v), "animals": per["animal"].nunique(),
                     "median_delta": np.median(v), "mean_delta": v.mean(), "ci95_low": lo, "ci95_high": hi,
                     "sections_higher": k, "sign_test_p": binomtest(k, len(v)).pvalue,
                     "wilcoxon_p": wilcoxon(v).pvalue if len(v) >= 6 else np.nan})
    return pd.DataFrame(rows)


def depth_profile(df: pd.DataFrame, edges=(0, 10, 20, 30, 40, 60, 80, 120, 160, 200, 300)) -> pd.DataFrame:
    """Elongated share (%) per depth bin below the surface, per slide (cells with a depth only)."""
    d = df[df["depth_um"].notna()]
    b = pd.cut(d["depth_um"], list(edges))
    g = d.assign(el=(d["eccentricity"] >= ELONG_MIN) * 100.0).groupby(["scene", b], observed=True)["el"].agg(
        ["mean", "size"]).reset_index()
    g["mid_um"] = g["depth_um"].map(lambda iv: iv.mid).astype(float)
    return g[g["size"] >= 50]


def meninges_layers(df: pd.DataFrame, edges=(0, 10, 30, 80), min_cells: int = 15) -> pd.DataFrame:
    """Elongated share (%) of surface-meninges nuclei per depth layer, per section (replicate slides
    averaged): is the elongation a property of the thin surface layer or of the whole meninges?"""
    d = df[reference_compartment(df) == "meninges"]
    labels = [f"{a}–{b} µm" for a, b in zip(edges[:-1], edges[1:], strict=False)]
    d = d.assign(layer=pd.cut(d["depth_um"], list(edges), labels=labels, include_lowest=True),
                 el=(d["eccentricity"] >= ELONG_MIN) * 100.0, pu1=d["pu1_pos"].astype(float) * 100)
    g = d.groupby(["scene", "section", "layer"], observed=True).agg(el=("el", "mean"), pu1=("pu1", "mean"),
                                                                      n=("el", "size")).reset_index()
    g = g[g["n"] >= min_cells]
    return g.groupby(["section", "layer"], observed=True)[["el", "pu1", "n"]].mean().reset_index()
