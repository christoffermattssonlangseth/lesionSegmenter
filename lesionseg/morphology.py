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
