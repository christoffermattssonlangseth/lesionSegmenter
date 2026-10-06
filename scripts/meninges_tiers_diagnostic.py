"""Diagnostic for a three-tier surface model: meninges | buffer | parenchyma (nothing is changed in the run).

The current pipeline labels the outer ``parenchyma.erode_um`` (30 µm) band of each section, measured
on the 10 µm grid, as meninges and never collects it. That band is too wide where the meninges are a
single cell layer, and too narrow where they swell into a dense infiltrate next to lesions (40–80 µm).
This script places the boundary per location on a ~1.3 µm raster of the whole scene:

* **surface**    – outline through the outermost nuclei (cell centroids, dilated by a nucleus radius,
  gaps closed) inside the curated section outlines; deeper than ``EDGE_UM`` inside the pipeline's grid
  surface everything is tissue (the cell-level outline breaks up in loose tissue);
* **meninges**   – ``--method adaptive`` (default): the outer ``--seed-um`` layer plus every patch of
  *compact* tissue (nuclear area fraction above the ``--compact-q`` percentile of deep parenchyma, i.e.
  nuclei touching) that is connected to that layer and lies within ``--max-um`` of the surface; dense
  lesion tissue that does not touch the surface layer is left alone. ``--method depth``: a fixed
  ``--men-um`` band. Thin flaps (opening with ``--open-um``) are meninges in both;
* **buffer**     – tissue within ``--gap-um`` of the meninges (``--method depth``: up to ``--buffer-um``),
  plus every parenchyma cell within ``--cell-gap-um`` of a meningeal cell – collected in **no** reaction;
* **parenchyma** – the rest; lesion zones as before.

    python scripts/meninges_tiers_diagnostic.py --run-dir outputs/sdata

Writes ``<run-dir>/cohort/meninges_tiers_summary.csv`` (per scene × section),
``<run-dir>/cohort/meninges_tiers_profile.csv`` and figures to ``--fig-dir``.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.filters import threshold_otsu

from lesionseg.report import C_LESION, load_runs, read_crop_um, section_bbox_um

IMG_SCALE = 0.25      # image read at 1/4 resolution (~1.3 µm/px) – enough to see nuclei touching
CELL_R_UM = 5.0       # nucleus radius: the surface runs along the outer edge of the outermost nuclei
CLOSE_UM = 30.0       # gaps between surface cells closed up to 2× this
EDGE_UM = 20.0        # deeper than this inside the grid surface is always tissue
# tier colours (validated on the dark image surface: lightness band, CVD and contrast pass)
C_MEN, C_REC, C_BUF, C_PAR = "#e0559c", "#729828", "#c3c2b7", "#898781"
SCENE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
LESION_ZONES = ("core", "rim", "peri", "deep")


def _edt(m):
    return ndi.distance_transform_edt(m)


def _norm(a):
    lo, hi = np.percentile(a[a > 0], [1, 99.5])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def scene_layers(run, a) -> dict:
    """Surface, depth, meninges and buffer masks on a whole-scene raster of ``res`` µm."""
    res = run.px / IMG_SCALE
    H, W = (math.ceil(s * IMG_SCALE) for s in run.log["scene_shape_px"])
    c = run.cells
    m = np.zeros((H, W), bool)
    m[(c.y_um / res).astype(int).clip(0, H - 1), (c.x_um / res).astype(int).clip(0, W - 1)] = True
    m = _edt(~m) * res <= CELL_R_UM
    m = _edt(_edt(~m) * res <= CLOSE_UM) * res > CLOSE_UM          # closing via distance transforms
    sec = run.map("sections")
    rows = (np.arange(H) * res / run.grid.bin_um).astype(int).clip(0, sec.shape[0] - 1)
    cols = (np.arange(W) * res / run.grid.bin_um).astype(int).clip(0, sec.shape[1] - 1)
    # the cell-level outline is only trusted at the edge: in loose / sparse tissue it can break up, so
    # everything deeper than EDGE_UM inside the pipeline's (grid) surface is tissue regardless
    grid_surface = run.map("surface").astype(bool)[rows][:, cols]
    m |= _edt(grid_surface) * res > EDGE_UM
    m = ndi.binary_fill_holes(m)
    m &= sec[rows][:, cols] > 0
    depth = (_edt(m) * res).astype(np.float32)
    opened = _edt(~(depth > a.open_um)) * res <= a.open_um
    flap = m & ~opened
    out = {"res": res, "tissue": m, "depth": depth}
    if a.method == "adaptive":
        r = run.open_image()
        nuc = _norm(r.read_overview(run.scene, IMG_SCALE, r.channel_index("SYTOG")))[:H, :W]
        if nuc.shape != (H, W):
            nuc = np.pad(nuc, ((0, H - nuc.shape[0]), (0, W - nuc.shape[1])))
        nuc_mask = nuc > threshold_otsu(nuc[m])
        frac = ndi.gaussian_filter(nuc_mask.astype(np.float32), a.compact_sigma_um / res)
        thr = float(np.percentile(frac[m & (depth > 200)], a.compact_q))
        compact = (frac > thr) & m & (depth <= a.max_um)
        seed = m & (depth <= a.seed_um)
        lab, _ = ndi.label(compact)
        keep = np.unique(lab[seed & compact])
        men = (np.isin(lab, keep[keep > 0]) | seed | flap) & m
        buffer = m & ~men & (_edt(~men) * res <= a.gap_um)
        out["compact_threshold"] = thr
    else:
        men = m & ((depth <= a.men_um) | flap)
        buffer = m & ~men & (depth <= a.buffer_um)
    out.update(meninges=men, buffer=buffer)
    return out


def thickness(lay, sid: int) -> tuple[float, float, float]:
    """Meninges thickness along a section's surface: depth of the inner meninges edge (median, p95),
    and area / perimeter."""
    in_sec = lay["section_ids"] == sid
    d = lay["depth"][lay["inner_edge"] & in_sec]
    n_perim = int((lay["perimeter"] & in_sec).sum())
    if not len(d) or not n_perim:
        return np.nan, np.nan, np.nan
    men_area = int((lay["meninges"] & in_sec).sum())
    return float(np.median(d)), float(np.percentile(d, 95)), float(men_area * lay["res"] / n_perim)


def classify(run, a) -> tuple[pd.DataFrame, dict]:
    lay = scene_layers(run, a)
    res = lay["res"]
    H, W = lay["tissue"].shape
    c = run.cells.copy()
    iy = (c.y_um / res).astype(int).clip(0, H - 1).to_numpy()
    ix = (c.x_um / res).astype(int).clip(0, W - 1).to_numpy()
    in_sec = c.section_id.to_numpy() > 0
    in_tis = lay["tissue"][iy, ix]
    c["depth_um"] = np.where(in_sec & in_tis, lay["depth"][iy, ix], np.nan)
    tier = np.where(in_sec, "parenchyma", "outside").astype(object)
    tier[in_sec & lay["buffer"][iy, ix]] = "buffer"
    tier[in_sec & (lay["meninges"][iy, ix] | ~in_tis)] = "meninges"
    # guaranteed separation at cell level: no parenchyma cell within cell_gap_um of a meningeal cell
    xy = c[["x_um", "y_um"]].to_numpy(float)
    men = tier == "meninges"
    if men.any() and a.cell_gap_um > 0:
        p_idx = np.flatnonzero(tier == "parenchyma")
        dist, _ = cKDTree(xy[men]).query(xy[p_idx], k=1)
        tier[p_idx[dist < a.cell_gap_um]] = "buffer"
    c["tier"] = tier
    # what the current run does: zone 'meninges' = excluded
    zone = c.zone.astype(str)
    c["was_meninges"] = zone == "meninges"
    c["recovered"] = c.was_meninges & (c.tier == "parenchyma")
    c["newly_excluded"] = ~c.was_meninges & c.tier.isin(["meninges", "buffer"])
    c["newly_excluded_lesion_zone"] = c.newly_excluded & zone.isin(LESION_ZONES)
    # would a recovered cell join a lesion? the lesion model map is computed before the parenchyma clip
    if (run.dir / "maps" / "model_prob.tif").exists():
        prob = run.map("model_prob")
        gx, gy = run.grid.to_grid(c.x_um.to_numpy(), c.y_um.to_numpy())
        nr, nc = run.grid.shape
        p = prob[np.clip(gy.astype(int), 0, nr - 1), np.clip(gx.astype(int), 0, nc - 1)]
    else:
        p = np.zeros(len(c))
    les_sec = c.section_has_lesion.to_numpy(bool)
    d_les = c.dist_to_lesion_um.to_numpy(float)
    c["recovered_lesion_est"] = c.recovered & les_sec & (d_les <= a.attach_um) & (p >= a.lesion_p)
    c["men_near_lesion"] = (c.tier == "meninges") & les_sec & (d_les <= 50)
    # section ids on the raster (for thickness)
    sec = run.map("sections")
    rows = (np.arange(H) * res / run.grid.bin_um).astype(int).clip(0, sec.shape[0] - 1)
    cols = (np.arange(W) * res / run.grid.bin_um).astype(int).clip(0, sec.shape[1] - 1)
    lay["section_ids"] = sec[rows][:, cols]
    m, men = lay["tissue"], lay["meninges"]
    lay["inner_edge"] = men & ndi.binary_dilation(m & ~men)
    lay["perimeter"] = m & ~ndi.binary_erosion(m)
    return c, lay


def summarize(run, c, lay) -> pd.DataFrame:
    pos = c[c.pu1_pos.astype(bool) & (c.tier != "outside")]
    g = pos.drop(columns="section_id").groupby(pos.section_id)
    out = pd.DataFrame({
        "pu1_meninges": g.apply(lambda s: int((s.tier == "meninges").sum())),
        "pu1_buffer": g.apply(lambda s: int((s.tier == "buffer").sum())),
        "pu1_parenchyma": g.apply(lambda s: int((s.tier == "parenchyma").sum())),
        "pu1_excluded_now": g.apply(lambda s: int(s.was_meninges.sum())),
        "pu1_recovered": g.apply(lambda s: int(s.recovered.sum())),
        "pu1_recovered_lesion_est": g.apply(lambda s: int(s.recovered_lesion_est.sum())),
        "pu1_recovered_manual_core": g.apply(lambda s: int((s.recovered & (s.manual_core_id > 0)).sum())),
        "pu1_newly_excluded": g.apply(lambda s: int(s.newly_excluded.sum())),
        "pu1_newly_excluded_lesion_zone": g.apply(lambda s: int(s.newly_excluded_lesion_zone.sum())),
        "pu1_meninges_near_lesion": g.apply(lambda s: int(s.men_near_lesion.sum())),
    }).reset_index()
    th = {sid: thickness(lay, int(sid)) for sid in out.section_id}
    out["meninges_thickness_median_um"] = out.section_id.map(lambda s: round(th[s][0], 1))
    out["meninges_thickness_p95_um"] = out.section_id.map(lambda s: round(th[s][1], 1))
    out["meninges_mean_thickness_um"] = out.section_id.map(lambda s: round(th[s][2], 1))
    names = run.sections.set_index("section_id")
    out.insert(0, "scene", run.name)
    out.insert(2, "section_name", out.section_id.map(names["section_name"]).astype(str))
    out.insert(3, "lesion_section", out.section_id.map(names["is_lesion_section"]).astype(bool))
    return out


def profile(run, c) -> pd.DataFrame:
    d = c[c.depth_um.notna() & (c.depth_um <= 100)]
    b = (d.depth_um // 4) * 4 + 2
    return pd.DataFrame({"scene": run.name, "depth_um": b}).assign(
        eccentricity=d.eccentricity, pu1=d.pu1_pos.astype(float),
        men=(d.tier == "meninges").astype(float)).groupby(["scene", "depth_um"]).agg(
        median_eccentricity=("eccentricity", "median"), pu1_fraction=("pu1", "mean"),
        meninges_fraction=("men", "mean"), n_cells=("pu1", "size")).reset_index()


def plot_profile(prof: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.2))
    titles = ["median nuclear eccentricity (elongation)", "Pu.1⁺ fraction of all cells",
              "fraction of cells called meninges"]
    for (scene, d), col in zip(prof.groupby("scene", sort=True), SCENE_COLORS, strict=False):
        d = d[d.n_cells >= 50]
        for ax, k in zip(axes, ["median_eccentricity", "pu1_fraction", "meninges_fraction"], strict=True):
            ax.plot(d.depth_um, d[k], color=col, lw=2, label=scene)
    for ax, ttl in zip(axes, titles, strict=True):
        ax.axvline(30, color="#52514e", ls=":", lw=1)
        ax.set_xlabel("depth below the surface (µm, cell level)")
        ax.set_title(ttl, loc="left")
        ax.set_xlim(0, 100)
    axes[0].text(30.5, axes[0].get_ylim()[0], "current 30 µm cut", va="bottom", fontsize=8, color="#52514e")
    axes[2].legend(fontsize=8, frameon=False)
    plt.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def _draw_tiers(ax, run, c, lay, sid, view, size=6):
    """Pu.1⁺ cells of section ``sid`` by tier, plus surface / meninges / buffer outlines within ``view``
    (x0, x1, y0, y1 in µm)."""
    pos = c[(c.section_id == sid) & c.pu1_pos.astype(bool)]
    par = pos[(pos.tier == "parenchyma") & ~pos.recovered]
    ax.scatter(par.x_um, par.y_um, s=size * 0.25, c=C_PAR, alpha=0.6, linewidths=0)
    b = pos[pos.tier == "buffer"]
    ax.scatter(b.x_um, b.y_um, s=size, facecolors="none", edgecolors=C_BUF, linewidths=0.7, zorder=3)
    m = pos[pos.tier == "meninges"]
    ax.scatter(m.x_um, m.y_um, s=size, c=C_MEN, edgecolors="black", linewidths=0.2, zorder=4)
    r = pos[pos.recovered]
    ax.scatter(r.x_um, r.y_um, s=size * 1.3, c=C_REC, edgecolors="black", linewidths=0.2, zorder=5)
    res = lay["res"]
    H, W = lay["tissue"].shape
    x0, x1, y0, y1 = view
    i0, i1 = max(int(y0 / res), 0), min(int(y1 / res) + 1, H)
    j0, j1 = max(int(x0 / res), 0), min(int(x1 / res) + 1, W)
    ext = (j0 * res, j1 * res, i1 * res, i0 * res)
    for key, col, ls in (("tissue", "white", "-"), ("meninges", C_MEN, "-"), ("men_or_buffer", C_BUF, "--")):
        arr = lay["meninges"] | lay["buffer"] if key == "men_or_buffer" else lay[key]
        sub = arr[i0:i1, j0:j1]
        if sub.any() and not sub.all():
            ax.contour(sub.astype(float), levels=[0.5], colors=col, linewidths=0.8, linestyles=ls,
                       extent=ext, origin="upper")
    ax.contour(run.map("parenchyma").astype(float), levels=[0.5], colors="#52514e", linewidths=0.8,
               linestyles=":", extent=run.extent_um, origin="upper")
    ax.contour(run.map("lesion_mask").astype(float), levels=[0.5], colors=C_LESION, linewidths=0.9,
               extent=run.extent_um, origin="upper")
    return pos


def _legend(ax, pos):
    n = {t: int((pos.tier == t).sum()) for t in ("meninges", "buffer", "parenchyma")}
    handles = [
        Line2D([], [], marker="o", ls="", mfc=C_MEN, mec="black", label=f"meninges ({n['meninges']})"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=C_BUF, label=f"buffer – never collected ({n['buffer']})"),
        Line2D([], [], marker="o", ls="", mfc=C_REC, mec="black",
               label=f"recovered: excluded now, parenchyma here ({int(pos.recovered.sum())})"),
        Line2D([], [], marker="o", ls="", mfc=C_PAR, mec="none", label="parenchyma (unchanged)"),
        Line2D([], [], color="white", lw=1, label="surface (outermost nuclei)"),
        Line2D([], [], color=C_MEN, lw=1, label="inner edge of meninges"),
        Line2D([], [], color=C_BUF, lw=1, ls="--", label="inner edge of buffer"),
        Line2D([], [], color="#52514e", lw=1, ls=":", label="current parenchyma edge (30 µm)"),
        Line2D([], [], color=C_LESION, lw=1, label="automatic lesion"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)


def plot_section(run, c, lay, sid, a, out: Path):
    x0, x1, y1, y0 = section_bbox_um(run, sid)
    cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    rgb, ext = read_crop_um(run, cx, cy, size, scale=0.2)
    if rgb is not None:
        axes[0].imshow(rgb * 0.5, extent=ext)
    pos = _draw_tiers(axes[0], run, c, lay, sid, (x0, x1, y0, y1), size=6)
    axes[0].set_xlim(x0, x1)
    axes[0].set_ylim(y1, y0)
    # zoom on the thickest meninges (else the densest patch of meningeal cells)
    men = pos[pos.tier == "meninges"]
    zs = a.zoom_um
    if len(men):
        zx, zy = men.loc[men.depth_um.fillna(0).idxmax(), ["x_um", "y_um"]]
    else:
        zx, zy = cx, cy
    rgb, ext = read_crop_um(run, zx, zy, zs, scale=0.5)
    if rgb is not None:
        axes[1].imshow(rgb * 0.6, extent=ext)
    view = (zx - zs / 2, zx + zs / 2, zy - zs / 2, zy + zs / 2)
    _draw_tiers(axes[1], run, c, lay, sid, view, size=40)
    axes[1].set_xlim(view[0], view[1])
    axes[1].set_ylim(view[3], view[2])
    axes[0].add_patch(plt.Rectangle((view[0], view[2]), zs, zs, fill=False, ec="white", lw=1))
    _legend(axes[1], pos)
    name = run.sections.set_index("section_id").loc[sid]
    kind = "lesion section" if bool(name.get("is_lesion_section", False)) else "control section"
    axes[0].set_title(f"{run.name} – {name.get('section_name', sid)} ({kind}) · Pu.1⁺ cells by tier", loc="left")
    axes[1].set_title(f"zoom {zs:g} µm on the thickest meninges", loc="left")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_closeups(run, c, lay, sid, out: Path, zs: float = 250.0, n: int = 4):
    """Full-resolution close-ups, raw image (top) and tiers (bottom), at the thickest meninges and at
    the densest patches of recovered cells – the places where the boundary matters."""
    pos = c[(c.section_id == sid) & c.pu1_pos.astype(bool)]
    cand = []
    men = pos[pos.tier == "meninges"].sort_values("depth_um", ascending=False)
    cand += list(men[["x_um", "y_um"]].to_numpy()[:200])
    rec = pos[pos.recovered]
    if len(rec) >= 5:
        xy = rec[["x_um", "y_um"]].to_numpy()
        k = cKDTree(xy).query_ball_point(xy, r=zs / 3, return_length=True)
        cand += list(xy[np.argsort(-k)][:200])
    picks = []
    for q in cand:
        if all(np.hypot(*(q - p)) > 1.5 * zs for p in picks):
            picks.append(q)
        if len(picks) == n:
            break
    if not picks:
        return
    fig, axes = plt.subplots(2, len(picks), figsize=(6 * len(picks), 12), squeeze=False)
    name = run.sections.set_index("section_id").loc[sid].get("section_name", sid)
    for j, (zx, zy) in enumerate(picks):
        rgb, ext = read_crop_um(run, zx, zy, zs, scale=1.0)
        view = (zx - zs / 2, zx + zs / 2, zy - zs / 2, zy + zs / 2)
        for i in range(2):
            ax = axes[i, j]
            if rgb is not None:
                ax.imshow(np.clip(rgb * 1.3, 0, 1), extent=ext)
            if i == 1:
                _draw_tiers(ax, run, c, lay, sid, view, size=50)
            ax.set_xlim(view[0], view[1])
            ax.set_ylim(view[3], view[2])
            ax.set_xticks([])
            ax.set_yticks([])
        axes[0, j].set_title(f"{run.name} – {name} @ ({zx:.0f}, {zy:.0f}) µm · {zs:g} µm", loc="left", fontsize=9)
    _legend(axes[1, -1], pos)
    plt.tight_layout()
    fig.savefig(out, dpi=85)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--fig-dir", default="results/meninges_tiers")
    ap.add_argument("--method", choices=["adaptive", "depth"], default="adaptive")
    # adaptive
    ap.add_argument("--seed-um", type=float, default=10.0, help="outer layer that is always meninges")
    ap.add_argument("--max-um", type=float, default=80.0, help="max depth of the swollen meninges")
    ap.add_argument("--compact-sigma-um", type=float, default=8.0, help="smoothing of the nuclear area fraction")
    ap.add_argument("--compact-q", type=float, default=99.0,
                    help="compact = nuclear area fraction above this percentile of deep (>200 µm) tissue")
    ap.add_argument("--gap-um", type=float, default=10.0, help="buffer width inward of the meninges")
    # depth
    ap.add_argument("--men-um", type=float, default=12.0)
    ap.add_argument("--buffer-um", type=float, default=20.0)
    # both
    ap.add_argument("--cell-gap-um", type=float, default=6.0, help="min distance parenchyma ↔ meningeal cell")
    ap.add_argument("--open-um", type=float, default=150.0)
    ap.add_argument("--lesion-p", type=float, default=0.25, help="model probability for 'would join a lesion'")
    ap.add_argument("--attach-um", type=float, default=50.0, help="max distance to an existing lesion")
    ap.add_argument("--zoom-um", type=float, default=500.0)
    ap.add_argument("--per-scene", type=int, default=2, help="lesion sections plotted per scene")
    ap.add_argument("--tag", default="", help="suffix for output files (compare settings)")
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args()
    fig_dir = Path(a.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    out = Path(a.run_dir) / "cohort"
    out.mkdir(exist_ok=True)
    summ, profs = [], []
    for run in load_runs(a.run_dir):
        c, lay = classify(run, a)
        s = summarize(run, c, lay)
        summ.append(s)
        profs.append(profile(run, c))
        thr = lay.get("compact_threshold")
        print(f"{run.name}: compact threshold {thr:.3f}" if thr is not None else run.name)
        if a.no_figures:
            continue
        tag = run.name.replace(" ", "_")
        les = s[s.lesion_section].sort_values("meninges_thickness_p95_um", ascending=False).head(a.per_scene)
        ctrl = s[~s.lesion_section].sort_values("pu1_meninges", ascending=False).head(1)
        for _, row in pd.concat([les, ctrl]).iterrows():
            plot_section(run, c, lay, int(row.section_id), a, fig_dir / f"{tag}_{row.section_name}.png")
        for _, row in les.iterrows():
            plot_closeups(run, c, lay, int(row.section_id), fig_dir / f"closeup_{tag}_{row.section_name}.png")
    summ = pd.concat(summ, ignore_index=True)
    prof = pd.concat(profs, ignore_index=True)
    summ.to_csv(out / f"meninges_tiers_summary{a.tag}.csv", index=False)
    prof.to_csv(out / f"meninges_tiers_profile{a.tag}.csv", index=False)
    if not a.no_figures:
        plot_profile(prof, fig_dir / "depth_profile.png")
    cols = [c for c in summ.columns if c.startswith("pu1_")]
    print("\nPu.1⁺ cells per scene:")
    print(summ.groupby(["scene", "lesion_section"])[cols].sum().to_string())
    print("\ntotal:")
    print(summ.groupby("lesion_section")[cols].sum().T.to_string())
    th = [c for c in summ.columns if "thickness" in c]
    print("\nmeninges thickness (µm) per section, lesion vs control:")
    print(summ.groupby("lesion_section")[th].describe(percentiles=[0.5]).T.round(1).to_string())


if __name__ == "__main__":
    main()
