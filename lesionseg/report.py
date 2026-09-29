"""Helpers for the report notebooks: load a finished run and draw evidence figures.

Everything here is *evidence*, not verdicts: image crops with the automatic outlines
(and the manual annotations for comparison) so a reader can judge each call.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile

from .density import Grid
from .io import CziReader, SlideReader

# validated palette (dataviz reference): zones are ordered categories with fixed hues
ZONE_COLORS = {"distal": "#3987e5", "peri": "#199e70", "rim": "#c98500", "core": "#e66767"}  # dark-surface steps
ZONE_ORDER = ["distal", "peri", "rim", "core"]
GROUP_COLORS = {  # LMD well groups, fixed order
    "core": "#eb6834", "rim": "#eda100", "ring_0_10": "#1baf7a", "ring_10_20": "#2a78d6", "ring_20_40": "#4a3aa7",
    "ring_40_60": "#e87ba4", "ring_60_plus": "#008300",
    "ring_0_100": "#1baf7a", "ring_100_200": "#2a78d6", "ring_200_400": "#4a3aa7", "GM": "#c3c2b7", "WM": "#52514e",
}
C_LESION, C_CORE, C_MANUAL, C_PU1, C_MODEL = "#eda100", "#e34948", "#e87ba4", "#ffffff", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#c3c2b7",
                     "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.grid": False, "figure.dpi": 100, "font.size": 10})


@dataclass
class SceneRun:
    sample: str
    scene: int
    dir: Path
    log: dict
    cells: pd.DataFrame
    lesions: pd.DataFrame
    sections: pd.DataFrame
    wells: pd.DataFrame | None
    maps: dict = field(default_factory=dict)
    grid: Grid | None = None
    reader: SlideReader | None = None

    @property
    def name(self) -> str:
        return f"{self.sample} scene {self.scene}"

    @property
    def px(self) -> float:
        return self.log["pixel_size_um"]

    def map(self, key: str) -> np.ndarray:
        if key not in self.maps:
            self.maps[key] = tifffile.imread(self.dir / "maps" / f"{key}.tif")
        return self.maps[key]

    @property
    def extent_um(self):
        r, c = self.grid.shape
        return (0, c * self.grid.bin_um, r * self.grid.bin_um, 0)

    def open_image(self) -> SlideReader | None:
        if self.reader is None and self.log.get("image") and Path(self.log["image"]).exists():
            self.reader = CziReader(self.log["image"])
        return self.reader


def load_runs(root: str | Path = "outputs/sdata") -> list[SceneRun]:
    runs = []
    for logf in sorted(Path(root).glob("*/scene*/run_log.json")):
        d = logf.parent
        log = json.loads(logf.read_text())
        meta = json.loads((d / "maps" / "maps.json").read_text())
        grid = Grid(meta["bin_um"], tuple(meta["grid_shape"]), meta["pixel_size_um"])
        wells = pd.read_csv(d / "wells.csv") if (d / "wells.csv").exists() else None
        runs.append(SceneRun(log["sample"], log["scene"], d, log, pd.read_parquet(d / "cells.parquet"),
                             pd.read_csv(d / "lesions.csv"), pd.read_csv(d / "section_summary.csv"), wells, grid=grid))
    return runs


def autoscale(a, low=1.0, high=99.7):
    lo, hi = np.percentile(a, [low, high])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)


def composite(stack: np.ndarray, nuc: int = 0, pu1: int = 1, iba1: int | None = 2) -> np.ndarray:
    """nuclei → blue-grey, Pu.1 → orange, Iba1 → green (dim)."""
    rgb = np.zeros(stack.shape[1:] + (3,), np.float32)
    n = autoscale(stack[nuc], 5, 99.5)
    rgb += n[..., None] * np.array([0.55, 0.6, 0.75])
    p = autoscale(stack[pu1], 50, 99.8)
    rgb += p[..., None] * np.array([1.0, 0.55, 0.1])
    if iba1 is not None and iba1 < stack.shape[0]:
        g = autoscale(stack[iba1], 50, 99.8)
        rgb += g[..., None] * np.array([0.1, 0.6, 0.25])
    return np.clip(rgb, 0, 1)


def read_crop_um(run: SceneRun, x_um: float, y_um: float, size_um: float, scale: float = 1.0):
    """(rgb, extent) of a square crop centred at (x, y) µm; ``scale`` < 1 for larger fields."""
    r = run.open_image()
    if r is None:
        return None, None
    px = run.px
    info = r.scenes[run.scene]
    half = size_um / 2
    x0 = int(max((x_um - half) / px, 0))
    y0 = int(max((y_um - half) / px, 0))
    w = int(min(size_um / px, info.width - x0))
    h = int(min(size_um / px, info.height - y0))
    chans = [r.channel_index(c) for c in ("SYTOG", "AF647", "AF555") if c in r.channel_names]
    if not chans:
        chans = list(range(min(3, len(r.channel_names))))
    if scale < 1.0:
        stack = np.stack([r._czi.read_mosaic(region=(info.x0 + x0, info.y0 + y0, w, h), scale_factor=scale, C=c)
                          .squeeze() for c in chans])
    else:
        stack = r.read_region_stack(run.scene, x0, y0, w, h, chans)
    extent = (x0 * px, (x0 + w) * px, (y0 + h) * px, y0 * px)
    return composite(stack), extent


def _cells_in(run: SceneRun, ext, pos_only: bool = True) -> pd.DataFrame:
    c = run.cells
    m = c.x_um.between(ext[0], ext[1]) & c.y_um.between(ext[3], ext[2])
    if pos_only:
        m &= c.pu1_pos
    return c[m]


def draw_outlines(ax, run: SceneRun, *, lesion=True, core=True, manual=True, model_prob: float | None = None, lw=1.2):
    ext = run.extent_um
    kw = dict(extent=ext, origin="upper")
    if lesion:
        ax.contour(run.map("lesion_mask").astype(float), levels=[0.5], colors=C_LESION, linewidths=lw, **kw)
    if core:
        ax.contour((run.map("zones") == 4).astype(float), levels=[0.5], colors=C_CORE, linewidths=lw, **kw)
    if manual and (run.dir / "maps" / "manual_core.tif").exists():
        ax.contour(run.map("manual_core").astype(float), levels=[0.5], colors=C_MANUAL, linewidths=lw,
                   linestyles="--", **kw)
    if model_prob is not None and (run.dir / "maps" / "lesion_score.tif").exists():
        ax.contour(run.map("lesion_score"), levels=[model_prob], colors=C_MODEL, linewidths=0.8, **kw)


def outline_legend(ax, manual=True, model=False):
    from matplotlib.lines import Line2D

    items = [Line2D([], [], color=C_LESION, lw=2, label="automatic lesion"),
             Line2D([], [], color=C_CORE, lw=2, label="automatic core")]
    if manual:
        items.append(Line2D([], [], color=C_MANUAL, lw=2, ls="--", label="manual CORE (collaborator)"))
    if model:
        items.append(Line2D([], [], color=C_MODEL, lw=1.5, label="model p = 0.5"))
    ax.legend(handles=items, loc="lower right", fontsize=8, framealpha=0.8)


def section_bbox_um(run: SceneRun, section_id: int, pad_um: float = 150.0):
    sec = run.map("sections") == section_id
    ys, xs = np.nonzero(sec)
    b = run.grid.bin_um
    return (xs.min() * b - pad_um, xs.max() * b + pad_um, ys.max() * b + pad_um, ys.min() * b - pad_um)


def plot_section(run: SceneRun, section_id: int, ax=None, scale: float = 0.2, show_cells: bool = True):
    """One section: image composite + outlines (+ Pu.1⁺ cells coloured by zone)."""
    x0, x1, y1, y0 = section_bbox_um(run, section_id)
    cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    rgb, ext = read_crop_um(run, cx, cy, size, scale=scale)
    if rgb is not None:
        ax.imshow(rgb, extent=ext)
    draw_outlines(ax, run)
    if show_cells:
        c = run.cells[(run.cells.section_id == section_id) & run.cells.pu1_pos]
        for z in ZONE_ORDER:
            s = c[c.zone == z]
            if len(s):
                ax.scatter(s.x_um, s.y_um, s=2, c=ZONE_COLORS[z], linewidths=0, label=f"{z} (n={len(s)})")
        ax.legend(markerscale=6, fontsize=8, loc="upper left", framealpha=0.8)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    name = run.sections.set_index("section_id").loc[section_id]
    ttl = name.get("section_name", f"S{section_id}")
    flag = "lesion section" if bool(name.get("is_lesion_section", True)) else "control section"
    ax.set_title(f"{run.name} – {ttl} ({flag})")
    ax.set_xlabel("µm")
    ax.set_ylabel("µm")
    return ax


def lesion_gallery(run: SceneRun, lesion_ids=None, size_um: float = 700.0, ncols: int = 3, max_n: int = 12):
    """Full-resolution crops around lesions with outlines; ``lesion_ids`` default = largest first."""
    les = run.lesions.sort_values("area_um2", ascending=False)
    if lesion_ids is not None:
        les = les[les.lesion_id.isin(lesion_ids)]
    les = les.head(max_n)
    n = len(les)
    if n == 0:
        print("no lesions")
        return None
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    for ax, (_, L) in zip(axes.ravel(), les.iterrows(), strict=False):
        size = max(size_um, 1.6 * L.equiv_diameter_um)
        rgb, ext = read_crop_um(run, L.centroid_x_um, L.centroid_y_um, size)
        if rgb is not None:
            ax.imshow(rgb, extent=ext)
        draw_outlines(ax, run)
        c = _cells_in(run, ext)
        ax.scatter(c.x_um, c.y_um, s=4, facecolors="none", edgecolors=C_PU1, linewidths=0.4)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        sec = L.get("section_name", L.get("section_id", ""))
        ax.set_title(f"lesion {int(L.lesion_id)} · {sec} · {L.area_um2/1e3:.0f}×10³ µm² · "
                     f"Pu.1⁺ frac {L.mean_pu1_fraction:.2f}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    outline_legend(axes.ravel()[0])
    fig.suptitle(f"{run.name}: lesions (white rings = Pu.1⁺ cells)", y=1.0)
    fig.tight_layout()
    return fig


def missed_manual_cores(run: SceneRun, min_frac: float = 0.5) -> pd.DataFrame:
    f = run.dir / "validation_manual_cores.csv"
    if not f.exists():
        return pd.DataFrame()
    v = pd.read_csv(f)
    return v[v.frac_in_auto_lesion < min_frac]


def manual_core_gallery(run: SceneRun, rows: pd.DataFrame, size_um: float = 500.0, ncols: int = 3, max_n: int = 9):
    """Crops around manual cores (e.g. the ones the automatic mask missed)."""
    from . import sdata

    manual = sdata.load_manual_annotations(run.log["cells"]["zarr"])["CORE"].reset_index(drop=True)
    rows = rows.head(max_n)
    if len(rows) == 0:
        print("nothing to show")
        return None
    nrows = int(np.ceil(len(rows) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 6 * nrows), squeeze=False)
    for ax, (_, r) in zip(axes.ravel(), rows.iterrows(), strict=False):
        g = manual.loc[int(r.manual_core_id) - 1, "geometry"]
        cx, cy = g.centroid.x * run.px, g.centroid.y * run.px
        size = max(size_um, 2.5 * np.sqrt(g.area) * run.px)
        rgb, ext = read_crop_um(run, cx, cy, size)
        if rgb is not None:
            ax.imshow(rgb, extent=ext)
        draw_outlines(ax, run, model_prob=0.5 if run.log["lesion"]["score"] == "model" else None)
        c = _cells_in(run, ext)
        ax.scatter(c.x_um, c.y_um, s=4, facecolors="none", edgecolors=C_PU1, linewidths=0.4)
        ax.set_xlim(ext[0], ext[1])
        ax.set_ylim(ext[2], ext[3])
        ax.set_title(f"manual core {int(r.manual_core_id)} · {r.area_um2/1e3:.1f}×10³ µm² · "
                     f"{100*r.frac_in_auto_lesion:.0f}% inside automatic lesion", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    outline_legend(axes.ravel()[0], model=run.log["lesion"]["score"] == "model")
    fig.tight_layout()
    return fig


def cohort_table(runs: list[SceneRun]) -> pd.DataFrame:
    rows = []
    for r in runs:
        v = r.log.get("validation", {})
        c = v.get("core_vs_lesion", {})
        rows.append({"scene": r.name, "cells": r.log["cells"]["n_cells"], "Pu.1+": r.log["pu1"]["n_pos"],
                     "Pu.1+ frac": round(r.log["pu1"]["frac_pos"], 3), "lesions": r.log["lesion"]["n_lesions"],
                     "lesion sections": len(r.log["section_focus"]["lesion_sections"]),
                     "sections": r.log["n_sections"], "auto lesion mm²": round(c.get("auto_area_mm2", np.nan), 2),
                     "manual core mm²": round(c.get("manual_area_mm2", np.nan), 2),
                     "manual cores": v.get("n_manual_cores"),
                     "cores ≥50% covered": v.get("manual_cores_detected_frac"),
                     "manual area covered": round(c.get("manual_covered_by_auto", np.nan), 2),
                     "wells": r.log.get("wells", {}).get("n_groups")})
    return pd.DataFrame(rows)


def sections_table(runs: list[SceneRun]) -> pd.DataFrame:
    frames = []
    for r in runs:
        s = r.sections.copy()
        s.insert(0, "scene", r.name)
        frames.append(s)
    return pd.concat(frames, ignore_index=True)


def bar_by_section(df: pd.DataFrame, value: str, ax=None, title: str = "", fmt: str = "{:.2f}"):
    """Horizontal bars per section, coloured lesion vs control (two fixed hues), direct labels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.35 * len(df) + 1))
    d = df.sort_values(value, ascending=True)
    col = np.where(d["is_lesion_section"], "#eb6834", "#2a78d6")
    labels = d["scene"].str.replace(" scene ", "/s") + " · " + d["section_name"].astype(str)
    ax.barh(labels, d[value], color=col, height=0.7)
    for i, v in enumerate(d[value]):
        ax.text(v, i, " " + fmt.format(v), va="center", fontsize=8, color=INK)
    ax.set_title(title, loc="left")
    ax.tick_params(axis="y", labelsize=8)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(color="#eb6834", label="lesion section"), Patch(color="#2a78d6", label="control section")],
              loc="lower right", fontsize=8)
    return ax
