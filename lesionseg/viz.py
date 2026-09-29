"""Overview figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .density import DensityMaps
from .lesion import LesionResult

ZONE_CMAP = ListedColormap([(0, 0, 0, 0), (0.55, 0.55, 0.55, 0.35), (1.0, 0.78, 0.0, 0.6),
                            (1.0, 0.39, 0.0, 0.75), (0.78, 0.0, 0.0, 0.85)])
ZONE_PALETTE = {"distal": "#8c8c8c", "peri": "#ffc800", "rim": "#ff6400", "core": "#c80000",
                "background": "#000000"}


def autoscale(a, low=1, high=99.5):
    lo, hi = np.percentile(a, [low, high])
    hi = hi if hi > lo else lo + 1
    return np.clip((a.astype(np.float32) - lo) / (hi - lo), 0, 1)


def overview_figure(overview_nuc: np.ndarray, overview_pu1: np.ndarray, maps: DensityMaps, res: LesionResult,
                    cells: pd.DataFrame, title: str, path: Path, *, overview_scale: float,
                    pos_col: str = "pu1_pos") -> None:
    """2×3 panel: composite, nuclei density, Pu.1+ density, lesion score, zones, Pu.1+ cells by zone."""
    px = maps.grid.pixel_size_um
    ov_um = px / overview_scale  # µm per overview px
    extent_ov = (0, overview_nuc.shape[1] * ov_um, overview_nuc.shape[0] * ov_um, 0)
    g = maps.grid
    extent_g = (0, g.shape[1] * g.bin_um, g.shape[0] * g.bin_um, 0)

    rgb = np.zeros(overview_nuc.shape + (3,), np.float32)
    rgb[..., 2] = autoscale(overview_nuc)
    rgb[..., 0] = autoscale(overview_pu1)
    rgb[..., 1] = rgb[..., 0] * 0.6

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    ax = axes[0, 0]
    ax.imshow(rgb, extent=extent_ov)
    ax.set_title("nuclei (blue) / Pu.1 (orange)")

    ax = axes[0, 1]
    im = ax.imshow(maps.nuclei, extent=extent_g, cmap="magma")
    ax.set_title("nuclei density (cells/mm²)")
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[0, 2]
    im = ax.imshow(maps.pu1, extent=extent_g, cmap="magma")
    ax.set_title("Pu.1+ density (cells/mm²)")
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1, 0]
    vmax = np.nanpercentile(res.score[maps.tissue], 99.5) if maps.tissue.any() else 1
    im = ax.imshow(res.score, extent=extent_g, cmap="viridis", vmin=0, vmax=max(vmax, 1e-3))
    ax.contour(res.lesion_mask.astype(float), levels=[0.5], colors="w", linewidths=0.8, extent=extent_g,
               origin="upper")
    ax.set_title(f"lesion score ({res.params['score']}, thr={res.params['threshold_value']:.2f})")
    fig.colorbar(im, ax=ax, fraction=0.03)

    ax = axes[1, 1]
    ax.imshow(rgb, extent=extent_ov)
    ax.imshow(res.zones, extent=extent_g, cmap=ZONE_CMAP, vmin=0, vmax=4, interpolation="nearest")
    if res.dense_nonmyeloid is not None and res.dense_nonmyeloid.any():
        ax.contour(res.dense_nonmyeloid.astype(float), levels=[0.5], colors="#00a0ff", linewidths=1.0,
                   extent=extent_g, origin="upper")
    ax.set_title(f"zones – {res.params['n_lesions']} lesion(s); rim {res.params['rim_width_um']:g} µm, "
                 f"peri {res.params['peri_width_um']:g} µm")

    ax = axes[1, 2]
    ax.imshow(autoscale(overview_nuc), extent=extent_ov, cmap="gray")
    pos = cells[cells[pos_col]] if pos_col in cells else cells
    for zname, col in ZONE_PALETTE.items():
        sub = pos[pos["zone"] == zname]
        if len(sub):
            ax.scatter(sub["x_um"], sub["y_um"], s=1, c=col, label=f"{zname} (n={len(sub)})", linewidths=0)
    ax.legend(markerscale=8, fontsize=8, loc="lower right")
    ax.set_title("Pu.1+ cells by zone")

    for a in axes.ravel():
        a.set_xlabel("µm")
        a.set_ylabel("µm")
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def marker_histogram(cells: pd.DataFrame, info: dict, path: Path, score_col: str = "pu1_pos_score") -> None:
    sig = cells[score_col].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log1p(sig), bins=200, color="0.4")
    ax.axvline(np.log1p(info["threshold"]), color="r", label=f"thr={info['threshold']:.1f} ({info['method']})")
    ax.set_xlabel(f"log1p(background-corrected {info['channel']} intensity)")
    ax.set_ylabel("nuclei")
    ax.set_title(f"Pu.1+: {info['n_pos']}/{info['n_total']} ({100 * info['frac_pos']:.1f}%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def distance_profile(cells: pd.DataFrame, path: Path, *, pos_col: str = "pu1_pos", step_um: float = 25.0,
                     max_um: float = 500.0) -> pd.DataFrame:
    """Pu.1+ fraction and cell counts vs. signed distance to the lesion edge."""
    d = cells["dist_to_lesion_um"].replace([np.inf, -np.inf], np.nan).dropna()
    edges = np.arange(-max_um, max_um + step_um, step_um)
    sub = cells.loc[d.index]
    b = pd.cut(sub["dist_to_lesion_um"], edges)
    g = sub.groupby(b, observed=False)
    prof = pd.DataFrame({"n_cells": g.size(), "n_pos": g[pos_col].sum()})
    prof["frac_pos"] = prof["n_pos"] / prof["n_cells"].replace(0, np.nan)
    prof["center_um"] = [iv.mid for iv in prof.index]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(prof["center_um"], prof["n_pos"], width=step_um * 0.9, color="#ff6400", label="Pu.1+ cells")
    ax2 = ax.twinx()
    ax2.plot(prof["center_um"], prof["frac_pos"], "k.-", label="Pu.1+ fraction")
    ax.axvline(0, color="r", ls="--")
    ax.set_xlabel("signed distance to lesion edge (µm; <0 inside)")
    ax.set_ylabel("Pu.1+ cells")
    ax2.set_ylabel("Pu.1+ fraction")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return prof.reset_index(drop=True)
