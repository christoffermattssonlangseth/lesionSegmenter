"""Supervised lesion model: learn "lesion" per density-grid bin from manual CORE annotations.

Features per grid bin (all computed from the cell table, so the model works on any
scene with cells + Pu.1 calls):

* Pu.1⁺ density, nuclear density and Pu.1⁺ fraction at three smoothing scales (20/40/80 µm)
* robust z-scores of the 40 µm maps (per scene → comparable across slides)
* Pu.1 and Iba1 mean-intensity "density" of Pu.1⁺ cells (sum of per-cell means, smoothed)
* size-normalised position: relative radial position (0 = section centre ≈ central canal,
  1 = pial edge), distance to the edge as a fraction of the section radius, and in µm

Labels: 1 inside manual CORE polygons, 0 in tissue ≥ ``neg_margin_um`` from any core
in annotated sections and everywhere in control sections; the band in between is
ignored (rim / uncertain). Model: ``HistGradientBoostingClassifier``.
Evaluation: leave-one-scene-out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from .density import Grid, count_map, density_map, robust_z

SCALES_UM = (20.0, 40.0, 80.0)


@dataclass
class SceneFeatures:
    name: str
    grid: Grid
    tissue: np.ndarray
    X: np.ndarray            # (n_tissue_bins, n_features)
    idx: np.ndarray          # flat indices of tissue bins
    names: list[str] = field(default_factory=list)


def _smooth_sum(vals_x, vals_y, weights, grid: Grid, sigma_um: float, tissue: np.ndarray) -> np.ndarray:
    counts = count_map(vals_x, vals_y, grid)
    if weights is not None:
        gx, gy = grid.to_grid(vals_x, vals_y)
        rows, cols = grid.shape
        ix = np.clip(np.floor(gx).astype(int), 0, cols - 1)
        iy = np.clip(np.floor(gy).astype(int), 0, rows - 1)
        counts = np.zeros(grid.shape, np.float32)
        np.add.at(counts, (iy, ix), weights)
    return density_map(counts, grid, sigma_um, tissue)


def build_features(cells: pd.DataFrame, grid: Grid, tissue: np.ndarray, sections: np.ndarray, *,
                   pos_col: str = "pu1_pos", pu1_int: str = "Pu1_mean", iba1_int: str = "Iba1_mean") -> SceneFeatures:
    x, y = cells["x_um"].to_numpy(), cells["y_um"].to_numpy()
    pos = cells[pos_col].to_numpy(bool)
    feats, names = [], []
    for s_um in SCALES_UM:
        nuc = _smooth_sum(x, y, None, grid, s_um, tissue)
        pu1 = _smooth_sum(x[pos], y[pos], None, grid, s_um, tissue)
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = np.where(nuc > 0, pu1 / nuc, 0.0).astype(np.float32)
        feats += [nuc, pu1, frac]
        names += [f"nuc_{s_um:g}", f"pu1_{s_um:g}", f"frac_{s_um:g}"]
        if s_um == 40.0:
            for arr, nm in ((nuc, "z_nuc"), (pu1, "z_pu1"), (frac, "z_frac")):
                feats.append(robust_z(arr, tissue)[0])
                names.append(nm)
    for col, nm in ((pu1_int, "pu1_int"), (iba1_int, "iba1_int")):
        if col in cells:
            w = np.nan_to_num(cells[col].to_numpy(float))
            feats.append(_smooth_sum(x[pos], y[pos], w[pos], grid, 40.0, tissue))
            names.append(f"{nm}_pos40")
            feats.append(_smooth_sum(x, y, w, grid, 40.0, tissue))
            names.append(f"{nm}_all40")
    # geometry, size-normalised: relative radial position (0 centre/canal → 1 pia), distance to the
    # edge as a fraction of the section radius, and the absolute edge distance
    from .assign import section_geometry

    geom = section_geometry(sections, grid)
    feats += [geom["rel_pos"], (geom["d_edge_um"] / np.maximum(geom["radius_um"], 1e-6)).astype(np.float32),
              geom["d_edge_um"]]
    names += ["rel_pos", "dist_edge_rel", "dist_edge_um"]
    idx = np.flatnonzero(tissue)
    X = np.column_stack([f.ravel()[idx] for f in feats]).astype(np.float32)
    return SceneFeatures("", grid, tissue, X, idx, names)


def make_labels(core: np.ndarray, sections: np.ndarray, annotated_sections: set[int], tissue: np.ndarray,
                grid: Grid, *, neg_margin_um: float = 200.0) -> np.ndarray:
    """int8 grid: 1 = core, 0 = negative, -1 = ignore (band around cores / non-tissue)."""
    lab = np.full(grid.shape, -1, np.int8)
    annot = np.isin(sections, list(annotated_sections))
    it = max(int(neg_margin_um / grid.bin_um), 1)
    far = ~ndi.binary_dilation(core, iterations=it) if core.any() else np.ones_like(core)
    lab[tissue & ~annot] = 0                    # control sections: all negative
    lab[tissue & annot & far] = 0               # annotated sections: far from cores
    lab[tissue & core] = 1
    return lab


def train(X: np.ndarray, y: np.ndarray, *, seed: int = 0, max_neg_per_pos: float = 10.0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if len(neg) > max_neg_per_pos * len(pos):
        neg = rng.choice(neg, int(max_neg_per_pos * len(pos)), replace=False)
    sel = np.concatenate([pos, neg])
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, class_weight="balanced", random_state=seed)
    clf.fit(X[sel], y[sel])
    return clf


def predict_map(clf, sf: SceneFeatures) -> np.ndarray:
    p = np.zeros(sf.grid.shape, np.float32)
    p.ravel()[sf.idx] = clf.predict_proba(sf.X)[:, 1]
    return p


def save_model(clf, names: list[str], path: Path, meta: dict | None = None) -> None:
    import joblib

    joblib.dump({"model": clf, "features": names, "meta": meta or {}}, path)


def load_model(path: Path):
    import joblib

    d = joblib.load(path)
    return d["model"], d["features"], d.get("meta", {})
