"""Supervised meninges model: learn "meninges" per cell from manual MENINGES annotations.

The adaptive boundary (:mod:`lesionseg.meninges`) is a rule; this module is the route to check and replace it
once the meninges are annotated by hand (protocol: ``docs/meninges_annotation.md``). Per slide, in scene px:

* ``MENINGES_ROI``     – boxes that were annotated **exhaustively**;
* ``MENINGES``         – polygons around all meningeal tissue inside those boxes (thin lining, swollen
  infiltrate, fissures / flaps);
* ``MENINGES_UNSURE``  – optional: regions the annotator could not call; ignored.

Labels per cell: 1 inside ``MENINGES``, 0 inside a box but outside ``MENINGES``, −1 elsewhere or unsure.
Read from the slide's SpatialData store (shapes layers) or from GeoJSON files ``<slide>_<LAYER>.geojson``.

Features per cell (cell table + nucleus orientation from the label image + the traced surface), in groups
so their contribution can be measured:

* ``shape``    – own eccentricity / area / solidity; neighbours' elongated share and mean eccentricity, how
  well their long axes agree with each other (``align``) and with the cell's own (``self``);
* ``markers``  – Pu.1⁺ flag and neighbourhood share, Pu.1 / Iba1 / nuclear intensity (scene-normalised);
* ``density``  – cells per mm² of tissue within each radius;
* ``image``    – tissue context from the image: nuclear packing, channel background between nuclei,
  texture (:func:`image_features`), normalised to the tissue inside the window;
* ``geometry`` – everything that knows where the edge is: raw neighbour counts, tissue fraction around the
  cell, one-sidedness (offset of the neighbours' centroid), depth below the traced surface, alignment of
  the long axis with the surface, distance to the section edge.

``shape``, ``markers`` and ``image`` are *position-free*: a virtual cut through deep parenchyma
(``scripts/meninges_weak_labels.py``) does not raise their prediction. ``density`` is not, despite the tissue
normalisation: the tissue raster is drawn around the cells, so cells per tissue area still rise at any edge
(96 % of cells at a virtual cut called meninges).

Radii 15 / 30 / 60 µm. Model: ``HistGradientBoostingClassifier``. Evaluation: leave-one-animal-out (replicate
slides and all sections of an animal stay together), falling back to leave-one-section-out with < 3 animals.
Baseline: the rule (``surface_tier == meninges``) scored against the same labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree

ROI_LAYER, ANNOT_LAYER, UNSURE_LAYER = "MENINGES_ROI", "MENINGES", "MENINGES_UNSURE"
RADII_UM = (15, 30, 60)
ELONG = 0.85
INTENSITIES = ("Pu1_mean", "Iba1_mean", "Dapi_mean", "SytoG_mean")


# --------------------------------------------------------------------------
# annotations → labels
# --------------------------------------------------------------------------
def slide_name(run) -> str:
    return str(run.log["cells"].get("slide_name", run.name.replace(" ", "_")))


def _read_geojson(path: Path) -> list:
    from shapely.geometry import shape

    d = json.loads(Path(path).read_text())
    return [shape(f["geometry"]) for f in d.get("features", [])]


def load_annotations(run, annot_dir: str | Path | None = None) -> dict[str, list]:
    """``{layer: [shapely geometries, scene px]}`` for the three layers (missing layers → empty). GeoJSON in
    ``annot_dir`` wins over the slide's SpatialData store."""
    out = {k: [] for k in (ROI_LAYER, ANNOT_LAYER, UNSURE_LAYER)}
    name = slide_name(run)
    for layer in out:
        f = Path(annot_dir) / f"{name}_{layer}.geojson" if annot_dir else None
        if f is not None and f.exists():
            out[layer] = _read_geojson(f)
            continue
        from .sdata import list_shape_layers, load_shapes

        z = run.log["cells"].get("zarr")
        if z and layer in list_shape_layers(z):
            out[layer] = list(load_shapes(z, layer)["geometry"])
    return out


def cell_labels(cells: pd.DataFrame, annot: dict[str, list]) -> pd.DataFrame:
    """``y`` (1 meninges / 0 not / −1 unlabelled) and ``roi`` (index of the box, −1 outside) per cell."""
    import shapely

    x, y = cells["x_px"].to_numpy(float), cells["y_px"].to_numpy(float)
    roi = np.full(len(cells), -1)
    for i, g in enumerate(annot.get(ROI_LAYER, [])):
        roi[(roi < 0) & shapely.contains_xy(g, x, y)] = i
    lab = np.where(roi >= 0, 0, -1).astype(np.int8)
    if annot.get(ANNOT_LAYER):
        lab[(roi >= 0) & shapely.contains_xy(shapely.union_all(annot[ANNOT_LAYER]), x, y)] = 1
    if annot.get(UNSURE_LAYER):
        lab[shapely.contains_xy(shapely.union_all(annot[UNSURE_LAYER]), x, y)] = -1
    return pd.DataFrame({"y": lab, "roi": roi}, index=cells.index)


# --------------------------------------------------------------------------
# features
# --------------------------------------------------------------------------
def _neighbours(xy: np.ndarray, r: float) -> sparse.csr_matrix:
    """Binary adjacency within ``r`` (self included)."""
    tree = cKDTree(xy)
    a = tree.sparse_distance_matrix(tree, r, output_type="coo_matrix").tocsr()
    a.data[:] = 1.0
    a.setdiag(1.0)
    return a


def neighbour_features(xy: np.ndarray, ecc: np.ndarray, orient: np.ndarray, pu1: np.ndarray,
                       radii=RADII_UM) -> pd.DataFrame:
    """Neighbourhood shape / density / marker / one-sidedness features per cell (no surface needed)."""
    ecc = np.nan_to_num(np.asarray(ecc, float))
    z = ecc * np.exp(2j * np.nan_to_num(np.asarray(orient, float)))   # axial vector, weighted by elongation
    el = (ecc >= ELONG).astype(float)
    f = {}
    for r in radii:
        a = _neighbours(xy, r)
        n = np.asarray(a.sum(axis=1)).ravel()
        s = a @ z
        f[f"geometry_n_{r}"] = n        # raw count: halves at an edge, so it is an edge signal
        f[f"shape_el_{r}"] = a @ el / n
        f[f"shape_ecc_{r}"] = a @ ecc / n
        f[f"shape_align_{r}"] = np.abs(s) / np.maximum(a @ ecc, 1e-6)
        f[f"shape_self_{r}"] = np.real(s * np.conj(z)) / np.maximum(np.abs(s) * ecc, 1e-6) * ecc
        f[f"markers_pu1_{r}"] = a @ pu1.astype(float) / n
        f[f"geometry_offset_{r}"] = np.hypot(*((a @ xy) / n[:, None] - xy).T) / r
    return pd.DataFrame(f)


IMAGE_CHANNELS = {"nuc": "SYTOG", "pu1": "AF647", "iba1": "AF555"}
IMAGE_SIGMAS_UM = (5.0, 15.0)
IMAGE_CACHE = "meninges_image_features.parquet"


def _disk(r_px: float) -> np.ndarray:
    k = int(np.ceil(r_px))
    yy, xx = np.mgrid[-k:k + 1, -k:k + 1]
    d = (xx ** 2 + yy ** 2 <= r_px ** 2).astype(np.float32)
    return d / d.sum()


def image_features(run, refresh: bool = False) -> pd.DataFrame:
    """Tissue context around each cell from the image (1.3 µm overview), all **normalised to the tissue
    inside the window** so the empty space beyond the edge does not leak in:

    * ``image_nucfrac_<σ>``      – nuclear area fraction (nuclei touching = compact tissue);
    * ``image_<ch>_bg_<σ>``      – mean of each channel between nuclei (neuropil vs loose connective tissue),
      relative to the scene's median background;
    * ``image_nuc_sd_<σ>``       – texture: local SD of the nuclear channel;
    * ``geometry_tissue_<r>``    – tissue fraction within each neighbour radius (the edge itself; geometry).

    Per-label table, cached in the scene directory."""
    from scipy import ndimage as ndi
    from scipy.signal import fftconvolve
    from skimage.filters import threshold_otsu

    from .report import surface_tiers

    f = run.dir / IMAGE_CACHE
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    tiers, res = surface_tiers(run)
    rd = run.open_image()
    if tiers is None or rd is None:
        return pd.DataFrame({"label": run.cells["label"].to_numpy()})
    tissue = tiers > 0
    H, W = tissue.shape
    scale = run.px / res

    def fit(a):
        a = a[:H, :W].astype(np.float32)
        return np.pad(a, ((0, H - a.shape[0]), (0, W - a.shape[1])))

    imgs = {k: fit(rd.read_overview(run.scene, scale, rd.channel_index(v)))
            for k, v in IMAGE_CHANNELS.items() if v in rd.channel_names}
    c = run.cells
    iy = (c["y_um"] / res).astype(int).clip(0, H - 1).to_numpy()
    ix = (c["x_um"] / res).astype(int).clip(0, W - 1).to_numpy()
    t = tissue.astype(np.float32)
    nuc = (imgs["nuc"] > threshold_otsu(imgs["nuc"][tissue])) & tissue
    bg = (tissue & ~nuc).astype(np.float32)
    out = {"label": c["label"].to_numpy()}
    for s in IMAGE_SIGMAS_UM:
        g = lambda a, s=s: ndi.gaussian_filter(a, s / res)[iy, ix]   # noqa: E731
        wt, wb = np.maximum(g(t), 1e-6), np.maximum(g(bg), 1e-6)
        out[f"image_nucfrac_{s:g}"] = g(nuc.astype(np.float32)) / wt
        m1, m2 = g(imgs["nuc"] * t) / wt, g(imgs["nuc"] ** 2 * t) / wt
        out[f"image_nuc_sd_{s:g}"] = np.sqrt(np.maximum(m2 - m1 ** 2, 0))
        for k, im in imgs.items():
            ref = float(np.median(im[bg > 0])) or 1.0
            out[f"image_{k}_bg_{s:g}"] = g(im * bg) / wb / ref
    for r in RADII_UM:
        out[f"geometry_tissue_{r}"] = fftconvolve(t, _disk(r / res), mode="same")[iy, ix].clip(0, 1)
    df = pd.DataFrame(out)
    df.to_parquet(f)
    return df


def cell_features(run, image: bool = True) -> pd.DataFrame:
    """All feature groups for every cell of ``run`` (index aligned with ``run.cells``). ``density_<r>`` is
    cells per mm² of *tissue* within r (needs the image features' tissue fraction)."""
    from .orientation import cell_surface_angles

    c = run.cells
    ang = c[["label"]].merge(cell_surface_angles(run), on="label", how="left")
    xy = c[["x_um", "y_um"]].to_numpy(float)
    pu1 = c["pu1_pos"].to_numpy(bool)
    f = neighbour_features(xy, c["eccentricity"].to_numpy(float), ang["orient_rad"].to_numpy(float), pu1)
    f["shape_ecc"] = c["eccentricity"].to_numpy(float)
    f["shape_area"] = c["area_um2"].to_numpy(float)
    f["shape_solidity"] = c["solidity"].to_numpy(float) if "solidity" in c else np.nan
    f["markers_pu1"] = pu1.astype(float)
    in_sec = c["section_id"].to_numpy() > 0 if "section_id" in c else np.ones(len(c), bool)
    for k in INTENSITIES:
        v = c[k].to_numpy(float) if k in c else np.full(len(c), np.nan)
        f[f"markers_{k}"] = v / np.nanmedian(v[in_sec]) if np.isfinite(v[in_sec]).any() else np.nan
    f["geometry_depth"] = c["depth_um"].to_numpy(float) if "depth_um" in c else np.nan
    f["geometry_surface_align"] = (np.cos(2 * np.radians(ang["surface_angle"].to_numpy(float)))
                                   * c["eccentricity"].to_numpy(float))
    f["geometry_edge_dist"] = (c["dist_to_section_edge_um"].to_numpy(float) if "dist_to_section_edge_um" in c
                               else np.nan)
    if image:
        im = c[["label"]].merge(image_features(run), on="label", how="left").drop(columns="label")
        f = pd.concat([f, im.set_axis(f.index)], axis=1)
        for r in RADII_UM:
            if f"geometry_tissue_{r}" in f:
                area_mm2 = np.pi * r ** 2 * 1e-6 * np.maximum(f[f"geometry_tissue_{r}"], 0.05)
                f[f"density_{r}"] = f[f"geometry_n_{r}"] / area_mm2
    f.index = c.index
    return f


FEATURE_SETS = {
    "shape": ("shape",),
    "shape + markers": ("shape", "markers"),
    "position-free (shape + markers + image)": ("shape", "markers", "image"),
    "+ density (edge-sensitive)": ("shape", "markers", "image", "density"),
    "geometry only": ("geometry",),
    "all": ("shape", "markers", "density", "image", "geometry"),
}


def columns(feats: pd.DataFrame | list, groups) -> list[str]:
    cols = feats.columns if isinstance(feats, pd.DataFrame) else feats
    return [k for k in cols if k.split("_")[0] in groups]


# --------------------------------------------------------------------------
# training table, evaluation, model
# --------------------------------------------------------------------------
def training_table(runs, annot_dir: str | Path | None = None, labelled_only: bool = True) -> pd.DataFrame:
    """Features + labels + bookkeeping for every (labelled) cell of ``runs``."""
    frames = []
    for r in runs:
        annot = load_annotations(r, annot_dir)
        if not annot[ROI_LAYER]:
            continue
        c = r.cells
        lab = cell_labels(c, annot)
        in_sec = c["section_id"].to_numpy() > 0 if "section_id" in c else np.ones(len(c), bool)
        keep = in_sec & ((lab["y"] >= 0).to_numpy() if labelled_only else True)
        if not keep.any():
            continue
        f = cell_features(r)
        sec = c["section_name"].astype(str) if "section_name" in c else pd.Series("", index=c.index)
        cen = np.array([[g.centroid.x, g.centroid.y] for g in annot[ROI_LAYER]])
        roi = lab["roi"].to_numpy()[keep]
        d = f[keep].assign(y=lab["y"][keep], roi=roi, roi_x_um=cen[roi, 0] * r.px, roi_y_um=cen[roi, 1] * r.px,
                           scene=r.name, label=c["label"][keep],
                           section=sec[keep],
                           rule=(c["surface_tier"].astype(str) == "meninges")[keep].astype(int)
                           if "surface_tier" in c else -1)
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["animal"] = out["section"].str.rsplit("_", n=1).str[0]
    return out


def fit(X: pd.DataFrame, y, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                                          l2_regularization=1.0, class_weight="balanced",
                                          random_state=seed).fit(X, y)


def _binary_metrics(y, pred) -> dict:
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp, fn = (y & pred).sum(), (~y & pred).sum(), (y & ~pred).sum()
    tn = (~y & ~pred).sum()
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / max(prec + rec, 1e-9),
            "balanced_acc": (rec + tn / max(tn + fp, 1)) / 2}


def evaluate(tab: pd.DataFrame, feature_sets: dict | None = None, by: str | None = None,
             threshold: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold predictions per feature set, held out by ``by`` (animal, or section with < 3 animals).

    Returns (report: one row per feature set + the rule, out-of-fold probabilities per cell)."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    feature_sets = feature_sets or FEATURE_SETS
    by = by or ("animal" if tab["animal"].nunique() >= 3 else "section")
    y = tab["y"].to_numpy(int)
    oof = pd.DataFrame(index=tab.index)
    rows = []
    two = len(np.unique(y)) == 2
    for name, groups in feature_sets.items():
        cols = columns(tab, groups)
        p = np.full(len(tab), np.nan)
        for g in tab[by].unique():
            te = (tab[by] == g).to_numpy()
            if len(np.unique(y[~te])) < 2:
                continue
            p[te] = fit(tab.loc[~te, cols], y[~te]).predict_proba(tab.loc[te, cols])[:, 1]
        oof[name] = p
        ok = ~np.isnan(p)
        rows.append({"model": name, "held_out_by": by, "folds": tab[by].nunique(), "cells": int(ok.sum()),
                     "meninges_share": y[ok].mean() if ok.any() else np.nan,
                     "auc": roc_auc_score(y[ok], p[ok]) if two and ok.any() else np.nan,
                     "avg_precision": average_precision_score(y[ok], p[ok]) if two and ok.any() else np.nan,
                     **_binary_metrics(y[ok], p[ok] >= threshold)})
    if (tab["rule"] >= 0).all():
        rows.append({"model": "rule (adaptive boundary)", "held_out_by": "–", "folds": 0, "cells": len(tab),
                     "meninges_share": y.mean(), "auc": np.nan, "avg_precision": np.nan,
                     **_binary_metrics(y, tab["rule"].to_numpy(bool))})
    return pd.DataFrame(rows), oof


def save(clf, cols: list[str], path: Path, meta: dict | None = None) -> None:
    from .model import save_model

    save_model(clf, cols, path, meta)


def predict(run, path: Path) -> pd.DataFrame:
    """``label`` and ``p_meninges`` for every cell of ``run`` from a saved model."""
    from .model import load_model

    clf, cols, _ = load_model(path)
    f = cell_features(run)
    return pd.DataFrame({"label": run.cells["label"].to_numpy(), "p_meninges": clf.predict_proba(f[cols])[:, 1]})
