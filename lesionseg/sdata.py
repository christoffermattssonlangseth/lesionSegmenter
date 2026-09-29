"""Input route for SpatialData zarr stores + AnnData cell tables.

Layout produced by the Cellpose/SpatialData pipeline (spatialdata 0.7):

    <slide>.zarr/labels/<name>_masks[_filtered]/s0   int32 (y, x) label image, label == cell_id
    <slide>.zarr/shapes/<name>_shapes/shapes.parquet  per-cell polygons ('label', 'geometry' WKB)
    <slide>.zarr/shapes/CORE|NA_GW_WM|VBO|Sample_category/shapes.parquet   manual annotations
    cell_table*.h5ad   obs: cell_id, slide_name, centroid_x/y (px), area (px²), Pu1_class,
                            Sample_category, manual_annotation; X: channel means (SytoG, Pu1, Iba1, Dapi)

Everything is in full-resolution scene pixels; ``pixel_size_um`` converts to µm.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd


def find_labels_path(zarr_path: str | Path, prefer: str = "filtered") -> Path:
    zarr_path = Path(zarr_path)
    cands = sorted(Path(p).parent.parent for p in glob.glob(str(zarr_path / "labels" / "*" / "s0" / "zarr.json")))
    if not cands:
        raise FileNotFoundError(f"no labels in {zarr_path}")
    if prefer:
        pick = [c for c in cands if c.name.endswith(prefer)]
        if pick:
            return pick[0] / "s0"
    pick = [c for c in cands if not c.name.endswith("filtered")]
    return (pick[0] if pick else cands[0]) / "s0"


def load_labels(zarr_path: str | Path, prefer: str = "filtered") -> np.ndarray:
    import zarr

    arr = zarr.open(str(find_labels_path(zarr_path, prefer)), mode="r")
    return np.asarray(arr[:])


def load_shapes(zarr_path: str | Path, name: str) -> pd.DataFrame:
    """Shapes layer → DataFrame with shapely ``geometry`` column (scene px)."""
    from shapely import wkb

    p = Path(zarr_path) / "shapes" / name / "shapes.parquet"
    df = pd.read_parquet(p)
    gcol = "geometry" if "geometry" in df else df.columns[-1]
    df = df.rename(columns={gcol: "geometry"})
    df["geometry"] = [wkb.loads(g) if isinstance(g, (bytes, bytearray)) else g for g in df["geometry"]]
    return df


def list_shape_layers(zarr_path: str | Path) -> list[str]:
    return sorted(p.parent.name for p in (Path(zarr_path) / "shapes").glob("*/shapes.parquet"))


MANUAL_LAYERS = ("CORE", "NA_GW_WM", "VBO", "Sample_category")


def load_manual_annotations(zarr_path: str | Path, layers: tuple[str, ...] = MANUAL_LAYERS) -> dict[str, pd.DataFrame]:
    out = {}
    avail = list_shape_layers(zarr_path)
    for name in layers:
        if name in avail:
            out[name] = load_shapes(zarr_path, name)
    return out


def load_table(table_path: str | Path, slide_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(obs, channel-means) for one slide from an AnnData table."""
    import anndata as ad

    a = ad.read_h5ad(table_path)
    sel = (a.obs["slide_name"].astype(str) == slide_name).to_numpy()
    if not sel.any():
        raise ValueError(f"slide {slide_name!r} not in {table_path}; have {sorted(a.obs['slide_name'].unique())}")
    sub = a[sel]
    x = sub.X.toarray() if hasattr(sub.X, "toarray") else np.asarray(sub.X)
    means = pd.DataFrame(x, columns=[str(v) for v in sub.var_names], index=sub.obs_names)
    return sub.obs.copy(), means


def cells_from_spatialdata(zarr_path: str | Path, table_path: str | Path, slide_name: str, pixel_size_um: float, *,
                           pu1_col: str = "Pu1_class", pu1_positive: str = "Pu1_positive",
                           carry_cols: tuple[str, ...] = ("Sample_category", "manual_annotation", "cells_selected"),
                           contours: str = "pu1", labels_prefer: str = "filtered",
                           progress: bool = True) -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Build the lesionseg cell table from a SpatialData store + table.

    Returns ``(cells, labels, info)``. Contours come from the per-cell shapes
    layer (exact Cellpose outlines) for Pu.1⁺ cells (``contours='pu1'``), all
    cells (``'all'``) or nobody (``'none'``).
    """
    zarr_path = Path(zarr_path)
    obs, means = load_table(table_path, slide_name)
    labels = load_labels(zarr_path, labels_prefer)
    info = {"route": "spatialdata", "zarr": str(zarr_path), "table": str(table_path), "slide_name": slide_name,
            "labels": str(find_labels_path(zarr_path, labels_prefer)), "shape": list(labels.shape),
            "n_table": int(len(obs))}

    cid = obs["cell_id"].astype(int).to_numpy()
    cells = pd.DataFrame({
        "label": cid,
        "x_px": obs["centroid_x"].astype(float).to_numpy(),
        "y_px": obs["centroid_y"].astype(float).to_numpy(),
    })
    cells["x_um"] = cells["x_px"] * pixel_size_um
    cells["y_um"] = cells["y_px"] * pixel_size_um
    cells["area_um2"] = obs["area"].astype(float).to_numpy() * pixel_size_um ** 2
    for c in ("eccentricity", "solidity", "perimeter", "major_axis_length", "minor_axis_length"):
        if c in obs:
            cells[c] = obs[c].astype(float).to_numpy()
    for ch in means.columns:
        cells[f"{ch}_mean"] = means[ch].to_numpy()
    if pu1_col in obs:
        cells["pu1_pos"] = (obs[pu1_col].astype(str) == pu1_positive).to_numpy()
    else:
        cells["pu1_pos"] = False
        info["warning"] = f"{pu1_col} not in table – no Pu.1 calls"
    for c in carry_cols:
        if c in obs:
            v = obs[c]
            cells[c.lower()] = v.astype(str).where(v.notna(), None).to_numpy()
    cells["source"] = "spatialdata"

    # sanity: labels present in the mask
    in_mask = np.isin(cid, np.unique(labels))
    info["n_not_in_mask"] = int((~in_mask).sum())

    if contours != "none":
        shp_layers = [n for n in list_shape_layers(zarr_path) if n.endswith("_shapes")]
        wk = {}
        if shp_layers:
            shp = load_shapes(zarr_path, shp_layers[0])
            want = cells["pu1_pos"].to_numpy() if contours == "pu1" else np.ones(len(cells), bool)
            want_ids = set(cid[want].tolist())
            for lab, g in zip(shp["label"].astype(int), shp["geometry"], strict=True):
                if lab in want_ids:
                    if g.geom_type == "MultiPolygon":
                        g = max(g.geoms, key=lambda p: p.area)
                    from shapely.affinity import scale as shp_scale

                    wk[lab] = shp_scale(g, xfact=pixel_size_um, yfact=pixel_size_um, origin=(0, 0)).wkt
            info["contours_from"] = shp_layers[0]
        else:  # fall back to tracing the label image
            from .masks import cells_from_labels

            want = cells["pu1_pos"].to_numpy() if contours == "pu1" else None
            tab = cells_from_labels(labels, pixel_size_um, contours=True,
                                    contour_only_for=cid[want] if want is not None else None, progress=progress)
            wk = dict(zip(tab["label"], tab["contour_wkt"], strict=True))
            info["contours_from"] = "labels"
        cells["contour_wkt"] = [wk.get(lab) for lab in cid]
    cells.insert(0, "cell_id", np.arange(len(cells)))
    return cells, labels, info


def annotate_cells_with_polygons(cells: pd.DataFrame, shapes: pd.DataFrame, col: str, value_col: str | None = None,
                                 default=None) -> pd.DataFrame:
    """Point-in-polygon: label each cell with the polygon it falls in (px frame)."""
    from shapely import STRtree
    from shapely.geometry import Point

    cells = cells.copy()
    geoms = list(shapes["geometry"])
    tree = STRtree(geoms)
    pts = [Point(x, y) for x, y in zip(cells["x_px"], cells["y_px"], strict=True)]
    res = tree.query(pts, predicate="within")
    out = np.full(len(cells), default, dtype=object)
    vals = shapes[value_col].to_numpy() if value_col else np.arange(1, len(geoms) + 1)
    out[res[0]] = vals[res[1]]
    cells[col] = out
    return cells
