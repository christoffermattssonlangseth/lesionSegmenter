"""Write results: tables, maps, polygons (QuPath GeoJSON) and LMD shapes (py-lmd)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ZONE_NAMES
from .density import DensityMaps
from .lesion import LesionResult, zone_polygons

ZONE_COLORS = {  # RGB for QuPath classes / figures
    "distal": (120, 120, 120), "peri": (255, 200, 0), "rim": (255, 100, 0), "core": (200, 0, 0),
    "lesion": (255, 0, 0), "background": (0, 0, 0), "dense_nonmyeloid": (0, 160, 255),
}


def save_cells(cells: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(path, index=False)
    slim = cells.drop(columns=[c for c in cells.columns if c.endswith("_wkt")], errors="ignore")
    slim.to_csv(path.with_suffix(".csv"), index=False)


def save_maps(maps: DensityMaps, res: LesionResult, out_dir: Path) -> None:
    """One multi-page float32 TIFF with all grid maps + a JSON describing the pages."""
    import tifffile

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    layers = {k: v for k, v in maps.as_dict().items() if isinstance(v, np.ndarray) and v.ndim == 2}
    layers["lesion_mask"] = res.lesion_mask.astype(np.uint8)
    layers["lesion_labels"] = res.lesion_labels.astype(np.int32)
    layers["zones"] = res.zones
    if res.dense_nonmyeloid is not None:
        layers["dense_nonmyeloid"] = res.dense_nonmyeloid.astype(np.uint8)
    layers["signed_distance_um"] = np.nan_to_num(res.signed_distance_um, posinf=1e6).astype(np.float32)
    bin_um = maps.grid.bin_um
    for name, arr in layers.items():
        tifffile.imwrite(out_dir / f"{name}.tif", arr, resolution=(1e4 / bin_um, 1e4 / bin_um),
                         resolutionunit="CENTIMETER", metadata={"axes": "YX", "bin_um": bin_um})
    meta = {"bin_um": bin_um, "grid_shape": list(maps.grid.shape), "pixel_size_um": maps.grid.pixel_size_um,
            "zone_codes": {v: k for k, v in ZONE_NAMES.items()}, "layers": sorted(layers),
            "lesion_params": res.params}
    (out_dir / "maps.json").write_text(json.dumps(meta, indent=2, default=str))


def _feature(geom, props):
    from shapely.geometry import mapping

    return {"type": "Feature", "geometry": mapping(geom), "properties": props}


def _qupath_props(name: str, cls: str, extra: dict | None = None) -> dict:
    p = {"objectType": "annotation", "name": name,
         "classification": {"name": cls, "color": list(ZONE_COLORS.get(cls, (0, 200, 255)))}}
    if extra:
        p.update(extra)
    return p


def save_zone_geojson(res: LesionResult, maps: DensityMaps, path: Path, *, units: str = "px") -> None:
    """Lesion / core / rim / peri outlines as GeoJSON.

    ``units='px'`` (full-resolution scene pixels) is what QuPath expects when the
    scene is opened as an image; ``'um'`` is convenient for plotting.
    """
    feats = []
    for item in zone_polygons(res, maps.grid.bin_um, maps.grid.pixel_size_um, level=units):
        cls = item["name"]
        nm = f"{cls}_{item['lesion_id']}" if item["lesion_id"] else cls
        feats.append(_feature(item["geometry"], _qupath_props(nm, cls, {"lesion_id": item["lesion_id"]})))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def save_cell_geojson(cells: pd.DataFrame, path: Path, *, units: str = "px", pixel_size_um: float = 1.0,
                      only: pd.Series | None = None, max_cells: int | None = None) -> int:
    """Cell contours as QuPath detections, classified by zone. Returns number written."""
    from shapely import wkt
    from shapely.affinity import scale as shp_scale

    sub = cells if only is None else cells[only]
    if "contour_wkt" not in sub:
        raise ValueError("cells table has no contour_wkt column (run with contours=True)")
    sub = sub.dropna(subset=["contour_wkt"])
    if max_cells:
        sub = sub.head(max_cells)
    f = 1.0 if units == "um" else 1.0 / pixel_size_um
    feats = []
    for r in sub.itertuples(index=False):
        g = wkt.loads(r.contour_wkt)
        if f != 1.0:
            g = shp_scale(g, xfact=f, yfact=f, origin=(0, 0))
        props = {"objectType": "detection", "classification": {"name": str(r.zone),
                 "color": list(ZONE_COLORS.get(str(r.zone), (0, 200, 255)))},
                 "cell_id": int(r.cell_id), "lesion_id": int(r.lesion_id),
                 "dist_to_lesion_um": float(r.dist_to_lesion_um)}
        feats.append(_feature(g, props))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    return len(feats)


# --------------------------------------------------------------------------
# LMD (Leica laser microdissection) via py-lmd
# --------------------------------------------------------------------------
def export_lmd(cells: pd.DataFrame, path: Path, *, calibration_points_px: np.ndarray,
               group_col: str = "zone", wells: dict[str, str] | None = None,
               pixel_size_um: float = 1.0, only: pd.Series | None = None,
               dilate_um: float = 0.0, orientation_transform: np.ndarray | None = None,
               min_points: int = 4) -> dict:
    """Write an LMD-compatible XML of cell contours grouped into wells.

    Parameters
    ----------
    cells : table with ``contour_wkt`` (µm) and ``group_col``.
    calibration_points_px : (3, 2) array of the three calibration crosses in the
        **same full-resolution scene pixel frame** as the cell coordinates.
        They must be the physical marks you will re-find on the LMD.
    wells : mapping group value → well name (e.g. ``{"core": "A1", "rim": "A2",
        "peri": "A3", "distal": "A4"}``). Groups not in the mapping are skipped.
    dilate_um : grow each contour (µm) so the laser cut does not clip the nucleus.
    orientation_transform : optional 2×2 matrix passed to py-lmd (axis flips).

    Returns a summary dict {group: n_shapes}.
    """
    try:
        from lmd.lib import Collection, Shape  # py-lmd
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install py-lmd") from e
    from shapely import wkt

    calib = np.asarray(calibration_points_px, dtype=float)
    if calib.shape != (3, 2):
        raise ValueError("calibration_points_px must be (3, 2)")
    coll = Collection(calibration_points=calib)
    if orientation_transform is not None:
        coll.orientation_transform = np.asarray(orientation_transform)

    sub = cells if only is None else cells[only]
    sub = sub.dropna(subset=["contour_wkt"])
    if group_col == "reaction_name":  # cohort reaction plan (scripts/summarize_cohort.py)
        if "reaction_name" not in sub:
            raise ValueError("cells lack reaction_name – merge cells_reactions.csv onto the cell table first "
                             "(lesionseg export-lmd does this when --reactions is given)")
        sub = sub[sub["reaction_id"] > 0]
        if not wells:
            from .wells import plate_positions

            names = sub.sort_values("reaction_id")["reaction_name"].astype(str).unique().tolist()
            wells = dict(zip(names, plate_positions(len(names), rows="ABCDEFGHIJKLMNOP", cols=24), strict=True))
    if group_col == "well_name":  # only selected cells; auto plate positions in group order
        sub = sub[sub["well_group"] > 0]
        if not wells:
            from .wells import plate_positions

            names = sub.sort_values("well_group")["well_name"].astype(str).unique().tolist()
            wells = dict(zip(names, plate_positions(len(names)), strict=True))
    wells = wells or {}
    counts: dict[str, int] = {}
    for r in sub.itertuples(index=False):
        grp = str(getattr(r, group_col))
        if wells and grp not in wells:
            continue
        g = wkt.loads(r.contour_wkt)
        if dilate_um > 0:
            g = g.buffer(dilate_um, join_style=1)
        if g.geom_type != "Polygon":
            g = max(g.geoms, key=lambda p: p.area)
        pts = np.asarray(g.exterior.coords) / pixel_size_um  # µm -> px
        if len(pts) < min_points:
            continue
        coll.add_shape(Shape(pts, well=wells.get(grp, "A1"), name=f"cell{int(r.cell_id)}_{grp}"))
        counts[grp] = counts.get(grp, 0) + 1
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    coll.save(str(path))
    return counts


def write_json(obj, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)
