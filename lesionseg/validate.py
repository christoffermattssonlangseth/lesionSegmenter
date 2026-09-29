"""Compare automatic lesion zones with manual annotation polygons."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .density import Grid
from .lesion import LesionResult


def rasterize(shapes, grid: Grid, pixel_size_um: float) -> np.ndarray:
    """Boolean grid mask from polygons given in scene px."""
    from skimage.draw import polygon as draw_polygon

    f = pixel_size_um / grid.bin_um  # px -> grid
    m = np.zeros(grid.shape, bool)
    for g in shapes:
        polys = g.geoms if g.geom_type == "MultiPolygon" else [g]
        for p in polys:
            x, y = np.asarray(p.exterior.coords).T
            rr, cc = draw_polygon(y * f, x * f, shape=grid.shape)
            m[rr, cc] = True
            for hole in p.interiors:
                hx, hy = np.asarray(hole.coords).T
                rr, cc = draw_polygon(hy * f, hx * f, shape=grid.shape)
                m[rr, cc] = False
    return m


def rasterize_labels(shapes, grid: Grid, pixel_size_um: float) -> np.ndarray:
    """int32 grid: polygon index (1-based) per bin, 0 outside; later polygons win on overlap."""
    out = np.zeros(grid.shape, np.int32)
    for i, g in enumerate(shapes, start=1):
        out[rasterize([g], grid, pixel_size_um)] = i
    return out


def sections_with_cores(core_geoms, sections: np.ndarray, grid: Grid, pixel_size_um: float) -> set[int]:
    """Section ids whose area contains at least one manual CORE polygon (by core centroid)."""
    f = pixel_size_um / grid.bin_um
    rows, cols = grid.shape
    out = set()
    for g in core_geoms:
        c = g.centroid
        r, q = int(c.y * f), int(c.x * f)
        if 0 <= r < rows and 0 <= q < cols and sections[r, q] > 0:
            out.add(int(sections[r, q]))
        else:  # centroid outside any section polygon: fall back to any bin the core covers
            m = rasterize([g], grid, pixel_size_um)
            ids = np.unique(sections[m])
            out.update(int(i) for i in ids if i > 0)
    return out


def compare_masks(auto: np.ndarray, manual: np.ndarray, tissue: np.ndarray, bin_um: float) -> dict:
    a, m = auto & tissue, manual & tissue
    inter = (a & m).sum()
    union = (a | m).sum()
    area = bin_um ** 2 / 1e6
    return {
        "auto_area_mm2": float(a.sum() * area), "manual_area_mm2": float(m.sum() * area),
        "iou": float(inter / union) if union else float("nan"),
        "manual_covered_by_auto": float(inter / m.sum()) if m.sum() else float("nan"),
        "auto_inside_manual": float(inter / a.sum()) if a.sum() else float("nan"),
    }


def validate_against_manual(res: LesionResult, manual: dict[str, pd.DataFrame], grid: Grid, pixel_size_um: float,
                            cells: pd.DataFrame, out_dir: Path) -> dict:
    """Write validation tables; returns a summary dict.

    * lesion mask vs manual ``CORE`` polygons (IoU, coverage)
    * per manual core: fraction covered by auto lesion, dominant auto zone
    * per-cell confusion: manual_annotation × auto zone
    """
    out_dir = Path(out_dir)
    summary: dict = {}
    if "CORE" in manual and len(manual["CORE"]):
        core_mask = rasterize(manual["CORE"]["geometry"], grid, pixel_size_um)
        summary["core_vs_lesion"] = compare_masks(res.lesion_mask, core_mask, res.zones > 0, grid.bin_um)
        summary["core_vs_core"] = compare_masks(res.zones == 4, core_mask, res.zones > 0, grid.bin_um)
        rows = []
        for i, g in enumerate(manual["CORE"]["geometry"], start=1):
            m = rasterize([g], grid, pixel_size_um)
            n = m.sum()
            if n == 0:
                continue
            zones = res.zones[m]
            rows.append({"manual_core_id": i, "area_um2": float(n * grid.bin_um ** 2),
                         "frac_in_auto_lesion": float(res.lesion_mask[m].mean()),
                         "frac_core": float((zones == 4).mean()), "frac_rim": float((zones == 3).mean()),
                         "frac_peri": float((zones == 2).mean()), "frac_deep": float((zones == 5).mean()),
                         "frac_distal": float((zones == 1).mean()),
                         "auto_lesion_ids": sorted(set(np.unique(res.lesion_labels[m]).tolist()) - {0})})
        per_core = pd.DataFrame(rows)
        per_core.to_csv(out_dir / "validation_manual_cores.csv", index=False)
        summary["n_manual_cores"] = len(per_core)
        detected = (per_core["frac_in_auto_lesion"] > 0.5).mean() if len(per_core) else None
        summary["manual_cores_detected_frac"] = float(detected) if detected is not None else None
    if "manual_annotation" in cells:
        ct = pd.crosstab(cells["manual_annotation"].fillna("none"), cells["zone"])
        ct.to_csv(out_dir / "validation_cell_confusion.csv")
        summary["cell_confusion"] = ct.to_dict()
    return summary


def manual_to_geojson(manual: dict[str, pd.DataFrame], path: Path, *, units: str = "px",
                      pixel_size_um: float = 1.0) -> None:
    import json

    from shapely.affinity import scale as shp_scale
    from shapely.geometry import mapping

    colors = {"CORE": (0, 255, 0), "NA_GW_WM": (0, 200, 255), "VBO": (255, 0, 255), "Sample_category": (255, 255, 0)}
    feats = []
    for name, df in manual.items():
        for i, row in df.iterrows():
            g = row["geometry"]
            if units == "um":
                g = shp_scale(g, xfact=pixel_size_um, yfact=pixel_size_um, origin=(0, 0))
            label = str(row.get("sample_category", f"{name}_{i}"))
            feats.append({"type": "Feature", "geometry": mapping(g),
                          "properties": {"objectType": "annotation", "name": label,
                                         "classification": {"name": f"manual_{name}",
                                                            "color": list(colors.get(name, (200, 200, 200)))}}})
    Path(path).write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
