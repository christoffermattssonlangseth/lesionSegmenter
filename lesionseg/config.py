"""Config loading with defaults (YAML)."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "output_dir": "outputs",
    "channels": {"nuclei": "SYTOG", "pu1": "AF647", "measure": None},  # measure: None = all channels
    "overview_scale": 0.05,
    "analysis_scale": 1.0,
    # tissue.source: cells (from cell positions; default) | image | both
    "tissue": {"source": "cells", "min_cells_per_mm2": 50.0, "sigma_um": 100.0, "min_area_um2": 5e4,
               "hole_area_um2": 2e5, "dilate_um": 30.0,
               # image-based mask (used to skip empty tiles when segmenting, or source: image)
               "image": {"sigma_um": 20.0, "threshold": None, "dilate_um": 300.0, "min_area_um2": 5e4,
                         "hole_area_um2": 2e5}},
    "masks": {"contours": "pu1", "min_overlap": 0.3},  # mask route options
    "segmentation": {"method": "classical", "tile": 2048, "overlap": 128, "contours": True,
                     "params": {"nucleus_diameter_um": 7.0, "min_area_um2": 8.0}},
    # pu1.method 'mask' = take positivity from the Pu.1 mask (mask route default);
    # intensity methods (gmm/otsu/quantile/absolute) need an image
    "pu1": {"method": "mask", "value": None, "background": "annulus", "min_fold": 1.5, "min_abs": 0.0},
    "density": {"bin_um": 10.0, "sigma_um": 40.0},
    "lesion": {"score": "pu1_density", "weights": None, "threshold": {"type": "zscore", "value": 2.5},
               "min_area_um2": 5000.0, "smooth_um": 20.0, "fill_holes": True, "rim_width_um": 50.0,
               "peri_width_um": 150.0, "core_method": "distance", "core_threshold": None,
               "merge_within_um": 0.0,
               # myeloid gate – dense but Pu.1-poor regions (central canal, grey matter) are not lesions
               "min_pu1_fraction": 0.15, "min_pu1_density": 0.0, "min_lesion_pu1_fraction": 0.2,
               "dense_nonmyeloid_z": 2.0},
    # separate tissue pieces on the slide (spinal-cord cross-sections) -> section_id
    "sections": {"min_area_um2": 2e5, "merge_um": 100.0},
    "assign": {"distance_bins_um": [-100, -50, 0, 50, 100, 150, 300]},
    "export": {"cell_geojson": "pu1", "max_geojson_cells": None},  # pu1 | all | none
    "samples": [],
}


def deep_update(base: dict, upd: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is None:
        return cfg
    user = yaml.safe_load(Path(path).read_text()) or {}
    cfg = deep_update(cfg, user)
    cfg["_config_path"] = str(Path(path).resolve())
    return cfg


def sample_config(cfg: dict, sample: dict) -> dict:
    """Merge a sample entry's overrides (any top-level key) onto the global config."""
    skip = {"name", "path", "image", "cells_mask", "pu1_mask", "scene", "scenes", "pixel_size_um", "channel_names"}
    over = {k: v for k, v in sample.items() if k not in skip}
    return deep_update(cfg, over)
