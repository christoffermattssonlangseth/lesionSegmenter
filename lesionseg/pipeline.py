"""End-to-end orchestration for one sample / scene.

Two input routes:

* **masks** (primary): ``cells_mask`` (+ ``pu1_mask``) label images, optional
  ``image`` for intensities / figures.
* **image**: segment nuclei from the scan (``segmentation.method``) and classify
  Pu.1+ from intensity.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import assign, density, export, features, lesion, masks, sdata, segment, tissue, validate, viz, wells
from .config import deep_update, sample_config
from .io import SlideReader, open_slide


def _overviews(reader: SlideReader | None, scene: int, cfg: dict, shape_px: tuple[int, int]):
    """(nuc_overview, pu1_overview, real_scale, image_tissue_mask | None)."""
    if reader is None:
        return None, None, None, None
    nuc_ch = reader.channel_index(cfg["channels"]["nuclei"])
    pu1_ch = reader.channel_index(cfg["channels"]["pu1"])
    ov_nuc = reader.read_overview(scene, cfg["overview_scale"], nuc_ch)
    ov_pu1 = reader.read_overview(scene, cfg["overview_scale"], pu1_ch)
    real_scale = ov_nuc.shape[1] / shape_px[1]
    tmask = tissue.tissue_mask(ov_nuc, reader.pixel_size_um / real_scale, **cfg["tissue"]["image"])
    return ov_nuc, ov_pu1, real_scale, tmask


def run_sample(cfg: dict, *, name: str, out_dir: Path, scene: int = 0, reader: SlideReader | None = None,
               cells_mask: Path | None = None, pu1_mask: Path | None = None, pixel_size_um: float | None = None,
               sdata_path: Path | None = None, table_path: Path | None = None, slide_name: str | None = None,
               from_cells: Path | None = None, progress: bool = True) -> dict:
    """Run the pipeline for one scene and write outputs to ``out_dir``.

    Give ``cells_mask`` (+ ``pu1_mask``) for the mask route, or only ``reader``
    for the image route. ``from_cells`` re-uses an existing ``cells.parquet``
    (skips the expensive step) so downstream parameters can be iterated quickly.
    """
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if reader is None and cells_mask is None and from_cells is None and sdata_path is None:
        raise ValueError("need an image reader, a cells_mask, a SpatialData store or from_cells")

    px = reader.pixel_size_um if reader is not None else pixel_size_um
    if px is None:
        raise ValueError("pixel_size_um is required when no image is given")
    log: dict = {"sample": name, "scene": scene, "pixel_size_um": px, "config": cfg,
                 "image": str(reader.path) if reader is not None else None}

    # -- geometry of the scene --------------------------------------------
    if reader is not None:
        info = reader.scenes[scene]
        H, W = info.height, info.width
        log["scene_name"] = info.name
    elif sdata_path is not None:
        import zarr

        H, W = zarr.open(str(sdata.find_labels_path(sdata_path)), mode="r").shape
    else:
        lab_shape = masks.load_label_image(cells_mask).shape if cells_mask is not None else None
        if lab_shape is None:
            c0 = pd.read_parquet(from_cells)
            H, W = int(c0["y_px"].max()) + 1, int(c0["x_px"].max()) + 1
        else:
            H, W = lab_shape
    log["scene_shape_px"] = [H, W]

    ov_nuc, ov_pu1, real_scale, img_tmask = _overviews(reader, scene, cfg, (H, W))
    if img_tmask is not None:
        log["image_tissue_area_mm2"] = float(img_tmask.sum() * (px / real_scale) ** 2 / 1e6)

    # -- cells --------------------------------------------------------------
    pu1_cfg = dict(cfg["pu1"])
    if from_cells is not None:
        cells = pd.read_parquet(from_cells)
        log["cells"] = {"from_cells": str(from_cells), "n_cells": len(cells)}
        route = "from_cells"
    elif sdata_path is not None:
        route = "spatialdata"
        sd_cfg = cfg.get("spatialdata", {})
        cells, _labels, sinfo = sdata.cells_from_spatialdata(
            sdata_path, table_path, slide_name or name, px, pu1_col=sd_cfg.get("pu1_col", "Pu1_class"),
            pu1_positive=sd_cfg.get("pu1_positive", "Pu1_positive"), contours=cfg["masks"]["contours"],
            labels_prefer=sd_cfg.get("labels", "filtered"), progress=progress)
        del _labels
        sinfo.update({"n_cells": len(cells), "seconds": round(time.time() - t0, 1)})
        log["cells"] = sinfo
    elif cells_mask is not None:
        route = "masks"
        measure_idx = None
        if reader is not None:
            measure = cfg["channels"].get("measure")
            measure_idx = (list(range(len(reader.channel_names))) if measure is None
                           else [reader.channel_index(c) for c in measure])
        cells, minfo = masks.cells_from_masks(cells_mask, pu1_mask, px, contours=cfg["masks"]["contours"],
                                              min_overlap=cfg["masks"]["min_overlap"], reader=reader, scene=scene,
                                              measure_channels=measure_idx, progress=progress)
        minfo.update({"route": route, "n_cells": len(cells), "seconds": round(time.time() - t0, 1),
                      "pu1_mask": str(pu1_mask) if pu1_mask else None})
        log["cells"] = minfo
    else:
        route = "image"
        nuc_ch = reader.channel_index(cfg["channels"]["nuclei"])
        measure = cfg["channels"].get("measure")
        measure_idx = (list(range(len(reader.channel_names))) if measure is None
                       else [reader.channel_index(c) for c in measure])
        seg_cfg = cfg["segmentation"]
        segmenter = segment.build_segmenter(seg_cfg)
        cells = segment.segment_scene(
            reader, scene, nuclear_channel=nuc_ch, measure_channels=measure_idx, segmenter=segmenter,
            seg_params=seg_cfg.get("params"), tile=seg_cfg["tile"], overlap=seg_cfg["overlap"],
            tissue_mask=img_tmask if seg_cfg.get("skip_empty_tiles", True) else None, tissue_scale=real_scale,
            contours=seg_cfg.get("contours", True), scale=cfg.get("analysis_scale", 1.0), progress=progress)
        log["cells"] = {"route": route, "method": seg_cfg["method"], "n_cells": len(cells),
                        "seconds": round(time.time() - t0, 1)}
    if cells.empty:
        raise RuntimeError("no cells – check masks / channel names / thresholds")
    log["segmentation"] = log["cells"]  # backwards-compatible alias

    # -- Pu.1 positivity ----------------------------------------------------
    method = pu1_cfg.pop("method", "mask")
    if method == "mask":
        if "pu1_pos" not in cells:
            raise ValueError("pu1.method 'mask' but no Pu.1 mask was given; use an intensity method + image")
        n = int(cells["pu1_pos"].sum())
        log["pu1"] = {"method": "mask", "n_pos": n, "n_total": len(cells), "frac_pos": n / len(cells)}
    else:
        if reader is None:
            raise ValueError(f"pu1.method {method!r} needs an image for intensities")
        pu1_name = reader.channel_names[reader.channel_index(cfg["channels"]["pu1"])]
        if f"{pu1_name}_mean" not in cells:
            raise ValueError(f"cells table lacks {pu1_name}_mean – rerun without from_cells")
        cells, pinfo = features.classify_marker(cells, pu1_name, method=method, **pu1_cfg)
        log["pu1"] = pinfo
        viz.marker_histogram(cells, pinfo, out_dir / "pu1_histogram.png")

    # -- density / lesions / zones -----------------------------------------
    grid = density.Grid.for_scene(W, H, px, cfg["density"]["bin_um"])
    tcfg = {k: v for k, v in cfg["tissue"].items() if k not in ("source", "image")}
    maps = density.compute_density_maps(cells, grid, tissue_mask_lowres=img_tmask, sigma_um=cfg["density"]["sigma_um"],
                                        tissue_source=cfg["tissue"].get("source", "cells"), tissue_params=tcfg)
    log["tissue_area_mm2"] = float(maps.tissue.sum() * grid.bin_area_mm2())

    # sections are needed before lesion detection (model features use them); manual layers for names/validation
    manual = sdata.load_manual_annotations(sdata_path) if sdata_path is not None else {}
    section_names = None
    if "Sample_category" in manual and len(manual["Sample_category"]):
        sc = manual["Sample_category"].reset_index(drop=True)
        sections = validate.rasterize_labels(sc["geometry"], grid, px)
        section_names = {i + 1: str(v) for i, v in enumerate(sc["sample_category"])}
        log["sections_from"] = "Sample_category polygons"
    else:
        sections = assign.label_sections(maps.tissue, grid.bin_um, **cfg.get("sections", {}))
        log["sections_from"] = "tissue pieces"
    maps.extra["sections"] = sections
    if section_names:
        maps.extra["section_names"] = section_names

    lcfg = dict(cfg["lesion"])
    model_path = lcfg.pop("model", None)
    # zone widths relative to section size: rim_width_rel / peri_width_rel are fractions of the
    # section's equivalent radius (sqrt(area/π)); they override the µm widths per bin
    geom = assign.section_geometry(sections, grid)
    maps.extra["rel_pos"] = geom["rel_pos"]
    rel_widths = {}
    for zname in ("rim", "peri"):
        frac = lcfg.pop(f"{zname}_width_rel", None)
        if frac is not None:
            wmap = (geom["radius_um"] * float(frac)).astype(np.float32)
            wmap[sections == 0] = float(lcfg.get(f"{zname}_width_um", 50.0))
            lcfg[f"{zname}_width_um"] = wmap
            rel_widths[zname] = frac
    if rel_widths:
        log["zone_widths_relative"] = rel_widths
    if lcfg.get("score") == "model":
        from . import model as lesion_model

        if not model_path:
            raise ValueError("lesion.score 'model' needs lesion.model = path to a trained .joblib")
        clf, feat_names, mmeta = lesion_model.load_model(model_path)
        sf = lesion_model.build_features(cells, grid, maps.tissue, sections)
        if sf.names != list(feat_names):
            raise ValueError(f"feature mismatch: model {feat_names} vs {sf.names}")
        maps.extra["model_prob"] = lesion_model.predict_map(clf, sf)
        log["lesion_model"] = {"path": str(model_path), "meta": {k: v for k, v in mmeta.items() if k != "loso"}}
    res = lesion.detect_lesions(maps, **lcfg)
    log["lesion"] = res.params
    res.lesions.to_csv(out_dir / "lesions.csv", index=False)
    if res.dense_regions is not None:
        res.dense_regions.to_csv(out_dir / "dense_nonmyeloid_regions.csv", index=False)

    # -- lesion vs control sections (data-driven) ----------------------------------
    # A section counts as a lesion section when the automatically detected lesion area
    # is at least `lesion_min_frac` of its area (and `lesion_min_area_mm2`); the rest are
    # lesion-free controls and get no lesion context. Manual CORE polygons are NOT used
    # here – only for validation (validation_*.csv, has_manual_core column).
    scfg = cfg.get("sections", {})
    focus = scfg.get("focus", "auto")
    _, les_all = assign.assign_sections(cells.iloc[:0], res.lesions, sections, grid)
    sec_area = pd.Series(np.bincount(sections.ravel())[1:] * grid.bin_um ** 2 / 1e6,
                         index=np.arange(1, int(sections.max()) + 1))
    les_area = les_all.groupby("section_id")["area_mm2"].sum() if len(les_all) else pd.Series(dtype=float)
    les_frac = (les_area.reindex(sec_area.index).fillna(0) / sec_area).fillna(0)
    n_auto_all = les_all.groupby("section_id").size() if len(les_all) else pd.Series(dtype=int)
    core_sections: set[int] = set()
    if "CORE" in manual and len(manual["CORE"]):
        core_sections = validate.sections_with_cores(manual["CORE"]["geometry"], sections, grid, px)
    if focus == "auto":
        lesion_sections = set(int(i) for i in sec_area.index
                              if les_frac[i] >= scfg.get("lesion_min_frac", 0.02)
                              and les_area.get(i, 0.0) >= scfg.get("lesion_min_area_mm2", 0.05))
    elif focus == "manual":
        lesion_sections = set(core_sections)
    else:
        lesion_sections = set(int(i) for i in sec_area.index)
    if focus != "none" and lesion_sections != set(int(i) for i in sec_area.index):
        res = lesion.restrict_to_sections(res, sections, lesion_sections, maps)
    log["section_focus"] = {"mode": focus, "lesion_sections": sorted(lesion_sections),
                            "lesion_area_frac": {int(k): round(float(v), 4) for k, v in les_frac.items()},
                            "manual_core_sections": sorted(core_sections)}

    cells = assign.assign_cells(cells, res, grid, **cfg["assign"])
    cells, res.lesions = assign.assign_sections(cells, res.lesions, sections, grid)
    cells = assign.add_relative_geometry(cells, geom, grid)
    cells["section_has_lesion"] = cells["section_id"].isin(lesion_sections)
    if focus != "none":
        # control sections: no lesion context at all (peri zones must not bleed across the gap)
        ctrl = ~cells["section_has_lesion"].to_numpy(bool)
        cells.loc[ctrl, "zone"] = "distal"
        cells.loc[ctrl, "zone_code"] = 1
        cells.loc[ctrl, "lesion_id"] = 0
        cells.loc[ctrl, "dist_to_lesion_um"] = np.inf
        cells.loc[ctrl, "dist_to_lesion_rel"] = np.inf
    if section_names:
        cells["section_name"] = cells["section_id"].map(section_names).fillna("unassigned")
        if len(res.lesions):
            res.lesions["section_name"] = res.lesions["section_id"].map(section_names).fillna("unassigned")
    res.lesions.to_csv(out_dir / "lesions.csv", index=False)
    sec = assign.section_summary(cells, res.lesions, sections, grid.bin_um)
    if section_names:
        sec.insert(1, "section_name", sec["section_id"].map(section_names))
    sec.insert(2, "is_lesion_section", sec["section_id"].isin(lesion_sections))
    sec.insert(3, "lesion_area_frac_unrestricted", sec["section_id"].map(les_frac).round(4))
    sec["n_auto_lesions_unrestricted"] = sec["section_id"].map(n_auto_all).fillna(0).astype(int)
    if "CORE" in manual:
        sec["has_manual_core"] = sec["section_id"].isin(core_sections)
    sec.to_csv(out_dir / "section_summary.csv", index=False)
    log["n_sections"] = int(sections.max())
    log["section_summary"] = sec.to_dict(orient="records")
    summary = assign.zone_summary(cells)
    summary.to_csv(out_dir / "zone_summary.csv", index=False)
    log["zone_summary"] = summary.to_dict(orient="records")
    if "lesion_id" in cells:
        per_lesion = (cells[cells["lesion_id"] > 0].groupby(["section_id", "lesion_id", "zone"], observed=True)
                      .agg(n_cells=("cell_id", "size"), n_pu1=("pu1_pos", "sum")).reset_index())
        per_lesion.to_csv(out_dir / "per_lesion_zone_counts.csv", index=False)

    # -- plain Pu.1+ counts from the segmentation mask + Pu.1 calls (no zoning involved) -------
    pc = (cells.groupby("section_name" if "section_name" in cells else "section_id", observed=True)
          .agg(n_cells=("cell_id", "size"), n_pu1_pos=("pu1_pos", "sum"), area_pu1_um2=("area_um2", lambda a: 0.0))
          .reset_index())
    pos_area = cells[cells["pu1_pos"]].groupby("section_name" if "section_name" in cells else "section_id",
                                                observed=True)["area_um2"].sum()
    pc["area_pu1_um2"] = pc.iloc[:, 0].map(pos_area).fillna(0.0).to_numpy()
    pc["frac_pu1"] = pc["n_pu1_pos"] / pc["n_cells"]
    pc.insert(0, "scene", scene)
    pc.insert(0, "sample", name)
    pc.to_csv(out_dir / "pu1_counts.csv", index=False)
    log["pu1_counts"] = {"n_cells": int(len(cells)), "n_pu1_pos": int(cells["pu1_pos"].sum())}

    # -- manual annotations: validation only (never used to define lesions) ----------
    if manual:
        if "VBO" in manual and len(manual["VBO"]):
            cells = sdata.annotate_cells_with_polygons(cells, manual["VBO"], "in_vbo", default=0)
            cells["in_vbo"] = cells["in_vbo"].astype(int) > 0
        if "CORE" in manual and len(manual["CORE"]):
            cells = sdata.annotate_cells_with_polygons(cells, manual["CORE"], "manual_core_id", default=0)
            cells["manual_core_id"] = cells["manual_core_id"].astype(int)
            mcore = validate.rasterize(manual["CORE"]["geometry"], grid, px)
            cells = assign.add_signed_distance(cells, mcore, grid, "dist_to_manual_core_um")
            maps.extra["manual_core"] = mcore.astype(np.uint8)
        log["validation"] = validate.validate_against_manual(res, manual, grid, px, cells, out_dir)
        validate.manual_to_geojson(manual, out_dir / "manual_annotations_px.geojson")

    # -- LMD well selection ---------------------------------------------------
    wcfg = dict(cfg.get("wells") or {})
    wcfg.setdefault("focus_lesion_sections", focus != "none")
    if wcfg.pop("enabled", True):
        cells, well_table = wells.select_wells(
            cells, lesions=res.lesions, sections_xy=assign.section_centroids_um(sections, grid.bin_um), **wcfg)
        well_table.to_csv(out_dir / "wells.csv", index=False)
        cap = wells.capture_site_summary(cells, wcfg.get("groups"),
                                         edge_exclusion_um=wcfg.get("edge_exclusion_um", 100.0),
                                         exclude_vbo=wcfg.get("exclude_vbo", True))
        cap.insert(0, "scene", scene)
        cap.insert(0, "sample", name)
        cap.to_csv(out_dir / "capture_sites.csv", index=False)
        log["wells"] = {"n_groups": int(cells["well_group"].max()), "n_cells": int((cells["well_group"] > 0).sum()),
                        "target_area_um2": wcfg.get("target_area_um2", 3000.0)}

    # -- exports --------------------------------------------------------------
    export.save_cells(cells, out_dir / "cells.parquet")
    export.save_maps(maps, res, out_dir / "maps")
    export.save_zone_geojson(res, maps, out_dir / "zones_px.geojson", units="px")
    export.save_zone_geojson(res, maps, out_dir / "zones_um.geojson", units="um")
    which = cfg["export"].get("cell_geojson", "pu1")
    if which != "none" and "contour_wkt" in cells:
        only = cells["pu1_pos"] if which == "pu1" else None
        log["cell_geojson"] = export.save_cell_geojson(cells, out_dir / f"cells_{which}_px.geojson", units="px",
                                                       pixel_size_um=px, only=only,
                                                       max_cells=cfg["export"].get("max_geojson_cells"))
    prof = viz.distance_profile(cells, out_dir / "distance_profile.png")
    prof.to_csv(out_dir / "distance_profile.csv", index=False)
    if ov_nuc is None:  # no image: render density maps as the "image" backdrop
        ov_nuc, ov_pu1, real_scale = maps.nuclei, maps.pu1, grid.scale
    viz.overview_figure(ov_nuc, ov_pu1, maps, res, cells, f"{name} – scene {scene}", out_dir / "overview.png",
                        overview_scale=real_scale)
    log["seconds_total"] = round(time.time() - t0, 1)
    export.write_json(log, out_dir / "run_log.json")
    return log


def run_config(cfg: dict, *, only: list[str] | None = None, from_cells: bool = False,
               progress: bool = True) -> list[dict]:
    """Run every sample in ``cfg['samples']``.

    Sample keys: ``name``; ``image`` (CZI/TIFF, optional if masks given);
    ``cells_mask`` / ``pu1_mask`` (label images); ``pixel_size_um`` (required
    without image); ``scene`` or ``scenes``; any top-level config key as override.
    ``path`` is accepted as an alias for ``image``.
    """
    logs = []
    root = Path(cfg["output_dir"])
    base = Path(cfg.get("_config_path", ".")).parent if cfg.get("_config_path") else Path(".")

    def _p(v):
        if v is None:
            return None
        p = Path(v)
        return p if p.is_absolute() or p.exists() else (base / p)

    for s in cfg["samples"]:
        if only and s["name"] not in only:
            continue
        scfg = sample_config(cfg, s)
        img = _p(s.get("image") or s.get("path"))
        model_spec = scfg.get("lesion", {}).get("model")
        reader = (open_slide(img, channel_names=s.get("channel_names"), pixel_size_um=s.get("pixel_size_um"))
                  if img else None)
        scenes = s.get("scenes") or [s.get("scene", 0)]
        cm, pm = s.get("cells_mask"), s.get("pu1_mask")
        for sc in scenes:
            out_dir = root / s["name"] / f"scene{sc}"
            fc = out_dir / "cells.parquet" if from_cells else None
            if fc is not None and not fc.exists():
                fc = None
            # per-scene masks may be given as dicts {scene_index: path}
            cm_s = _p(cm.get(sc) if isinstance(cm, dict) else cm) if cm else None
            pm_s = _p(pm.get(sc) if isinstance(pm, dict) else pm) if pm else None
            sd = s.get("sdata")
            sd_s = _p(sd.get(sc) if isinstance(sd, dict) else sd) if sd else None
            sn = s.get("slide_name")
            sn_s = (sn.get(sc) if isinstance(sn, dict) else sn) if sn else None
            if model_spec:
                mp = model_spec.get(sc) if isinstance(model_spec, dict) else model_spec
                scfg = deep_update(scfg, {"lesion": {"model": str(_p(mp))}})
            logs.append(run_sample(scfg, name=s["name"], out_dir=out_dir, scene=sc, reader=reader, cells_mask=cm_s,
                                   pu1_mask=pm_s, pixel_size_um=s.get("pixel_size_um"), sdata_path=sd_s,
                                   table_path=_p(s.get("table") or cfg.get("table")), slide_name=sn_s,
                                   from_cells=fc, progress=progress))
    return logs
