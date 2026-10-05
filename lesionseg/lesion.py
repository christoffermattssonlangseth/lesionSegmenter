"""Lesion score → lesion mask → zones (core / rim / peri-lesion / distal).

Definitions (all distances measured from the lesion boundary, in µm):

    distal   tissue farther than ``peri_width_um`` from any lesion
    peri     outside the lesion, within ``peri_width_um`` of the edge
    deep     outside the peri band, a further ``deep_width_um`` outward (the additional DVP step)
    rim      inside the lesion, within ``rim_width_um`` of the edge
    core     inside the lesion, deeper than ``rim_width_um``
             (or, with ``core_method='score'``, lesion bins whose score exceeds
             ``core_threshold``)

Besides the categorical zone every cell also gets a **signed distance** to the
nearest lesion edge (negative inside), so zones can be re-binned later without
re-running segmentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure

from . import ZONE_CODES
from .density import DensityMaps, robust_z
from .tissue import disk, remove_small


@dataclass
class LesionResult:
    score: np.ndarray            # float32, grid shape
    lesion_mask: np.ndarray      # bool
    lesion_labels: np.ndarray    # int32, 0 = none
    zones: np.ndarray            # uint8 zone codes (see ZONE_CODES)
    signed_distance_um: np.ndarray  # float32, negative inside lesions, +inf-free
    lesions: pd.DataFrame        # per-lesion table
    params: dict = field(default_factory=dict)
    dense_nonmyeloid: np.ndarray | None = None   # bool: hypercellular but Pu.1-poor (canal, grey matter)
    dense_labels: np.ndarray | None = None
    dense_regions: pd.DataFrame | None = None
    rim_width_map: np.ndarray | None = None      # per-bin zone widths (µm) actually used
    peri_width_map: np.ndarray | None = None
    deep_width_map: np.ndarray | None = None


# --------------------------------------------------------------------------
def lesion_score(maps: DensityMaps, *, score: str = "pu1_density",
                 weights: dict | None = None) -> tuple[np.ndarray, dict]:
    """Per-bin lesion score (robust z-score vs. all tissue bins).

    score
        ``pu1_density`` | ``pu1_fraction`` | ``nuclei_density`` | ``combined``
        (weighted sum of the z-scored maps; default weights
        ``{pu1_density: 1, nuclei_density: 0.5, pu1_fraction: 0.5}``).
    """
    t = maps.tissue
    comps = {}
    z_maps = {}
    for name, arr in (("pu1_density", maps.pu1), ("nuclei_density", maps.nuclei),
                      ("pu1_fraction", maps.pu1_fraction)):
        z, med, mad = robust_z(arr, t)
        z_maps[name] = z
        comps[name] = {"median": med, "mad_sigma": mad}
    if score == "model":
        if "model_prob" not in maps.extra:
            raise ValueError("score 'model' needs maps.extra['model_prob'] (see lesionseg.model)")
        s = maps.extra["model_prob"].astype(np.float32)
        comps["model"] = "probability map (0–1); use an absolute threshold"
    elif score == "combined":
        w = {"pu1_density": 1.0, "nuclei_density": 0.5, "pu1_fraction": 0.5}
        w.update(weights or {})
        s = sum(w[k] * z_maps[k] for k in w) / sum(abs(v) for v in w.values())
        comps["weights"] = w
    elif score in z_maps:
        s = z_maps[score]
    else:
        raise ValueError(f"unknown score {score!r}")
    s = np.where(t, s, 0.0).astype(np.float32)
    for k, v in z_maps.items():
        maps.extra[f"z_{k}"] = v
    maps.extra["lesion_score"] = s
    return s, comps


def _width_map(w, shape) -> np.ndarray:
    """Zone width as a per-bin array (scalar µm or an array of µm per bin)."""
    return np.broadcast_to(np.asarray(w, dtype=np.float32), shape) if np.ndim(w) == 0 else np.asarray(w, np.float32)


def _threshold_value(score: np.ndarray, tissue: np.ndarray, spec: dict) -> float:
    kind = spec.get("type", "zscore")
    if kind == "zscore":
        return float(spec.get("value", 2.5))
    if kind == "otsu":
        from skimage.filters import threshold_otsu

        return float(threshold_otsu(score[tissue]))
    if kind == "quantile":
        return float(np.quantile(score[tissue], spec.get("value", 0.95)))
    if kind == "absolute":  # threshold on the raw map is applied by the caller
        return float(spec["value"])
    raise ValueError(f"unknown threshold type {kind!r}")


def detect_lesions(maps: DensityMaps, *, score: str = "pu1_density", weights: dict | None = None,
                   threshold: dict | None = None, min_area_um2: float = 5000.0,
                   fill_holes: bool = True, smooth_um: float = 20.0,
                   rim_width_um: float = 50.0, peri_width_um: float = 150.0, deep_width_um: float = 150.0,
                   core_method: str = "distance", core_threshold: float | None = None,
                   merge_within_um: float = 0.0, min_pu1_fraction: float = 0.15,
                   min_pu1_density: float = 0.0, min_lesion_pu1_fraction: float = 0.2,
                   dense_nonmyeloid_z: float = 2.0) -> LesionResult:
    """Full lesion + zone computation on the density grid.

    Myeloid gate
        Hypercellular regions that are *not* myeloid (central canal ependyma,
        grey matter, meninges) must never become lesions. Every candidate bin
        therefore also needs ``pu1_fraction >= min_pu1_fraction`` and
        ``pu1_density >= min_pu1_density`` (cells/mm²), and every connected
        lesion needs a mean Pu.1+ fraction ≥ ``min_lesion_pu1_fraction``.
        Regions that are dense (nuclei z ≥ ``dense_nonmyeloid_z``) but fail
        the gate are kept separately as ``dense_nonmyeloid`` (mask + table)
        for review.
    """
    threshold = dict(threshold or {"type": "zscore", "value": 2.5})
    grid = maps.grid
    bin_um = grid.bin_um
    tissue = maps.tissue

    s, comps = lesion_score(maps, score=score, weights=weights)
    if threshold.get("type") == "absolute":
        raw = {"pu1_density": maps.pu1, "nuclei_density": maps.nuclei,
               "pu1_fraction": maps.pu1_fraction, "model": maps.extra.get("model_prob")}.get(score)
        if raw is None:
            raise ValueError("absolute threshold needs a single-map score")
        thr = float(threshold["value"])
        mask = (raw > thr) & tissue
    else:
        thr = _threshold_value(s, tissue, threshold)
        mask = (s > thr) & tissue
    # hysteresis / seeded detection: a candidate must contain at least one bin above `seed`
    # (same units as the threshold) – weakly elevated grey matter never becomes a lesion
    seed = threshold.get("seed")
    if seed is not None:
        raw_for_seed = s if threshold.get("type") != "absolute" else raw
        lab0, n0 = ndi.label(mask)
        if n0:
            has_seed = ndi.maximum(raw_for_seed, lab0, np.arange(1, n0 + 1)) > float(seed)
            keep_ids = np.arange(1, n0 + 1)[np.asarray(has_seed)]
            mask = np.isin(lab0, keep_ids)

    # myeloid gate: dense-but-not-myeloid bins (canal, grey matter) are excluded
    gate = (maps.pu1_fraction >= min_pu1_fraction) & (maps.pu1 >= min_pu1_density)
    dense = tissue & (maps.extra["z_nuclei_density"] >= dense_nonmyeloid_z)
    dense_nonmyeloid = dense & ~gate
    mask = mask & gate

    # morphological clean-up
    r = max(int(round(smooth_um / bin_um)), 1)
    if r > 0:
        mask = ndi.binary_opening(mask, structure=disk(r))
        mask = ndi.binary_closing(mask, structure=disk(r))
    if merge_within_um > 0:
        rm = int(round(merge_within_um / 2 / bin_um))
        mask = ndi.binary_closing(mask, structure=disk(rm))
    if fill_holes:
        mask = ndi.binary_fill_holes(mask)
    mask = remove_small(mask, int(min_area_um2 / bin_um ** 2))
    mask &= tissue

    labels = measure.label(mask, connectivity=1).astype(np.int32)
    rejected = []
    if labels.max() > 0 and min_lesion_pu1_fraction > 0:
        idx = np.arange(1, labels.max() + 1)
        frac = ndi.mean(maps.pu1_fraction, labels, idx)
        bad = idx[np.asarray(frac) < min_lesion_pu1_fraction]
        if len(bad):
            rejected = [int(b) for b in bad]
            mask &= ~np.isin(labels, bad)
            labels = measure.label(mask, connectivity=1).astype(np.int32)
    dense_nonmyeloid = remove_small(dense_nonmyeloid & ~mask, int(min_area_um2 / bin_um ** 2))
    dense_labels = measure.label(dense_nonmyeloid, connectivity=1).astype(np.int32)

    # signed distance (µm): >0 outside, <0 inside
    if mask.any():
        d_out = ndi.distance_transform_edt(~mask) * bin_um
        d_in = ndi.distance_transform_edt(mask) * bin_um
        sdist = np.where(mask, -d_in, d_out).astype(np.float32)
    else:
        sdist = np.full(mask.shape, np.inf, np.float32)

    rim_w = _width_map(rim_width_um, mask.shape)
    peri_w = _width_map(peri_width_um, mask.shape)
    deep_w = _width_map(deep_width_um, mask.shape)
    zones = np.zeros(mask.shape, np.uint8)
    zones[tissue] = ZONE_CODES["distal"]
    zones[tissue & ~mask & (sdist > peri_w) & (sdist <= peri_w + deep_w)] = ZONE_CODES["deep"]
    zones[tissue & ~mask & (sdist <= peri_w)] = ZONE_CODES["peri"]
    if core_method == "distance":
        core = mask & (sdist <= -rim_w)
    elif core_method == "score":
        ct = core_threshold if core_threshold is not None else thr * 2
        core = mask & (s > ct)
        core = ndi.binary_opening(core, structure=disk(max(r // 2, 1)))
    else:
        raise ValueError(f"unknown core_method {core_method!r}")
    zones[mask & ~core] = ZONE_CODES["rim"]
    zones[core] = ZONE_CODES["core"]

    # per-lesion table
    lesions = _region_table(labels, maps, s, bin_um, core, "lesion_id")
    dense_table = _region_table(dense_labels, maps, s, bin_um, None, "region_id")
    maps.extra["dense_nonmyeloid"] = dense_nonmyeloid.astype(np.uint8)

    def _w(w):
        if np.ndim(w) == 0:
            return float(w)
        return {"per_bin_um": True, "median_um": float(np.median(w[tissue])) if tissue.any() else None}

    params = {"score": score, "threshold": threshold, "threshold_value": thr, "min_area_um2": min_area_um2,
              "smooth_um": smooth_um, "rim_width_um": _w(rim_width_um), "peri_width_um": _w(peri_width_um),
              "deep_width_um": _w(deep_width_um),
              "core_method": core_method, "core_threshold": core_threshold, "bin_um": bin_um,
              "components": comps, "n_lesions": int(labels.max()),
              "min_pu1_fraction": min_pu1_fraction, "min_pu1_density": min_pu1_density,
              "min_lesion_pu1_fraction": min_lesion_pu1_fraction, "rejected_low_pu1": rejected,
              "n_dense_nonmyeloid": int(dense_labels.max())}
    res = LesionResult(s, mask, labels, zones, sdist, lesions, params, dense_nonmyeloid=dense_nonmyeloid,
                       dense_labels=dense_labels, dense_regions=dense_table)
    res.rim_width_map = rim_w
    res.peri_width_map = peri_w
    res.deep_width_map = deep_w
    return res


def _region_table(labels: np.ndarray, maps: DensityMaps, score: np.ndarray, bin_um: float,
                  core: np.ndarray | None, id_col: str) -> pd.DataFrame:
    n = int(labels.max())
    if n == 0:
        return pd.DataFrame(columns=[id_col, "area_um2", "area_mm2", "centroid_x_um", "centroid_y_um",
                                     "equiv_diameter_um", "mean_score", "max_score", "mean_pu1_density",
                                     "mean_nuclei_density", "mean_pu1_fraction"])
    idx = np.arange(1, n + 1)
    area = np.bincount(labels.ravel(), minlength=n + 1)[1:].astype(float)
    cy, cx = np.array(ndi.center_of_mass(np.ones_like(labels), labels, idx)).T
    df = pd.DataFrame({
        id_col: idx,
        "area_um2": area * bin_um ** 2,
        "area_mm2": area * bin_um ** 2 / 1e6,
        "centroid_x_um": cx * bin_um,
        "centroid_y_um": cy * bin_um,
        "equiv_diameter_um": np.sqrt(4 * area / np.pi) * bin_um,
        "mean_score": ndi.mean(score, labels, idx),
        "max_score": ndi.maximum(score, labels, idx),
        "mean_pu1_density": ndi.mean(maps.pu1, labels, idx),
        "mean_nuclei_density": ndi.mean(maps.nuclei, labels, idx),
        "mean_pu1_fraction": ndi.mean(maps.pu1_fraction, labels, idx),
    })
    if core is not None:
        core_area = np.bincount(labels[core].ravel(), minlength=n + 1)[1:].astype(float)
        df["core_area_um2"] = core_area * bin_um ** 2
        df["rim_area_um2"] = (area - core_area) * bin_um ** 2
    return df


def restrict_to_sections(res: LesionResult, sections: np.ndarray, keep: set[int], maps: DensityMaps) -> LesionResult:
    """Drop lesions whose bins lie outside the ``keep`` sections and recompute zones / distances.

    Used to confine lesion analysis to the data-driven lesion sections; other
    sections become plain (distal) control tissue.
    """
    if not keep:
        return res
    return clip_lesions(res, np.isin(sections, list(keep)), maps)


def clip_lesions(res: LesionResult, allowed: np.ndarray, maps: DensityMaps, *, meninges: np.ndarray | None = None,
                 note: str | None = None) -> LesionResult:
    """Keep only lesion bins inside ``allowed`` and recompute zones / distances.

    ``meninges`` (bool grid) marks bins that are tissue but not parenchyma; they get the
    ``meninges`` zone and never become core / rim / peri / deep.
    """
    mask = res.lesion_mask & allowed
    labels = measure.label(mask, connectivity=1).astype(np.int32)
    bin_um = res.params["bin_um"]
    tissue = res.zones > 0
    if mask.any():
        sdist = np.where(mask, -ndi.distance_transform_edt(mask), ndi.distance_transform_edt(~mask)) * bin_um
        sdist = sdist.astype(np.float32)
    else:
        sdist = np.full(mask.shape, np.inf, np.float32)
    # rebuild zones keeping the original core definition inside surviving lesions
    core = (res.zones == ZONE_CODES["core"]) & mask
    zones = np.zeros(mask.shape, np.uint8)
    zones[tissue] = ZONE_CODES["distal"]
    if meninges is not None:
        tissue = tissue & ~meninges
    peri_w = res.peri_width_map
    if peri_w is None:
        peri_w = _width_map(res.params["peri_width_um"], mask.shape)
    deep_w = res.deep_width_map
    if deep_w is None:
        deep_w = _width_map(res.params.get("deep_width_um", 150.0), mask.shape)
    zones[tissue & ~mask & (sdist > peri_w) & (sdist <= peri_w + deep_w)] = ZONE_CODES["deep"]
    zones[tissue & ~mask & (sdist <= peri_w)] = ZONE_CODES["peri"]
    zones[mask & ~core] = ZONE_CODES["rim"]
    zones[core] = ZONE_CODES["core"]
    if meninges is not None:
        zones[meninges & (res.zones > 0)] = ZONE_CODES["meninges"]
    lesions = _region_table(labels, maps, res.score, bin_um, core, "lesion_id")
    params = dict(res.params)
    params["n_lesions"] = int(labels.max())
    if note:
        params.setdefault("clips", []).append(note)
    out = LesionResult(res.score, mask, labels, zones, sdist, lesions, params, dense_nonmyeloid=res.dense_nonmyeloid,
                       dense_labels=res.dense_labels, dense_regions=res.dense_regions)
    out.rim_width_map, out.peri_width_map, out.deep_width_map = res.rim_width_map, peri_w, deep_w
    return out


def zone_polygons(res: LesionResult, bin_um: float, pixel_size_um: float, level: str = "um",
                  sections: np.ndarray | None = None, section_names: dict | None = None) -> list[dict]:
    """Vectorise lesion outlines + zone rings to shapely polygons.

    Returns a list of dicts ``{"name", "lesion_id", "section", "geometry"}``; coordinates
    are µm (``level='um'``) or full-res pixels (``level='px'``). ``lesion`` / ``core`` /
    ``rim`` are per lesion; ``peri`` and ``deep`` bands are per connected component
    (they may wrap several lesions), each attributed to the section it lies in.
    """
    from shapely.affinity import affine_transform
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    f = bin_um if level == "um" else bin_um / pixel_size_um
    out = []

    def shp_affine(g, s):
        return affine_transform(g, [s, 0, 0, s, -0.5 * s, -0.5 * s])

    def _section_of(mask):
        if sections is None or not mask.any():
            return None
        ids = sections[mask]
        ids = ids[ids > 0]
        if ids.size == 0:
            return None
        sid = int(np.bincount(ids).argmax())
        return (section_names or {}).get(sid, f"S{sid}")

    min_area = 2 * f * f  # drop sub-bin artefacts (< 2 grid bins)

    def _polys(mask):
        """Polygons WITH holes. Contours are classified by nesting depth (even = shell, odd = hole)."""
        padded = np.pad(mask, 1)
        cands = []
        for c in measure.find_contours(padded.astype(np.float32), 0.5):
            if len(c) < 4:
                continue
            p = Polygon(np.column_stack([c[:, 1], c[:, 0]]))  # grid coords (col, row), padded frame
            if not p.is_valid:
                p = p.buffer(0)
            if p.geom_type == "MultiPolygon":
                p = max(p.geoms, key=lambda q: q.area)
            if p.is_empty or p.geom_type != "Polygon" or p.area < 2:
                continue
            cands.append(Polygon(p.exterior.coords))
        cands.sort(key=lambda q: q.area, reverse=True)
        pts = [q.representative_point() for q in cands]
        depth = [sum(1 for j in range(i) if cands[j].contains(pts[i])) for i in range(len(cands))]
        shells = [i for i, d in enumerate(depth) if d % 2 == 0]
        holes = [i for i, d in enumerate(depth) if d % 2 == 1]
        out_polys = []
        for i in shells:
            # holes directly inside this shell: contained, and no smaller shell between them
            hs = [cands[h] for h in holes if depth[h] == depth[i] + 1 and cands[i].contains(pts[h])]
            geom = Polygon(cands[i].exterior.coords, [h.exterior.coords for h in hs])
            if not geom.is_valid:
                geom = geom.buffer(0)
            if geom.is_empty:
                continue
            geom = shp_affine(geom, f)  # padded grid coords -> µm / px
            if geom.area >= min_area:
                out_polys.append(geom)
        return out_polys

    for lab in range(1, res.lesion_labels.max() + 1):
        m = res.lesion_labels == lab
        sec = _section_of(m)
        for p in _polys(m):
            out.append({"name": "lesion", "lesion_id": lab, "section": sec, "geometry": p})
        for zname in ("core", "rim"):
            zm = m & (res.zones == ZONE_CODES[zname])
            if zm.any():
                parts = _polys(zm)
                if parts:
                    out.append({"name": zname, "lesion_id": lab, "section": sec, "geometry": unary_union(parts)})
    for zname in ("peri", "deep"):
        if zname not in ZONE_CODES:
            continue
        band = res.zones == ZONE_CODES[zname]
        if not band.any():
            continue
        comp, n = ndi.label(band)
        for k in range(1, n + 1):
            cm = comp == k
            parts = _polys(cm)
            if not parts:
                continue
            geom = unary_union(parts)
            # lesions this band component touches (dilate by one bin)
            touch = np.unique(res.lesion_labels[ndi.binary_dilation(cm, iterations=max(int(200 / bin_um), 1))])
            touch = [int(t) for t in touch if t > 0]
            out.append({"name": zname, "lesion_id": touch[0] if len(touch) == 1 else 0, "section": _section_of(cm),
                        "geometry": geom, "lesion_ids": touch})
    if res.dense_labels is not None:
        for lab in range(1, int(res.dense_labels.max()) + 1):
            for p in _polys(res.dense_labels == lab):
                out.append({"name": "dense_nonmyeloid", "lesion_id": lab, "section": None, "geometry": p})
    return [o for o in out if o["geometry"] is not None and not o["geometry"].is_empty]
