"""Meningeal swelling along the tissue surface (report notebook 11).

Each section's outer surface (outline of the fine tissue raster, ``maps/surface_tiers.tif``) is traced,
resampled by arc length and cut into ``segment_um`` segments. Every surface-meninges nucleus (depth
≤ ``max_depth_um``; flaps / roots excluded) is assigned to its nearest surface point, so each segment gets:

* ``thickness_um``      – 90th percentile depth of its meningeal nuclei (+ a nucleus radius): how far the
  meninges reach into the section at that spot;
* ``cells_per_100um``   – meningeal nuclei per 100 µm of surface (cellularity);
* ``pu1_per_100um``     – Pu.1⁺ meningeal nuclei per 100 µm of surface (myeloid infiltrate);
* ``dist_to_lesion_um`` – distance from the segment midpoint to the nearest automatic lesion.

Caveat: thickness follows the adaptive meninges boundary, which never enters a manual lesion core, so where
an annotated core reaches the surface the meninges – and their measured swelling – are capped.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

NUC_R_UM = 3.0


def _outer_contour(mask: np.ndarray):
    from skimage.measure import find_contours

    cs = find_contours(np.pad(mask, 1).astype(float), 0.5)
    return (max(cs, key=len) - 1) if cs else None


def _resample(poly: np.ndarray, step: float) -> np.ndarray:
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(poly, axis=0).T))]
    t = np.arange(0, d[-1], step)
    return np.c_[np.interp(t, d, poly[:, 0]), np.interp(t, d, poly[:, 1])], t


def surface_segments(run, segment_um: float = 100.0, step_um: float = 5.0, max_depth_um: float = 80.0,
                     min_section_mm: float = 1.0) -> pd.DataFrame:
    """One row per surface segment of every section of ``run``."""
    from .report import surface_tiers

    tiers, res = surface_tiers(run)
    if tiers is None:
        return pd.DataFrame()
    c = run.cells
    sec = run.map("sections")
    g = run.grid.bin_um
    H, W = tiers.shape
    rows = (np.arange(H) * res / g).astype(int).clip(0, sec.shape[0] - 1)
    cols = (np.arange(W) * res / g).astype(int).clip(0, sec.shape[1] - 1)
    les = run.map("lesion_mask").astype(bool)
    d_les = ndi.distance_transform_edt(~les) * g if les.any() else np.full(les.shape, np.inf)
    names = run.sections.set_index("section_id")
    tier = c["surface_tier"].astype(str) if "surface_tier" in c else pd.Series("", index=c.index)
    men = (tier == "meninges") & (c["depth_um"] <= max_depth_um)
    out = []
    for sid in names.index:
        sm = sec == sid
        if not sm.any():
            continue
        ys, xs = np.nonzero(sm)
        r0, r1 = max(int(ys.min() * g / res) - 5, 0), min(int((ys.max() + 1) * g / res) + 5, H)
        c0, c1 = max(int(xs.min() * g / res) - 5, 0), min(int((xs.max() + 1) * g / res) + 5, W)
        m = (tiers[r0:r1, c0:c1] > 0) & (sec[rows[r0:r1]][:, cols[c0:c1]] == sid)
        m = ndi.binary_fill_holes(m)
        lab, n = ndi.label(m)
        if n == 0:
            continue
        m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)   # main piece of the section
        poly = _outer_contour(m)
        if poly is None:
            continue
        poly_um = np.c_[(poly[:, 1] + c0 + 0.5) * res, (poly[:, 0] + r0 + 0.5) * res]   # x, y µm
        pts, t = _resample(poly_um, step_um)
        if t[-1] < min_section_mm * 1000:
            continue
        seg = (t // segment_um).astype(int)
        n_seg = seg.max() + 1
        if t[-1] - n_seg * segment_um < segment_um / 2 and n_seg > 1:  # merge a short tail into the last one
            seg[seg == n_seg] = n_seg - 1
        cs = c[(c["section_id"] == sid) & men]
        if len(cs):
            _, nearest = cKDTree(pts).query(cs[["x_um", "y_um"]].to_numpy(float))
            cell_seg = seg[nearest]
        else:
            cell_seg = np.array([], int)
        for k in np.unique(seg):
            pk = pts[seg == k]
            length = float(np.hypot(*np.diff(pk, axis=0).T).sum() + step_um)
            sel = cs[cell_seg == k] if len(cs) else cs
            mid = pk[len(pk) // 2]
            gi = int(np.clip(mid[1] / g, 0, d_les.shape[0] - 1))
            gj = int(np.clip(mid[0] / g, 0, d_les.shape[1] - 1))
            out.append({
                "scene": run.name, "section_id": int(sid), "section": str(names.loc[sid].get("section_name", sid)),
                "lesion_section": bool(names.loc[sid].get("is_lesion_section", False)), "segment": int(k),
                "x_um": float(mid[0]), "y_um": float(mid[1]), "length_um": length,
                "n_meninges": len(sel), "n_pu1": int(sel["pu1_pos"].sum()) if len(sel) else 0,
                "thickness_um": float(np.percentile(sel["depth_um"], 90) + NUC_R_UM) if len(sel) >= 3 else np.nan,
                "dist_to_lesion_um": float(d_les[gi, gj]),
                "manual_core_near": bool((sel["manual_core_id"] > 0).any()) if "manual_core_id" in sel else False,
            })
    df = pd.DataFrame(out)
    if len(df):
        df["cells_per_100um"] = 100 * df["n_meninges"] / df["length_um"]
        df["pu1_per_100um"] = 100 * df["n_pu1"] / df["length_um"]
        df["pu1_share"] = df["n_pu1"] / df["n_meninges"].replace(0, np.nan)
        df["animal"] = df["section"].str.rsplit("_", n=1).str[0]
        df["level"] = df["section"].str.rsplit("_", n=1).str[-1]
    return df


def segment_class(seg: pd.DataFrame, near_um: float = 100.0) -> pd.Series:
    """``near lesion`` (lesion section, segment within ``near_um`` of a lesion), ``far from lesion``
    (lesion section, further) or ``control section``."""
    return pd.Series(np.where(~seg["lesion_section"], "control section",
                              np.where(seg["dist_to_lesion_um"] <= near_um, "near lesion", "far from lesion")),
                     index=seg.index)


def swollen_threshold(seg: pd.DataFrame, q: float = 95.0) -> float:
    """Thickness above which a segment counts as swollen: the ``q``-th percentile of control-section
    segments (what an uninflamed surface looks like)."""
    ctrl = seg.loc[~seg["lesion_section"], "thickness_um"].dropna()
    return float(np.percentile(ctrl, q)) if len(ctrl) else np.nan


def section_table(seg: pd.DataFrame, thr: float) -> pd.DataFrame:
    """Per slide × section: median thickness / cellularity / Pu.1⁺ per 100 µm and % swollen segments."""
    s = seg.assign(swollen=seg["thickness_um"] > thr)
    return (s.groupby(["scene", "section", "animal", "level", "lesion_section"])
            .agg(segments=("segment", "size"), surface_mm=("length_um", lambda v: v.sum() / 1000),
                 thickness_median_um=("thickness_um", "median"),
                 thickness_p90_um=("thickness_um", lambda v: v.quantile(0.9)),
                 cells_per_100um=("cells_per_100um", "median"), pu1_per_100um=("pu1_per_100um", "median"),
                 swollen_pct=("swollen", lambda v: 100 * v.mean()))
            .reset_index())


def paired_near_far(seg: pd.DataFrame, value: str, near_um: float = 100.0) -> pd.DataFrame:
    """Lesion sections: median ``value`` of near-lesion vs far-from-lesion segments per section
    (replicate slides averaged) and the difference."""
    d = seg[seg["lesion_section"]].assign(cls=segment_class(seg[seg["lesion_section"]], near_um))
    t = d.groupby(["scene", "section", "cls"])[value].median().unstack("cls")
    t = t.groupby("section").mean().dropna()
    t["delta"] = t["near lesion"] - t["far from lesion"]
    return t


def plot_section_swelling(ax, run, seg: pd.DataFrame, section: str, value: str = "thickness_um",
                          vmax: float | None = None):
    """Section image (dim) with each surface segment drawn as a thick dot coloured by ``value`` and the
    automatic lesion outline."""
    from matplotlib.colors import LinearSegmentedColormap

    from .report import C_LESION, read_crop_um, section_bbox_um

    s = seg[(seg["scene"] == run.name) & (seg["section"] == section)]
    sid = int(s["section_id"].iloc[0])
    x0, x1, y1, y0 = section_bbox_um(run, sid)
    cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    rgb, ext = read_crop_um(run, cx, cy, size, scale=0.2)
    if rgb is not None:
        ax.imshow(rgb.mean(axis=2) * 0.4, extent=ext, cmap="gray", vmin=0, vmax=1)
    # single-hue ramp, light = high (the background is dark)
    cmap = LinearSegmentedColormap.from_list("swell", ["#184f95", "#3987e5", "#86b6ef", "#e6f0fd"])
    vmax = vmax or float(np.nanpercentile(seg[value], 98))
    sc = ax.scatter(s["x_um"], s["y_um"], c=s[value], cmap=cmap, vmin=0, vmax=vmax, s=60, edgecolors="white",
                    linewidths=0.6, zorder=3)
    ax.contour(run.map("lesion_mask").astype(float), levels=[0.5], colors=C_LESION, linewidths=0.9,
               extent=run.extent_um, origin="upper")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{run.name} – {section}", loc="left", fontsize=10)
    return sc
