"""Where to annotate the meninges by hand: sections and boxes that teach a meninges model the most.

Per section, boxes (``roi_um`` square, centred on the traced surface or a meningeal fold) of five kinds:

* ``swollen``        – the surface segment with the most meningeal nuclei per 100 µm (notebook 11);
* ``core_at_surface`` – where a manual lesion core comes within ``core_um`` of the surface: lesion vs
  meninges is hardest to tell apart here, and it decides what gets collected;
* ``internal``       – the densest patch of meninges deeper than 80 µm (flaps, roots, the ventral fissure):
  the rule finds these only by shape;
* ``bare``           – a surface segment with (almost) no meningeal nuclei: a tear or a very thin pia;
* ``typical``        – a segment with the section's median cellularity, as the reference.

Sections are chosen so that every animal is covered (leave-one-animal-out needs that) and lesion sections
with swelling or cores at the surface come first, spread over animals. One replicate slide per section: the
replicate slides are kept balanced unless one replicate is clearly more intact (longer traced surface).
The previews show the raw image only – never the boundary the annotator is meant to judge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

KINDS = ["core_at_surface", "swollen", "internal", "bare", "typical"]


def _pick(cands: pd.DataFrame, taken: list, sep: float) -> pd.Series | None:
    for _, row in cands.iterrows():
        if all(np.hypot(row.x_um - x, row.y_um - y) >= sep for x, y in taken):
            return row
    return None


def section_targets(run, seg: pd.DataFrame, section: str, roi_um: float = 300.0, core_um: float = 50.0,
                    thr_swollen: float | None = None) -> list[dict]:
    """Boxes for one section of ``run`` (``seg`` = ``swelling.surface_segments`` of this run)."""
    s = seg[seg["section"] == section].copy()
    if s.empty:
        return []
    c = run.cells[run.cells["section_name"].astype(str) == section]
    out, taken = [], []

    def add(kind, x, y, why):
        taken.append((x, y))
        out.append({"kind": kind, "x_um": float(x), "y_um": float(y), "why": why})

    if "manual_core_id" in c and (c["manual_core_id"] > 0).any():
        core_xy = c.loc[c["manual_core_id"] > 0, ["x_um", "y_um"]].to_numpy(float)
        d, _ = cKDTree(core_xy).query(s[["x_um", "y_um"]].to_numpy(float))
        s["d_core"] = d
        near = s[s["d_core"] <= core_um].sort_values("d_core")
        row = _pick(near, taken, roi_um)
        if row is not None:
            add("core_at_surface", row.x_um, row.y_um, f"manual core {row.d_core:.0f} µm from the surface")
    row = _pick(s.sort_values("cells_per_100um", ascending=False), taken, roi_um)
    if row is not None and (thr_swollen is None or row.cells_per_100um > thr_swollen):
        add("swollen", row.x_um, row.y_um, f"{row.cells_per_100um:.0f} meningeal nuclei / 100 µm")
    deep = c[(c["surface_tier"].astype(str) == "meninges") & (c["depth_um"] > 80)]
    if len(deep) >= 10:
        xy = deep[["x_um", "y_um"]].to_numpy(float)
        n = cKDTree(xy).query_ball_point(xy, r=roi_um / 3, return_length=True)
        cand = deep.assign(n=n).sort_values("n", ascending=False)
        row = _pick(cand, taken, roi_um)
        if row is not None:
            add("internal", row.x_um, row.y_um, f"{row.n} meningeal nuclei > 80 µm deep nearby")
    bare = s[s["n_meninges"] <= 1].sort_values(["n_meninges", "length_um"], ascending=[True, False])
    row = _pick(bare, taken, roi_um)
    if row is not None:
        add("bare", row.x_um, row.y_um, f"{int(row.n_meninges)} meningeal nuclei on {row.length_um:.0f} µm")
    med = s["cells_per_100um"].median()
    row = _pick(s.assign(dm=(s["cells_per_100um"] - med).abs()).sort_values("dm"), taken, roi_um)
    if row is not None:
        add("typical", row.x_um, row.y_um, f"{row.cells_per_100um:.1f} nuclei / 100 µm (section median)")
    px = run.px
    for o in out:
        h = roi_um / 2
        o.update(x0_px=(o["x_um"] - h) / px, y0_px=(o["y_um"] - h) / px, x1_px=(o["x_um"] + h) / px,
                 y1_px=(o["y_um"] + h) / px)
    return out


def choose_sections(seg: pd.DataFrame, runs, n_sections: int = 8) -> pd.DataFrame:
    """Every animal once, then the most informative sections spread over animals; per section the replicate
    slide that keeps the slides balanced, unless the other is clearly more intact (≥ 25 % more surface)."""
    thr = float(np.percentile(seg.loc[~seg["lesion_section"], "cells_per_100um"], 95))
    st = (seg.assign(swollen=seg["cells_per_100um"] > thr)
          .groupby(["scene", "section", "animal", "lesion_section"])
          .agg(surface_mm=("length_um", lambda v: v.sum() / 1000), swollen=("swollen", "sum"),
               bare=("n_meninges", lambda v: int((v <= 1).sum())))
          .reset_index())
    cores = []
    for r in runs:
        c = r.cells
        if "manual_core_id" in c:
            cores += [(r.name, s) for s in c.loc[c["manual_core_id"] > 0, "section_name"].astype(str).unique()]
    st["has_core"] = [(a, b) in set(cores) for a, b in zip(st["scene"], st["section"], strict=True)]
    st["score"] = st["swollen"] + 5 * st["has_core"] + 3 * st["lesion_section"]
    # one entry per section for ranking (best replicate), slides assigned afterwards
    best = st.sort_values("score", ascending=False).drop_duplicates("section").set_index("section")
    chosen: list[str] = []
    per_animal: dict[str, int] = {}
    while len(chosen) < min(n_sections, len(best)):
        cand = best.drop(chosen)
        adj = cand["score"] - 1000 * cand["animal"].map(per_animal).fillna(0)   # every animal first, then spread
        sec = adj.idxmax()
        chosen.append(sec)
        per_animal[best.loc[sec, "animal"]] = per_animal.get(best.loc[sec, "animal"], 0) + 1
    # slide per section: balance the replicate slides unless one replicate is much more intact (surface traced)
    used: dict[str, int] = {}
    rows = []
    for sec in chosen:
        reps = st[st["section"] == sec].sort_values("surface_mm", ascending=False)
        top = reps.iloc[0]
        ok = reps[reps["surface_mm"] >= 0.8 * top["surface_mm"]]
        r = ok.loc[ok["scene"].map(used).fillna(0).idxmin()] if len(ok) > 1 else top
        used[r["scene"]] = used.get(r["scene"], 0) + 1
        rows.append(r)
    return pd.DataFrame(rows).assign(swollen_threshold=thr).reset_index(drop=True)


def annotation_targets(runs, n_sections: int = 8, roi_um: float = 300.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(chosen sections, boxes) for the annotator."""
    from .meninges_model import slide_name
    from .swelling import surface_segments

    seg = pd.concat([surface_segments(r) for r in runs], ignore_index=True)
    secs = choose_sections(seg, runs, n_sections)
    byname = {r.name: r for r in runs}
    rows = []
    for s in secs.itertuples():
        r = byname[s.scene]
        for i, t in enumerate(section_targets(r, seg[seg["scene"] == s.scene], s.section, roi_um,
                                              thr_swollen=s.swollen_threshold), start=1):
            rows.append({"target_id": f"{s.section}-{i}", "scene": s.scene, "slide": slide_name(r),
                         "section": s.section, **t})
    return secs, pd.DataFrame(rows)
