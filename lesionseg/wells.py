"""Select Pu.1⁺ cells per section and zone for the LMD wells (DVP).

Mirrors the collaborator's protocol (``docs/collaborator/code/2_2_Annotation.ipynb``):
per section, a fixed **target area** (default 3000 µm²) of Pu.1⁺ cells is picked
for every group – core, distance rings from the lesion edge, grey/white matter –
after excluding cells near the tissue edge, inside VBO, and (optionally) cells on
the outward side of their lesion. Cells far from the group's median size are
dropped so each well holds comparable cells.

Groups are declared in config; every key is a filter (all must hold):

    - {name: core,        zone: [core]}
    - {name: rim,         zone: [rim]}
    - {name: ring_0_100,  dist_um: [0, 100]}        # signed distance to lesion edge, [lo, hi)
    - {name: ring_0_10pct, dist_rel: [0, 0.1]}      # …as a fraction of the section radius
    - {name: central,     rel_pos: [0, 0.3]}        # radial position 0 (centre) → 1 (pia)
    - {name: ring_100_200, dist_um: [100, 200]}
    - {name: GM,          manual: [GM]}             # manual_annotation column
    - {name: WM,          manual: [WM], zone: [distal]}
    - {name: mc_ring_0_100, manual_dist_um: [0, 100]}  # rings from the *manual* CORE boundary
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_GROUPS = [
    {"name": "core", "zone": ["core"]},
    {"name": "rim", "zone": ["rim"]},
    {"name": "ring_0_100", "dist_um": [0, 100]},
    {"name": "ring_100_200", "dist_um": [100, 200]},
    {"name": "ring_200_400", "dist_um": [200, 400]},
    {"name": "GM", "manual": ["GM"], "zone": ["distal"]},
    {"name": "WM", "manual": ["WM"], "zone": ["distal"]},
]


def _group_mask(cells: pd.DataFrame, g: dict) -> np.ndarray:
    m = np.ones(len(cells), bool)
    if "zone" in g:
        m &= cells["zone"].astype(str).isin(g["zone"]).to_numpy()
    if "dist_um" in g:
        lo, hi = g["dist_um"]
        d = cells["dist_to_lesion_um"].to_numpy(float)
        m &= (d >= lo) & (d < hi)
    if "dist_rel" in g:  # signed lesion distance / section radius
        lo, hi = g["dist_rel"]
        col = "dist_to_lesion_rel"
        d = cells[col].to_numpy(float) if col in cells else np.full(len(cells), np.inf)
        m &= (d >= lo) & (d < hi)
    if "rel_pos" in g:  # radial position 0 (centre) → 1 (pia)
        lo, hi = g["rel_pos"]
        d = cells["rel_pos"].to_numpy(float) if "rel_pos" in cells else np.full(len(cells), np.nan)
        m &= (d >= lo) & (d < hi)
    if "manual_dist_um" in g:  # distance to the manual CORE boundary (collaborator's reference)
        lo, hi = g["manual_dist_um"]
        col = "dist_to_manual_core_um"
        d = cells[col].to_numpy(float) if col in cells else np.full(len(cells), np.inf)
        m &= (d >= lo) & (d < hi)
    if "manual" in g:
        col = cells["manual_annotation"] if "manual_annotation" in cells else pd.Series("none", index=cells.index)
        m &= col.astype(str).isin(g["manual"]).to_numpy()
    if "section" in g:
        m &= cells["section_name"].astype(str).isin(g["section"]).to_numpy()
    return m


def _inward(cells: pd.DataFrame, lesions: pd.DataFrame, sections_xy: dict) -> np.ndarray:
    """True if the cell lies on the side of its lesion that faces the section centroid."""
    ok = np.ones(len(cells), bool)
    if lesions is None or not len(lesions) or "lesion_id" not in cells:
        return ok
    lc = lesions.set_index("lesion_id")[["centroid_x_um", "centroid_y_um"]]
    lid = cells["lesion_id"].to_numpy()
    has = (lid > 0) & np.isin(lid, lc.index)
    if not has.any():
        return ok
    L = lc.loc[lid[has]].to_numpy()
    S = np.array([sections_xy.get(int(s), (np.nan, np.nan)) for s in cells.loc[has, "section_id"]])
    C = cells.loc[has, ["x_um", "y_um"]].to_numpy()
    inward = S - L
    cell_v = C - L
    dot = (inward * cell_v).sum(1)
    ok[np.nonzero(has)[0]] = np.where(np.isfinite(dot), dot > 0, True)
    return ok


def select_wells(cells: pd.DataFrame, *, groups: list[dict] | None = None, target_area_um2: float = 3000.0,
                 size_filter_sd: float | None = 1.0, order: str = "spatial", seed: int = 0,
                 pos_col: str = "pu1_pos", section_col: str = "section_name", edge_exclusion_um: float = 100.0,
                 exclude_vbo: bool = True, inward_only: bool = False, lesions: pd.DataFrame | None = None,
                 sections_xy: dict | None = None, min_cells: int = 5,
                 focus_lesion_sections: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign ``well_group`` (int, 0 = not selected) and ``well_name`` to cells.

    Returns ``(cells, wells)`` where ``wells`` summarises each group
    (section, name, n_cells, area_um2, shortfall). With ``focus_lesion_sections`` (and
    a ``section_has_lesion`` column), lesion groups are only drawn from sections
    classified as lesion sections; GM / WM control groups come from any section.
    """
    groups = groups or DEFAULT_GROUPS
    cells = cells.copy()
    cells["well_group"] = 0
    cells["well_name"] = None
    rng = np.random.default_rng(seed)

    # copy! .to_numpy() on a bool column can be a view of the DataFrame and `&=` would mutate it
    base = cells[pos_col].to_numpy(bool).copy() if pos_col in cells else np.ones(len(cells), bool)
    if edge_exclusion_um > 0 and "dist_to_section_edge_um" in cells:
        base &= cells["dist_to_section_edge_um"].to_numpy(float) >= edge_exclusion_um
    if exclude_vbo and "in_vbo" in cells:
        base &= ~cells["in_vbo"].to_numpy(bool)
    if inward_only:
        base &= _inward(cells, lesions, sections_xy or {})
    if section_col not in cells:
        cells[section_col] = cells["section_id"].map(lambda s: f"S{s}")
    sections = [s for s in cells.loc[base, section_col].dropna().unique() if str(s) not in ("unassigned", "0", "None")]

    has_les = cells["section_has_lesion"].to_numpy(bool) if "section_has_lesion" in cells else None
    rows = []
    gid = 0
    for sec in sorted(sections, key=str):
        in_sec = base & (cells[section_col].astype(str) == str(sec)).to_numpy()
        for g in groups:
            m = in_sec & _group_mask(cells, g) & (cells["well_group"].to_numpy() == 0)
            # lesion-related groups (zone / distance filters) only in sections that carry lesions
            lesion_group = ("dist_um" in g or "manual_dist_um" in g or bool(set(g.get("zone", [])) - {"distal"}))
            if has_les is not None and focus_lesion_sections and lesion_group:
                m &= has_les
            idx = np.nonzero(m)[0]
            if len(idx) < min_cells:
                rows.append({"section": sec, "group": g["name"], "well_group": 0, "n_cells": int(len(idx)),
                             "area_um2": 0.0, "n_available": int(len(idx)), "note": "too few cells"})
                continue
            sub = cells.iloc[idx]
            if size_filter_sd is not None:
                med, sd = sub["area_um2"].median(), sub["area_um2"].std()
                keep = (sub["area_um2"] >= med - size_filter_sd * sd) & (sub["area_um2"] <= med + size_filter_sd * sd)
                sub = sub[keep]
            if order == "spatial":
                sub = sub.sort_values(["x_um", "y_um"])
            elif order == "random":
                sub = sub.iloc[rng.permutation(len(sub))]
            elif order == "central":  # most lesion-central first (deepest inside for core, closest for rings)
                sub = sub.sort_values("dist_to_lesion_um")
            cum = sub["area_um2"].cumsum().to_numpy()
            n_take = int(np.searchsorted(cum, target_area_um2, side="right"))
            n_take = max(min(n_take, len(sub)), 0)
            chosen = sub.index[:n_take]
            if n_take == 0:
                rows.append({"section": sec, "group": g["name"], "well_group": 0, "n_cells": 0, "area_um2": 0.0,
                             "n_available": int(len(sub)), "note": "no cells"})
                continue
            gid += 1
            cells.loc[chosen, "well_group"] = gid
            cells.loc[chosen, "well_name"] = f"{sec}|{g['name']}"
            area = float(cells.loc[chosen, "area_um2"].sum())
            rows.append({"section": sec, "group": g["name"], "well_group": gid, "n_cells": int(n_take),
                         "area_um2": round(area, 1), "n_available": int(len(sub)),
                         "note": "" if area >= 0.95 * target_area_um2 else f"short: {area:.0f} µm²"})
    wells = pd.DataFrame(rows)
    return cells, wells


MANUAL_GROUPS = ("GM", "WM", "vbo", "core")


def capture_site_summary(cells: pd.DataFrame, groups: list[dict] | None = None, *, pos_col: str = "pu1_pos",
                         section_col: str = "section_name", edge_exclusion_um: float = 100.0,
                         exclude_vbo: bool = True) -> pd.DataFrame:
    """Pu.1⁺ cells per capture site (section × compartment) – the material available for DVP.

    Rows: every section × every well group (lesion compartments only in lesion sections) plus the
    manual-annotation compartments (GM, WM, vbo, manual core, unannotated). Columns:

    * ``n_pu1_raw``      Pu.1⁺ cells in the compartment
    * ``n_pu1_eligible`` …after edge exclusion / VBO exclusion (what a well can draw from)
    * ``area_pu1_eligible_um2`` their summed nuclear area
    * ``n_selected`` / ``area_selected_um2``  cells actually placed in a well (if selection ran)
    * ``n_cells_all``    all cells (any Pu.1 status) in the compartment, for the fraction
    """
    groups = groups or DEFAULT_GROUPS
    pos = cells[pos_col].to_numpy(bool)
    elig = pos.copy()
    if edge_exclusion_um > 0 and "dist_to_section_edge_um" in cells:
        elig &= cells["dist_to_section_edge_um"].to_numpy(float) >= edge_exclusion_um
    if exclude_vbo and "in_vbo" in cells:
        elig &= ~cells["in_vbo"].to_numpy(bool)
    sel = cells["well_group"].to_numpy() > 0 if "well_group" in cells else np.zeros(len(cells), bool)
    area = cells["area_um2"].to_numpy(float)
    has_les = cells["section_has_lesion"].to_numpy(bool) if "section_has_lesion" in cells else np.ones(len(cells), bool)
    secs = cells[section_col].astype(str)
    rows = []
    for sec in sorted(secs.unique()):
        if sec in ("unassigned", "None", "nan", "0"):
            continue
        in_sec = (secs == sec).to_numpy()
        les_sec = bool(has_les[in_sec].any())
        comps = []
        for g in groups:
            lesion_group = ("dist_um" in g or "dist_rel" in g or "manual_dist_um" in g
                            or bool(set(g.get("zone", [])) - {"distal"}))
            if lesion_group and not les_sec:
                continue
            comps.append((g["name"], "well group", _group_mask(cells, g)))
        if "manual_annotation" in cells:
            ma = cells["manual_annotation"].astype(str)
            for m in MANUAL_GROUPS:
                comps.append((f"manual:{m}", "manual annotation", (ma == m).to_numpy()))
            comps.append(("manual:unannotated", "manual annotation", (~ma.isin(MANUAL_GROUPS)).to_numpy()))
        comps.append(("all", "whole section", np.ones(len(cells), bool)))
        for name, kind, m in comps:
            mm = in_sec & m
            rows.append({"section": sec, "lesion_section": les_sec, "compartment": name, "kind": kind,
                         "n_cells_all": int(mm.sum()), "n_pu1_raw": int((mm & pos).sum()),
                         "n_pu1_eligible": int((mm & elig).sum()),
                         "area_pu1_eligible_um2": float(area[mm & elig].sum()),
                         "n_selected": int((mm & sel).sum()), "area_selected_um2": float(area[mm & sel].sum())})
    out = pd.DataFrame(rows)
    out["frac_pu1"] = out["n_pu1_raw"] / out["n_cells_all"].replace(0, np.nan)
    return out


def plate_positions(n: int, rows: str = "ABCDEFGH", cols: int = 12) -> list[str]:
    """A1, A2, … row-major 96-well positions (extend ``rows``/``cols`` for 384)."""
    out = []
    for i in range(n):
        r, c = divmod(i, cols)
        out.append(f"{rows[r % len(rows)]}{c + 1}")
    return out
