"""Mass-spec **reaction plan** for DVP: pool Pu.1⁺ cells into ≤ ``max_reactions`` LMD collections.

Rules (important-info.md, agreed with Ting):

* A *section* is the ``Sample_category`` name (animal + spinal level, e.g. ``P2_3_T``). Every
  section is present on two replicate slides; replicates of the same section × compartment are
  **pooled into one reaction**. Sections are never pooled with each other, except where stated.
* Lesion sections → one reaction per section per lesion compartment: ``core``, ``rim``, ``peri`` and
  ``deep`` (the "additional step": remaining tissue outward of the peri band, i.e. zone ``distal``
  inside a lesion section).
* Non-lesion sections → ``GM`` and ``WM`` (manual annotation) per section. Sections whose name starts
  with ``CFA`` are pooled together (one GM + one WM reaction). Other control sections keep their own
  GM / WM reactions unless the budget is exceeded (``pool_other_gm_wm: auto``); then the smallest pooling
  that fits is applied: first sections sharing a control prefix (``control_prefixes``, default OS) are
  pooled, and only if still over budget all remaining control sections are pooled.
* ``VBO`` (vascular barrier niche, manual polygons) → one reaction per section that has VBO cells,
  pooled across replicate slides only. VBO reactions take **every cell** inside the polygons (Pu.1⁺ and
  Pu.1⁻; ``vbo_all_cells: true``) and ignore the edge exclusion.
* Lesion / GM / WM compartments count Pu.1⁺ cells only and respect the edge exclusion used for wells
  (``dist_to_section_edge_um`` ≥ ``edge_exclusion_um``); VBO cells are excluded from the other
  compartments.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LESION_COMPARTMENTS = ("core", "rim", "peri", "deep")
IGNORED_SECTIONS = {"unassigned", "None", "nan", "0", ""}


def _compartment(cells: pd.DataFrame, edge_exclusion_um: float) -> pd.Series:
    """Compartment label per cell (object dtype; None if the cell belongs to no compartment)."""
    zone = cells["zone"].astype(str)
    has_les = cells["section_has_lesion"].to_numpy(bool) if "section_has_lesion" in cells else np.ones(len(cells), bool)
    manual = (cells["manual_annotation"].astype(str) if "manual_annotation" in cells
              else pd.Series("nan", index=cells.index))
    in_vbo = cells["in_vbo"].to_numpy(bool) if "in_vbo" in cells else np.zeros(len(cells), bool)
    in_vbo = in_vbo | (manual == "vbo").to_numpy()
    edge_ok = (cells["dist_to_section_edge_um"].to_numpy(float) >= edge_exclusion_um
               if "dist_to_section_edge_um" in cells and edge_exclusion_um > 0 else np.ones(len(cells), bool))
    comp = np.full(len(cells), None, dtype=object)
    # lesion compartments (lesion sections only)
    for z, name in (("core", "core"), ("rim", "rim"), ("peri", "peri"), ("distal", "deep")):
        comp[has_les & (zone == z).to_numpy() & edge_ok] = name
    # GM / WM (non-lesion sections only)
    for m in ("GM", "WM"):
        comp[~has_les & (manual == m).to_numpy() & edge_ok] = m
    # VBO overrides everything and ignores the edge exclusion
    comp[in_vbo] = "VBO"
    return pd.Series(comp, index=cells.index)


def _select(df: pd.DataFrame, n: int, order: str, rng: np.random.Generator) -> pd.Index:
    if order == "random":
        return df.index[rng.permutation(len(df))[:n]]
    if order == "central" and "dist_to_lesion_um" in df:
        return df.sort_values("dist_to_lesion_um").index[:n]
    return df.sort_values(["scene", "x_um", "y_um"]).index[:n]


def plan_reactions(cells_by_scene: dict[str, pd.DataFrame], cfg: dict | None = None
                   ) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Build the reaction plan.

    Returns ``(plan, cells_by_scene, budget)``; each cell table gets ``reaction_id`` (0 = not
    selected) and ``reaction_name`` columns. ``budget`` lists reactions per pool type, the total
    against ``max_reactions`` and which pooling was applied.
    """
    cfg = dict(cfg or {})
    target = int(cfg.get("target_cells", 250))
    max_rx = int(cfg.get("max_reactions", 60))
    pool_cfa = bool(cfg.get("pool_cfa", True))
    pool_other = cfg.get("pool_other_gm_wm", "auto")
    control_prefixes = list(cfg.get("control_prefixes", ["OS"]))
    lesion_comps = tuple(cfg.get("lesion_compartments", LESION_COMPARTMENTS))
    vbo_per_section = bool(cfg.get("vbo_per_section", True))
    edge = float(cfg.get("edge_exclusion_um", 100.0))
    order = cfg.get("order", "spatial")
    shortfall_frac = float(cfg.get("shortfall_frac", 0.8))
    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    vbo_all = bool(cfg.get("vbo_all_cells", True))

    # ---- one long table of candidate cells with compartment ----------------------------------
    # Pu.1+ cells everywhere; inside VBO polygons every cell (the barrier niche is captured whole)
    frames = []
    for scene, c in cells_by_scene.items():
        d = c.copy()
        d["scene"] = scene
        d["section"] = d["section_name"].astype(str) if "section_name" in d else d["section_id"].astype(str)
        d["compartment"] = _compartment(d, edge)
        take = d["pu1_pos"].to_numpy(bool)
        if vbo_all:
            take = take | (d["compartment"] == "VBO").to_numpy()
        d = d[take]
        d["orig_idx"] = d.index
        keep = ["scene", "section", "compartment", "orig_idx", "x_um", "y_um", "area_um2", "pu1_pos"]
        if "dist_to_lesion_um" in d:
            keep.append("dist_to_lesion_um")
        frames.append(d[keep])
    pos = pd.concat(frames, ignore_index=True)  # unique index; (scene, orig_idx) maps back to the source tables
    pos = pos[pos["compartment"].notna() & ~pos["section"].isin(IGNORED_SECTIONS)]

    # ---- pools ------------------------------------------------------------------------------
    def build(pool_level: int) -> list[dict]:
        """pool_level 0: every control section separate; 1: pool sections sharing a control prefix
        (``control_prefixes``, e.g. OS); 2: pool all remaining control sections."""
        pools: list[dict] = []
        secs = sorted(pos["section"].unique())
        lesion_secs = sorted(pos.loc[pos.compartment.isin(lesion_comps), "section"].unique())
        for sec in lesion_secs:
            for comp in lesion_comps:
                m = (pos.section == sec) & (pos.compartment == comp)
                pools.append({"pool_type": "lesion", "compartment": comp, "sections": [sec], "mask": m,
                              "name": f"{sec}|{comp}", "priority": 1})
        ctrl_secs = [s for s in secs if s not in lesion_secs]
        cfa = [s for s in ctrl_secs if s.upper().startswith("CFA")]
        other = [s for s in ctrl_secs if s not in cfa]
        for comp in ("GM", "WM"):
            if cfa and pool_cfa:
                m = pos.section.isin(cfa) & (pos.compartment == comp)
                pools.append({"pool_type": comp, "compartment": comp, "sections": cfa, "mask": m,
                              "name": f"CFA(pooled)|{comp}", "priority": 2})
            else:
                for s in cfa:
                    m = (pos.section == s) & (pos.compartment == comp)
                    pools.append({"pool_type": comp, "compartment": comp, "sections": [s], "mask": m,
                                  "name": f"{s}|{comp}", "priority": 2})
            if other and pool_level >= 2:
                m = pos.section.isin(other) & (pos.compartment == comp)
                pools.append({"pool_type": comp, "compartment": comp, "sections": other, "mask": m,
                              "name": f"controls(pooled)|{comp}", "priority": 3})
            elif other and pool_level == 1:
                rest = list(other)
                for pref in control_prefixes:
                    grp = [s for s in rest if s.upper().startswith(pref.upper())]
                    if len(grp) > 1:
                        m = pos.section.isin(grp) & (pos.compartment == comp)
                        pools.append({"pool_type": comp, "compartment": comp, "sections": grp, "mask": m,
                                      "name": f"{pref}(pooled)|{comp}", "priority": 3})
                        rest = [s for s in rest if s not in grp]
                for s in rest:
                    m = (pos.section == s) & (pos.compartment == comp)
                    pools.append({"pool_type": comp, "compartment": comp, "sections": [s], "mask": m,
                                  "name": f"{s}|{comp}", "priority": 3})
            else:
                for s in other:
                    m = (pos.section == s) & (pos.compartment == comp)
                    pools.append({"pool_type": comp, "compartment": comp, "sections": [s], "mask": m,
                                  "name": f"{s}|{comp}", "priority": 3})
        vbo_secs = sorted(pos.loc[pos.compartment == "VBO", "section"].unique())
        if vbo_per_section:
            for s in vbo_secs:
                m = (pos.section == s) & (pos.compartment == "VBO")
                pools.append({"pool_type": "VBO", "compartment": "VBO", "sections": [s], "mask": m,
                              "name": f"{s}|VBO", "priority": 4})
        elif vbo_secs:
            m = pos.compartment == "VBO"
            pools.append({"pool_type": "VBO", "compartment": "VBO", "sections": vbo_secs, "mask": m,
                          "name": "VBO(pooled)", "priority": 4})
        return [p for p in pools if int(p["mask"].sum()) > 0]

    variants = {lvl: build(lvl) for lvl in (0, 1, 2)}
    n_by_level = {lvl: len(v) for lvl, v in variants.items()}
    n_unpooled, n_pooled = n_by_level[0], n_by_level[2]
    if pool_other == "auto":  # smallest amount of pooling that fits the budget
        use_level = next((lvl for lvl in (0, 1, 2) if n_by_level[lvl] <= max_rx), 2)
    else:
        use_level = 2 if pool_other is True else (1 if str(pool_other).lower() in ("prefix", "1") else 0)
    use_pooled = use_level > 0
    pools = variants[use_level]

    # ---- selection --------------------------------------------------------------------------
    plan_rows = []
    assign = {scene: (np.zeros(len(c), int), np.full(len(c), None, dtype=object))
              for scene, c in cells_by_scene.items()}
    for rid, p in enumerate(pools, start=1):
        avail = pos[p["mask"]]
        chosen = _select(avail, target, order, rng)
        sel = avail.loc[chosen]
        for scene, grp in sel.groupby("scene"):
            ids, names = assign[scene]
            loc = cells_by_scene[scene].index.get_indexer(grp["orig_idx"])
            ids[loc] = rid
            names[loc] = p["name"]
        plan_rows.append({
            "reaction_id": rid, "reaction_name": p["name"], "pool_type": p["pool_type"],
            "compartment": p["compartment"],
            "sections": ";".join(p["sections"]), "scenes": ";".join(sorted(avail["scene"].unique())),
            "n_sections": len(p["sections"]), "n_available": int(len(avail)), "n_selected": int(len(sel)),
            "n_pu1_available": int(avail["pu1_pos"].sum()), "n_pu1_selected": int(sel["pu1_pos"].sum()),
            "area_selected_um2": float(sel["area_um2"].sum()),
            "shortfall": bool(len(avail) < shortfall_frac * target), "priority": p["priority"],
        })
    plan = pd.DataFrame(plan_rows)
    out_cells = {}
    for scene, c in cells_by_scene.items():
        c = c.copy()
        ids, names = assign[scene]
        c["reaction_id"] = ids
        c["reaction_name"] = names
        out_cells[scene] = c

    # ---- budget -----------------------------------------------------------------------------
    per_type = plan.groupby("pool_type").size() if len(plan) else pd.Series(dtype=int)
    budget = pd.DataFrame([
        *[{"item": f"reactions: {k}", "value": int(v)} for k, v in per_type.items()],
        {"item": "reactions: total", "value": int(len(plan))},
        {"item": "max_reactions", "value": max_rx},
        {"item": "within budget", "value": bool(len(plan) <= max_rx)},
        {"item": "target cells per reaction", "value": target},
        {"item": "reactions with shortfall", "value": int(plan["shortfall"].sum()) if len(plan) else 0},
        {"item": "CFA GM/WM pooled", "value": pool_cfa},
        {"item": "other control GM/WM pooled", "value": use_pooled},
        {"item": "control pooling level (0 none, 1 by prefix, 2 all)", "value": use_level},
        {"item": "reactions if other controls unpooled", "value": n_unpooled},
        {"item": "reactions if controls pooled by prefix", "value": n_by_level[1]},
        {"item": "reactions if other controls pooled", "value": n_pooled},
        {"item": "VBO sections", "value": int((plan.pool_type == "VBO").sum()) if len(plan) else 0},
    ])
    return plan, out_cells, budget
