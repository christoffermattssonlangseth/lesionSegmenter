import numpy as np
import pandas as pd

from lesionseg.reactions import plan_reactions


def _cells(scene, section, n, zone="distal", has_lesion=False, manual=None, vbo=False, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "cell_id": np.arange(n), "pu1_pos": True, "zone": zone, "section_name": section,
        "section_has_lesion": has_lesion, "manual_annotation": manual, "in_vbo": vbo,
        "dist_to_section_edge_um": 500.0, "x_um": rng.random(n) * 1000, "y_um": rng.random(n) * 1000,
        "area_um2": 30.0, "dist_to_lesion_um": rng.random(n) * 100,
    })


def _scene(scene, rep_seed):
    parts = [
        _cells(scene, "P1_T", 200, "core", True, seed=rep_seed),
        _cells(scene, "P1_T", 150, "rim", True, seed=rep_seed + 1),
        _cells(scene, "P1_T", 90, "peri", True, seed=rep_seed + 2),
        _cells(scene, "P1_T", 400, "deep", True, seed=rep_seed + 3),
        _cells(scene, "P1_T", 20, "distal", True, vbo=True, seed=rep_seed + 4),
        _cells(scene, "CFA_1_C", 80, "distal", False, "GM"), _cells(scene, "CFA_1_C", 30, "distal", False, "WM"),
        _cells(scene, "CFA_1_L", 80, "distal", False, "GM"), _cells(scene, "CFA_1_L", 30, "distal", False, "WM"),
        _cells(scene, "OS_1_C", 120, "distal", False, "GM"), _cells(scene, "OS_1_C", 60, "distal", False, "WM"),
        _cells(scene, "OS_1_C", 5, "distal", False, "vbo"),
    ]
    df = pd.concat(parts, ignore_index=True)
    df["cell_id"] = np.arange(len(df))
    return df


def test_plan_pools_replicates_and_cfa():
    cells = {"A": _scene("A", 0), "B": _scene("B", 100)}
    plan, out, budget = plan_reactions(cells, {"target_cells": 250, "max_reactions": 60})
    # lesion section: 4 compartments, each pooled over both scenes
    les = plan[plan.pool_type == "lesion"]
    assert set(les.compartment) == {"core", "rim", "peri", "deep"}
    assert (les.scenes == "A;B").all()
    core = les[les.compartment == "core"].iloc[0]
    assert core.n_available == 400 and core.n_selected == 400 and not core.shortfall  # 250 is a target, not a cap
    peri = les[les.compartment == "peri"].iloc[0]
    assert peri.n_available == 180 and peri.shortfall  # < 0.8 * 250
    # CFA pooled into one GM + one WM; OS keeps its own (budget not exceeded)
    assert set(plan[plan.reaction_name.str.startswith("CFA(pooled)")].compartment) == {"GM", "WM"}
    assert {"OS_1_C|GM", "OS_1_C|WM"} <= set(plan.reaction_name)
    # VBO per section, incl. the manual 'vbo' label in a control section; ignores edge exclusion
    vbo = plan[plan.pool_type == "VBO"]
    assert (vbo.n_available >= vbo.n_pu1_available).all()  # VBO takes every cell, not only Pu.1+
    assert set(vbo.sections) == {"P1_T", "OS_1_C"}
    assert vbo.set_index("sections").loc["P1_T", "n_available"] == 40
    # cells carry reaction ids; VBO cells never end up in 'deep'
    a = out["A"]
    n_sel = sum((o.reaction_id > 0).sum() for o in out.values())
    assert n_sel == plan.n_selected.sum()
    assert set(a.loc[a.in_vbo & (a.reaction_id > 0), "reaction_name"]) == {"P1_T|VBO"}
    assert budget.set_index("item").loc["reactions: total", "value"] == len(plan)
    assert budget.set_index("item").loc["within budget", "value"]


def test_max_cells_caps_and_stratifies():
    cells = {"A": _scene("A", 0), "B": _scene("B", 100)}
    plan, out, _ = plan_reactions(cells, {"target_cells": 250, "max_cells": 250, "max_reactions": 60})
    core = plan[plan.reaction_name == "P1_T|core"].iloc[0]
    assert core.n_available == 400 and core.n_selected == 250
    per_scene = [int((o.reaction_name == "P1_T|core").sum()) for o in out.values()]
    assert sum(per_scene) == 250 and min(per_scene) > 0  # drawn from both replicate slides


def test_budget_auto_pooling():
    # 20 OS control sections + 5 other controls: unpooled = 50 reactions.
    parts = [_cells("A", f"OS_{i}_C", 50, "distal", False, m) for i in range(20) for m in ("GM", "WM")]
    parts += [_cells("A", f"X_{i}_C", 50, "distal", False, m) for i in range(5) for m in ("GM", "WM")]
    cells = {"A": pd.concat(parts, ignore_index=True)}
    # budget 20 -> level 1: OS pooled (2) + 5 others x 2 = 12 reactions
    plan, _, budget = plan_reactions(cells, {"max_reactions": 20, "pool_other_gm_wm": "auto"})
    assert {"OS(pooled)|GM", "OS(pooled)|WM"} <= set(plan.reaction_name) and len(plan) == 12
    b = budget.set_index("item")["value"]
    assert b["control pooling level (0 none, 1 by prefix, 2 all)"] == 1
    assert b["reactions if other controls unpooled"] == 50 and b["reactions if other controls pooled"] == 2
    # budget 5 -> level 2: everything pooled
    plan1, _, _ = plan_reactions(cells, {"max_reactions": 5, "pool_other_gm_wm": "auto"})
    assert set(plan1.reaction_name) == {"controls(pooled)|GM", "controls(pooled)|WM"}
    # generous budget -> no pooling
    plan2, _, _ = plan_reactions(cells, {"max_reactions": 100, "pool_other_gm_wm": "auto"})
    assert len(plan2) == 50
    plan3, _, _ = plan_reactions(cells, {"max_reactions": 100, "pool_other_gm_wm": True})
    assert len(plan3) == 2
