"""Pool the per-scene outputs into cohort tables (outputs/<run>/cohort/).

    python scripts/summarize_cohort.py --run-dir outputs/sdata

Writes
  pu1_counts_by_section.csv   Pu.1+ cells per section (segmentation mask + Pu.1 calls, no zoning)
  pu1_counts_by_animal.csv    …pooled per animal (section name minus the level suffix) and per level
  pu1_counts_total.csv        …per scene and grand total
  capture_sites_by_section.csv  Pu.1+ cells per capture site (section × compartment)
  capture_sites_pooled.csv      …pooled over all sections, split lesion vs control sections
  capture_sites_by_animal.csv   …pooled per animal
  sections_all.csv, cohort_table.csv
  + per scene: lesion_cells.png and section_cells_<section>.png (segmented Pu.1+ cells filled by zone)
  reactions_plan.csv, reactions_budget.csv   mass-spec reaction plan (important-info.md), plus per-scene
                                             cells_reactions.csv (cell -> reaction) for LMD export
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lesionseg.config import DEFAULTS, load_config
from lesionseg.reactions import plan_reactions
from lesionseg.report import cohort_table, load_runs, sections_table


def animal_of(section: pd.Series) -> pd.Series:
    """'P2_3_T' -> 'P2_3'; 'OS1_2_' -> 'OS1_2'; level = trailing T/C/L or ''."""
    s = section.astype(str)
    return s.str.replace(r"_[TCL]?$", "", regex=True)


def level_of(section: pd.Series) -> pd.Series:
    return section.astype(str).str.extract(r"_([TCL])$")[0].fillna("")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--config", default=None, help="YAML with a `reactions:` block (defaults otherwise)")
    a = ap.parse_args()
    root = Path(a.run_dir)
    out = root / "cohort"
    out.mkdir(exist_ok=True)
    runs = load_runs(root)

    pc = pd.concat([pd.read_csv(r.dir / "pu1_counts.csv") for r in runs], ignore_index=True)
    pc = pc.rename(columns={pc.columns[2]: "section"})
    pc["animal"] = animal_of(pc["section"])
    pc["level"] = level_of(pc["section"])
    pc.to_csv(out / "pu1_counts_by_section.csv", index=False)
    by_animal = pc.groupby(["animal", "level"]).agg(
        n_sections=("section", "size"), n_cells=("n_cells", "sum"), n_pu1_pos=("n_pu1_pos", "sum"),
        area_pu1_um2=("area_pu1_um2", "sum"))
    by_animal["frac_pu1"] = by_animal["n_pu1_pos"] / by_animal["n_cells"]
    by_animal.reset_index().to_csv(out / "pu1_counts_by_animal.csv", index=False)
    tot = pc.groupby(["sample", "scene"]).agg(n_cells=("n_cells", "sum"), n_pu1_pos=("n_pu1_pos", "sum")).reset_index()
    tot.loc[len(tot)] = ["ALL", "", tot.n_cells.sum(), tot.n_pu1_pos.sum()]
    tot["frac_pu1"] = tot["n_pu1_pos"] / tot["n_cells"]
    tot.to_csv(out / "pu1_counts_total.csv", index=False)

    cap = pd.concat([pd.read_csv(r.dir / "capture_sites.csv") for r in runs if (r.dir / "capture_sites.csv").exists()],
                    ignore_index=True)
    cap["animal"] = animal_of(cap["section"])
    cap["level"] = level_of(cap["section"])
    cap.to_csv(out / "capture_sites_by_section.csv", index=False)
    num = ["n_cells_all", "n_pu1_raw", "n_pu1_eligible", "area_pu1_eligible_um2", "n_selected", "area_selected_um2"]
    pooled = cap.groupby(["lesion_section", "kind", "compartment"])[num].sum().reset_index()
    pooled["n_sections"] = cap.groupby(["lesion_section", "kind", "compartment"])["section"].nunique().to_numpy()
    pooled["frac_pu1"] = pooled["n_pu1_raw"] / pooled["n_cells_all"].replace(0, pd.NA)
    pooled.to_csv(out / "capture_sites_pooled.csv", index=False)
    by_an = cap.groupby(["animal", "lesion_section", "kind", "compartment"])[num].sum().reset_index()
    by_an.to_csv(out / "capture_sites_by_animal.csv", index=False)

    if not a.no_figures:
        from lesionseg.report import save_cell_zone_figures, save_zone_polygon_figures

        for r in runs:
            save_cell_zone_figures(r)
            save_zone_polygon_figures(r)
    sections_table(runs).to_csv(out / "sections_all.csv", index=False)
    cohort_table(runs).to_csv(out / "cohort_table.csv", index=False)

    # ---- mass-spec reaction plan (pools across replicate slides) ---------------------------
    rcfg = (load_config(a.config) if a.config else DEFAULTS).get("reactions", {})
    if not a.config:  # fall back to the config recorded in the run
        rcfg = runs[0].log.get("config", {}).get("reactions", rcfg)
    cells_by_scene = {r.name: r.cells for r in runs}
    plan, cells_rx, budget = plan_reactions(cells_by_scene, rcfg)
    plan.to_csv(out / "reactions_plan.csv", index=False)
    budget.to_csv(out / "reactions_budget.csv", index=False)
    for r in runs:
        c = cells_rx[r.name]
        sel = c[c["reaction_id"] > 0][["cell_id", "label", "x_px", "y_px", "reaction_id", "reaction_name"]]
        sel.to_csv(r.dir / "cells_reactions.csv", index=False)
    # both pooling variants for the report
    alt = dict(rcfg)
    alt["pool_other_gm_wm"] = not bool(budget.set_index("item").loc["other control GM/WM pooled", "value"])
    plan_alt, _, budget_alt = plan_reactions(cells_by_scene, alt)
    plan_alt.to_csv(out / "reactions_plan_alternative.csv", index=False)
    budget_alt.to_csv(out / "reactions_budget_alternative.csv", index=False)
    print("\nreaction plan:")
    print(budget.to_string(index=False))
    print(tot.to_string(index=False))
    cols = ["lesion_section", "compartment", "n_sections", "n_pu1_raw", "n_pu1_eligible", "n_selected",
            "area_selected_um2"]
    print(pooled[pooled.compartment != "all"][cols].to_string(index=False))


if __name__ == "__main__":
    main()
