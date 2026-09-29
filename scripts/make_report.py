"""Render results/REPORT.md from the cohort tables and run logs (re-runnable).

    python scripts/make_report.py --run-dir outputs/sdata --out results/REPORT.md
"""
from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from lesionseg.report import cohort_table, load_runs, sections_table


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--out", default="results/REPORT.md")
    a = ap.parse_args()
    root = Path(a.run_dir)
    runs = load_runs(root)
    coh = root / "cohort"
    L = runs[0].log
    ct = cohort_table(runs)
    sec = sections_table(runs)
    tot = pd.read_csv(coh / "pu1_counts_total.csv")
    tot["scene"] = tot["scene"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    by_animal = pd.read_csv(coh / "pu1_counts_by_animal.csv")
    pooled = pd.read_csv(coh / "capture_sites_pooled.csv")
    wg = pooled[(pooled.kind == "well group") & (pooled.n_pu1_raw > 0)]
    ma = pooled[(pooled.kind == "manual annotation") & (pooled.n_pu1_raw > 0)]
    trep = Path("models/training_report.csv")
    grid = Path("outputs/lesion_threshold_grid.csv")

    rel = L.get("zone_widths_relative", {})
    out = []
    out.append(f"# lesionSegmenter – report ({date.today().isoformat()})\n")
    out.append("""
Automatic outlining of EAE lesions from Cellpose segmentation masks and the collaborator's Pu.1 calls,
zoning into core / rim / peri-lesion, and selection of Pu.1⁺ cells per capture site for Deep Visual
Proteomics (laser microdissection + mass spectrometry). This report is generated from the run outputs;
the notebooks in `notebooks/` hold the image evidence behind every number.

**Decisions taken in this run** (all configurable in `configs/eae_dvp_sdata.yaml`):
""")
    out.append(f"""
- Input: SpatialData export `cellpose_output.zip` (4 scenes, {int(tot[tot['sample']=='ALL'].n_cells.iloc[0]):,} cells, label = cell_id; Pu.1 positivity from `Pu1_class`).
- Lesion score: **{L['lesion']['score']}** – a gradient-boosting model trained on the collaborator's manual CORE polygons, one leave-one-scene-out model per scene (each scene scored by a model that never saw its own annotations). Threshold {L['lesion']['threshold']}.
- Sections (animal + spinal level) from the curated `Sample_category` polygons; **only sections with manually annotated cores carry lesions** (`sections.focus: {L['section_focus']['mode']}`); all other sections are lesion-free controls.
- Distances relative to section size: zone widths are fractions of each section's equivalent radius (rim {rel.get('rim')}, peri {rel.get('peri')} → median {L['lesion']['rim_width_um'].get('median_um', 0):.0f} / {L['lesion']['peri_width_um'].get('median_um', 0):.0f} µm); rings for wells at 0–10 %, 10–20 %, 20–40 % of the radius; every cell carries `rel_pos` (0 = section centre / canal, 1 = pia).
- Manual annotations are used **only** for model training and validation, never to draw lesions directly.
""")
    out.append("## Headline numbers per scene\n")
    out.append(md_table(ct))
    out.append("\n\n## Pu.1⁺ cells from the segmentation mask (no zoning)\n")
    out.append(md_table(tot.round(3), "{:.3f}"))
    out.append("\n\nPer animal and spinal level (T = thoracic, C = cervical, L = lumbar; CFA = adjuvant-only control, OS = other control):\n")
    out.append(md_table(by_animal.round(3), "{:.3f}"))
    out.append("\n\n## Sections\n")
    cols = ["scene", "section_name", "is_lesion_section", "has_manual_core", "area_mm2", "n_cells", "n_pu1_pos", "frac_pu1_pos",
            "n_lesions", "lesion_area_mm2", "lesion_frac_of_section", "lesion_area_frac_unrestricted"]
    out.append(md_table(sec[[c for c in cols if c in sec]].round(3), "{:.3f}"))
    out.append("""

`lesion_area_frac_unrestricted` is what the detector found before control sections were cleared – the
sections R1_3_C, R1_2_L, P3_1_L (EAE animals without manual cores) still show Pu.1⁺ infiltrates there;
whether those are lesions is a decision for the reader (notebook 02, control-section panels).
""")
    out.append("## Validation against the manual cores\n")
    rows = []
    for r in runs:
        v = r.log["validation"]
        c = v["core_vs_lesion"]
        rows.append({"scene": r.name, "manual cores": v["n_manual_cores"], "cores ≥50% inside auto lesion": v["manual_cores_detected_frac"],
                     "manual core area inside auto lesion": c["manual_covered_by_auto"], "auto lesion mm²": c["auto_area_mm2"],
                     "manual core mm²": c["manual_area_mm2"]})
    out.append(md_table(pd.DataFrame(rows)))
    if trep.exists():
        out.append("\n\nModel, leave-one-scene-out (bin level; avg_precision and cores_detected_frac are the informative columns):\n")
        out.append(md_table(pd.read_csv(trep).round(3), "{:.3f}"))
    if grid.exists():
        g = pd.read_csv(grid)
        best = g.sort_values(["det_frac", "ctrl_area_mm2"], ascending=[False, True]).head(3)
        out.append("\n\nThreshold baseline (Pu.1⁺ density z-score) for comparison – best three of the grid:\n")
        out.append(md_table(best.round(3), "{:.3f}"))
        out.append("\n\nThe threshold detector kept ~94 % of manual cores but produced 3–5× the manual core area, including "
                   "the central-canal region and grey matter; the model halves the excess and removes the canal calls at the "
                   "price of missing some small manual cores (notebook 04 shows each missed core).")
    out.append("\n\n## Capture sites: Pu.1⁺ cells available for DVP\n")
    out.append("Pooled over all sections. `n_pu1_eligible` excludes cells within 100 µm of the section edge and inside VBO; "
               "`n_selected` are the cells placed in a 3000 µm² well.\n")
    cols = ["lesion_section", "compartment", "n_sections", "n_cells_all", "n_pu1_raw", "n_pu1_eligible", "area_pu1_eligible_um2", "n_selected", "area_selected_um2"]
    out.append(md_table(wg[cols].round(0), "{:.0f}"))
    out.append("\n\nBy manual annotation compartment (collaborator's GM / WM / VBO / core polygons):\n")
    out.append(md_table(ma[cols].round(0), "{:.0f}"))
    out.append("\n\nPer section and per animal: `results/cohort/capture_sites_by_section.csv`, `capture_sites_by_animal.csv`; "
               "wells actually formed: `results/<sample>/scene<i>/wells.csv`.\n")
    out.append("## Figures\n")
    for r in runs:
        out.append(f"### {r.name}\n\n![{r.name}]({r.dir.parent.name}/{r.dir.name}/overview.png)\n")
    out.append("""
## Notebooks (executed, in `notebooks/`)

1. `01_cohort_overview` – settings, headline numbers, Pu.1⁺ counts, overview figures, per-section bars
2. `02_judge_lesions` – every lesion section and every lesion at full resolution with automatic and manual outlines; control sections; the score map
3. `03_zones_relative_distance` – section radii, relative zone widths, Pu.1⁺ fraction vs relative distance, radial position of lesions, intensity by zone
4. `04_manual_vs_automatic` – model report, threshold baseline, missed manual cores and automatic lesions without a manual core, cell-level confusion
5. `05_capture_sites_and_wells` – capture-site tables (pooled, per section, per animal), wells, selected-cell maps, overlap with the collaborator's selection, LMD export demo

## Open decisions

- Accept the model-based outlines, or the threshold baseline, or tune the operating point (`lesion.threshold.value`, `.seed`)? Notebook 02/04 are the evidence.
- The EAE sections without manual cores (R1_3_C, R1_2_L, P3_1_L) show infiltrates – keep them lesion-free (current) or annotate them?
- Relative zone widths: rim 5 %, peri 15 % of section radius; rings 0–10 / 10–20 / 20–40 / 40–60 / >60 %. More material per well: raise `wells.target_area_um2`.
- LMD calibration marks are needed before `lesionseg export-lmd` can produce cut files.
- Scans 2023-1 and CML_2 have no Cellpose output yet.
""")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
