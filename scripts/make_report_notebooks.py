"""Generate (and optionally execute) the report notebooks in notebooks/.

    python scripts/make_report_notebooks.py            # write + execute
    python scripts/make_report_notebooks.py --no-exec  # write only

The notebooks are *evidence* for a human decision: images with the automatic outlines next to
the manual annotations, and the tables behind every number. They read outputs/sdata (or --run-dir).
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
KERNEL = {"kernelspec": {"name": "lesionseg", "display_name": "Python (lesionseg)", "language": "python"}}


def md(t):
    return new_markdown_cell(t.strip("\n"))


def code(t):
    return new_code_cell(t.strip("\n"))


SETUP = '''
import os, json, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
os.chdir(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import Image, display, Markdown
%config InlineBackend.figure_format = 'jpeg'   # image-heavy notebooks: keep files small
plt.rcParams["figure.dpi"] = 80
from lesionseg import report as R
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40); pd.set_option("display.max_rows", 120)
RUN_DIR = "{run_dir}"
runs = R.load_runs(RUN_DIR)
print("scenes:", [r.name for r in runs])
'''


def nb_overview(run_dir):
    cells = [md("""
# 01 · Cohort overview

What was run, on which scenes, and the headline numbers. Nothing here is a verdict – the purpose of
these notebooks is to lay out the evidence so **you** decide whether the lesion outlines are right.

**Colour key used throughout**: <span style="color:#eda100">■</span> automatic lesion outline ·
<span style="color:#e34948">■</span> automatic core · <span style="color:#e87ba4">■</span> manual CORE (collaborator, dashed) ·
zones <span style="color:#3987e5">■ distal (control)</span> <span style="color:#9085e9">■ deep</span> <span style="color:#199e70">■ peri</span> <span style="color:#c98500">■ rim</span> <span style="color:#e66767">■ core</span>.
"""), code(SETUP.format(run_dir=run_dir)), md("## Settings of this run"), code('''
L = runs[0].log
settings = {
    "lesion score": L["lesion"]["score"],
    "model": L.get("lesion_model", {}).get("path", "–"),
    "threshold": L["lesion"]["threshold"],
    "rim width": L["lesion"]["rim_width_um"], "peri width": L["lesion"]["peri_width_um"],
    "relative widths (fraction of section radius)": L.get("zone_widths_relative"),
    "section focus": L["section_focus"]["mode"],
    "Pu.1 calls": L["pu1"]["method"],
    "density grid / smoothing (µm)": (L["config"]["density"]["bin_um"], L["config"]["density"]["sigma_um"]),
}
pd.DataFrame({"setting": list(settings), "value": [str(v) for v in settings.values()]})
'''), md("## Headline numbers per scene"), code('''
R.cohort_table(runs)
'''), md("""
## Pu.1⁺ cells straight from the segmentation mask

Counts of segmented cells and Pu.1⁺ calls (the collaborator's `Pu1_class`) per scene, per section
(animal + spinal level) and per animal. No zoning involved.
"""), code('''
coh = Path(RUN_DIR) / "cohort"
display(pd.read_csv(coh / "pu1_counts_total.csv").round(3))
pc = pd.read_csv(coh / "pu1_counts_by_section.csv")
display(pc.pivot_table(index=["animal", "level", "section"], columns="sample", values="n_pu1_pos", aggfunc="sum", fill_value=0).astype(int))
display(pd.read_csv(coh / "pu1_counts_by_animal.csv").round(3))
'''), md("## Per-scene overview figures"), code('''
for r in runs:
    display(Markdown(f"### {r.name}"))
    display(Image(filename=str(r.dir / "overview.png"), width=1400))
'''), md("""
## Sections: lesion vs control, Pu.1⁺ fraction, lesion burden

`is_lesion_section` is the data-driven / manual-guided classification used in this run; `has_manual_core`
says whether the collaborator drew a core there. `lesion_area_frac_unrestricted` is what the detector
found *before* the control sections were cleared.
"""), code('''
sec = R.sections_table(runs)
cols = ["scene", "section_name", "is_lesion_section", "has_manual_core", "area_mm2", "n_cells", "n_pu1_pos", "frac_pu1_pos",
        "n_lesions", "lesion_area_mm2", "lesion_frac_of_section", "lesion_area_frac_unrestricted", "n_auto_lesions_unrestricted"]
sec[[c for c in cols if c in sec]].round(3)
'''), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 0.32 * len(sec) + 1.5))
R.bar_by_section(sec, "frac_pu1_pos", ax=axes[0], title="Pu.1⁺ fraction of all cells, per section")
R.bar_by_section(sec, "lesion_area_frac_unrestricted", ax=axes[1], title="lesion area fraction before control sections were cleared")
plt.tight_layout()
''')]
    return cells


def nb_judge(run_dir):
    return [md("""
# 02 · Judge the lesion outlines

Every lesion section at low magnification, then every lesion at full resolution, then the control
sections. Outlines: <span style="color:#eda100">automatic lesion</span>, <span style="color:#e34948">automatic core</span>,
<span style="color:#e87ba4">manual CORE (dashed)</span>; white rings = Pu.1⁺ cells. Composite: nuclei blue-grey, Pu.1 orange, Iba1 green.

Things to look at
- Does the automatic outline follow the dense Pu.1⁺ / Iba1-bright infiltrate, or does it include ordinary grey matter?
- Is the rim (between automatic lesion and core) a sensible transition zone?
- Are there Pu.1⁺ clusters *without* an outline that should have one (subpial / meningeal infiltrates)?
- Do the control sections stay empty?
- Is the central canal region (section centre) ever outlined?
"""), code(SETUP.format(run_dir=run_dir)), md("## Lesion sections"), code('''
for r in runs:
    les_secs = r.sections[r.sections.is_lesion_section]
    n = len(les_secs)
    if n == 0:
        continue
    fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(16, 8 * int(np.ceil(n / 2))), squeeze=False)
    for ax, sid in zip(axes.ravel(), les_secs.section_id):
        R.plot_section(r, int(sid), ax=ax, scale=0.2)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    R.outline_legend(axes.ravel()[0])
    fig.suptitle(r.name, y=1.0)
    plt.tight_layout(); plt.show()
'''), md("## Every lesion at full resolution (largest first)"), code('''
for r in runs:
    fig = R.lesion_gallery(r, max_n=12, ncols=3)
    plt.show()
'''), md("## Control sections (should stay empty)"), code('''
for r in runs:
    ctrl = r.sections[~r.sections.is_lesion_section]
    n = len(ctrl)
    if n == 0:
        continue
    fig, axes = plt.subplots(int(np.ceil(n / 3)), 3, figsize=(18, 6 * int(np.ceil(n / 3))), squeeze=False)
    for ax, sid in zip(axes.ravel(), ctrl.section_id):
        R.plot_section(r, int(sid), ax=ax, scale=0.12, show_cells=False)
        c = r.cells[(r.cells.section_id == sid) & r.cells.pu1_pos]
        ax.scatter(c.x_um, c.y_um, s=1, c="#ffffff", alpha=0.6, linewidths=0)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{r.name} – control sections (white = Pu.1⁺ cells)", y=1.0)
    plt.tight_layout(); plt.show()
'''), md("## The score behind the calls"), code('''
for r in runs:
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(r.map("lesion_score"), extent=r.extent_um, cmap="magma", vmin=0, vmax=1 if r.log["lesion"]["score"] == "model" else None)
    R.draw_outlines(ax, r, lesion=True, core=False, manual=True)
    plt.colorbar(im, ax=ax, fraction=0.03, label="lesion score" + (" (model probability)" if r.log["lesion"]["score"] == "model" else ""))
    ax.set_title(f"{r.name}: score, automatic lesion (yellow) and manual CORE (pink dashed)")
    plt.show()
''')]


def nb_zones(run_dir):
    return [md("""
# 03 · Zones and relative distances

Sections differ in size (cervical vs thoracic vs lumbar), so all zone widths and rings are fractions
of each section's equivalent radius √(area/π), and each cell carries a **relative radial position**
`rel_pos` (0 = section centre ≈ central canal, 1 = pial edge) and `dist_to_lesion_rel`
(signed distance to the lesion edge / section radius). Absolute µm values are kept alongside.
"""), code(SETUP.format(run_dir=run_dir)), md("## Section sizes and the zone widths that resulted"), code('''
rows = []
for r in runs:
    g = r.cells.groupby("section_name", observed=True)
    t = g["section_radius_um"].first().rename("radius_um").to_frame()
    rel = r.log.get("zone_widths_relative", {})
    t["rim_width_um"] = t.radius_um * rel.get("rim", np.nan)
    t["peri_width_um"] = t.radius_um * rel.get("peri", np.nan)
    t.insert(0, "scene", r.name)
    rows.append(t.reset_index())
pd.concat(rows).round(0)
'''), md("## Pu.1⁺ fraction versus relative distance to the lesion edge (per lesion section)"), code('''
edges = np.arange(-0.3, 0.62, 0.04)
for r in runs:
    les = r.sections[r.sections.is_lesion_section].section_name.tolist()
    if not les:
        continue
    fig, axes = plt.subplots(1, len(les), figsize=(4.2 * len(les), 3.6), sharey=True, squeeze=False)
    for ax, sname in zip(axes.ravel(), les):
        c = r.cells[(r.cells.section_name == sname) & np.isfinite(r.cells.dist_to_lesion_rel)]
        b = pd.cut(c.dist_to_lesion_rel, edges)
        g = c.groupby(b, observed=False)
        frac = g["pu1_pos"].mean(); n = g.size()
        mids = [iv.mid for iv in frac.index]
        ax.plot(mids, frac.values, color="#2a78d6", lw=2)
        ax.fill_between(mids, 0, frac.values, color="#2a78d6", alpha=0.15)
        ax.axvline(0, color="#898781", ls="--", lw=1)
        ax.set_title(f"{sname} (n={len(c)})", fontsize=10)
        ax.set_xlabel("distance to lesion edge / section radius")
    axes[0, 0].set_ylabel("Pu.1⁺ fraction")
    fig.suptitle(f"{r.name}: <0 inside the lesion", y=1.02)
    plt.tight_layout(); plt.show()
'''), md("## Where do the lesions sit? Radial position of Pu.1⁺ cells by zone"), code('''
fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 3.8), sharey=True, squeeze=False)
for ax, r in zip(axes.ravel(), runs):
    c = r.cells[r.cells.pu1_pos & r.cells.section_has_lesion]
    for z in R.ZONE_ORDER:
        s = c[c.zone == z].rel_pos.dropna()
        if len(s):
            ax.hist(s, bins=np.linspace(0, 1, 26), histtype="step", lw=2, color=R.ZONE_COLORS[z], label=f"{z} (n={len(s)})", density=True)
    ax.set_title(r.name); ax.set_xlabel("relative radial position (0 centre → 1 pia)"); ax.legend(fontsize=8)
axes[0, 0].set_ylabel("density")
plt.tight_layout()
'''), md("## Zone composition of Pu.1⁺ cells per section"), code('''
sec = R.sections_table(runs)
d = sec[sec.is_lesion_section].copy()
d["label"] = d.scene.str.replace(" scene ", "/s") + " · " + d.section_name.astype(str)
cols = {"core": "n_pu1_pos_core", "rim": "n_pu1_pos_rim", "peri": "n_pu1_pos_peri"}
fig, ax = plt.subplots(figsize=(10, 0.4 * len(d) + 1))
left = np.zeros(len(d))
for z, col in cols.items():
    ax.barh(d.label, d[col], left=left, color=R.ZONE_COLORS[z], label=z, height=0.7, edgecolor="white", linewidth=1)
    left += d[col].to_numpy()
ax.set_xlabel("Pu.1⁺ cells"); ax.legend(loc="lower right"); ax.set_title("Pu.1⁺ cells per zone, lesion sections", loc="left")
plt.tight_layout()
'''), md("## Pu.1 and Iba1 intensity by zone (table channel means, per cell)"), code('''
fig, axes = plt.subplots(2, len(runs), figsize=(4.5 * len(runs), 7), squeeze=False)
for j, r in enumerate(runs):
    c = r.cells[r.cells.pu1_pos & r.cells.section_has_lesion]
    for i, ch in enumerate(["Pu1_mean", "Iba1_mean"]):
        if ch not in c:
            continue
        data = [c.loc[c.zone == z, ch].dropna() for z in R.ZONE_ORDER]
        bp = axes[i, j].boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
        axes[i, j].set_xticks(range(1, len(R.ZONE_ORDER) + 1), R.ZONE_ORDER)
        for patch, z in zip(bp["boxes"], R.ZONE_ORDER):
            patch.set_facecolor(R.ZONE_COLORS[z]); patch.set_alpha(0.8)
        axes[i, j].set_title(f"{r.name}: {ch.replace('_mean', '')} intensity of Pu.1⁺ cells", fontsize=10)
plt.tight_layout()
''')]


def nb_manual(run_dir):
    return [md("""
# 04 · Automatic vs manual annotations, and the model

The collaborator drew lesion **cores** (not rims), grey/white matter and vessel (VBO) regions. They
were used for two things only: training the lesion model (leave-one-scene-out) and validation.
This notebook shows where the two agree and, more importantly, where they do not – so you can decide
which one is right in each case.
"""), code(SETUP.format(run_dir=run_dir)), md("## Model: leave-one-scene-out training report"), code('''
rep = Path("models/training_report.csv")
display(pd.read_csv(rep).round(3) if rep.exists() else "no model report")
import joblib
m = joblib.load(runs[0].log["lesion_model"]["path"]) if "lesion_model" in runs[0].log else None
print("features:", m["features"] if m else "–")
'''), md("""
Bin-level AUC is near 1 because most tissue is trivially negative; `avg_precision` and
`cores_detected_frac` (manual cores ≥50 % inside the automatic lesion) are the informative columns.
"""), md("## Threshold baseline for comparison"), code('''
g = Path("outputs/lesion_threshold_grid.csv")
if g.exists():
    grid = pd.read_csv(g)
    display(grid.sort_values(["det_frac", "ctrl_area_mm2"], ascending=[False, True]).head(12).round(3))
    print("det_frac = manual cores detected; ctrl_area_mm2 = lesion area in CFA/OS control sections; excess_frac = automatic area in annotated sections outside manual core + 100 µm")
'''), md("## Agreement summary per scene"), code('''
rows = []
for r in runs:
    v = r.log["validation"]; c = v["core_vs_lesion"]; k = v["core_vs_core"]
    rows.append({"scene": r.name, "manual cores": v["n_manual_cores"], "cores ≥50% inside auto lesion": v["manual_cores_detected_frac"],
                 "manual area inside auto lesion": c["manual_covered_by_auto"], "auto lesion mm²": c["auto_area_mm2"], "manual core mm²": c["manual_area_mm2"],
                 "IoU lesion vs core": c["iou"], "IoU auto core vs manual core": k["iou"]})
pd.DataFrame(rows).round(3)
'''), md("## Per manual core: how much of it lies inside the automatic lesion"), code('''
fig, axes = plt.subplots(1, len(runs), figsize=(4.5 * len(runs), 3.5), sharey=True, squeeze=False)
for ax, r in zip(axes.ravel(), runs):
    v = pd.read_csv(r.dir / "validation_manual_cores.csv")
    ax.hist(v.frac_in_auto_lesion, bins=np.linspace(0, 1, 11), color="#2a78d6", edgecolor="white")
    ax.set_title(f"{r.name} (n={len(v)})"); ax.set_xlabel("fraction of manual core inside automatic lesion")
axes[0, 0].set_ylabel("manual cores")
plt.tight_layout()
'''), md("## Manual cores the automatic mask missed (< 50 % covered)"), code('''
for r in runs:
    miss = R.missed_manual_cores(r)
    display(Markdown(f"### {r.name}: {len(miss)} missed manual core(s)"))
    if len(miss):
        display(miss.round(2))
        fig = R.manual_core_gallery(r, miss.sort_values("area_um2", ascending=False), max_n=9)
        plt.show()
'''), md("## Automatic lesions with no manual core inside them"), code('''
for r in runs:
    v = pd.read_csv(r.dir / "validation_manual_cores.csv")
    touched = set()
    for ids in v.auto_lesion_ids.astype(str):
        touched |= {int(x) for x in ids.strip("[]").split(",") if x.strip()}
    extra = r.lesions[~r.lesions.lesion_id.isin(touched)]
    display(Markdown(f"### {r.name}: {len(extra)} automatic lesion(s) without a manual core ({extra.area_um2.sum()/1e6:.2f} mm²)"))
    if len(extra):
        fig = R.lesion_gallery(r, lesion_ids=extra.lesion_id.tolist(), max_n=9)
        plt.show()
'''), md("## Cell-level confusion: manual annotation × automatic zone"), code('''
for r in runs:
    display(Markdown(f"### {r.name}"))
    display(pd.read_csv(r.dir / "validation_cell_confusion.csv", index_col=0))
''')]


def nb_wells(run_dir):
    return [md("""
# 05 · Capture sites and LMD wells

Deep Visual Proteomics captures **Pu.1⁺ cells** per capture site (section × compartment) by laser
microdissection and measures them by mass spectrometry. This notebook summarises how many Pu.1⁺ cells
each capture site offers, how many were selected for a well, and where they are.

Compartments: in lesion sections `core`, `rim`, and rings outside the lesion edge at 0–10 %, 10–20 %,
20–40 %, 40–60 % and >60 % of the section radius (the last two reach deep into the tissue); in control
sections the manual `GM` / `WM` regions. Manual
annotation compartments (GM, WM, vbo, manual core, unannotated) are listed for every section as well.
"""), code(SETUP.format(run_dir=run_dir)), md("## Pooled over all sections"), code('''
coh = Path(RUN_DIR) / "cohort"
pooled = pd.read_csv(coh / "capture_sites_pooled.csv")
cols = ["lesion_section", "kind", "compartment", "n_sections", "n_cells_all", "n_pu1_raw", "n_pu1_eligible", "area_pu1_eligible_um2", "n_selected", "area_selected_um2", "frac_pu1"]
pooled[pooled.compartment != "all"][cols].round(3)
'''), md("## Per section (Pu.1⁺ cells eligible for capture)"), code('''
cap = pd.read_csv(coh / "capture_sites_by_section.csv")
wg = cap[cap.kind == "well group"]
display(wg.pivot_table(index=["sample", "scene", "section", "lesion_section"], columns="compartment", values="n_pu1_eligible", aggfunc="sum", fill_value=0).astype(int))
ma = cap[cap.kind == "manual annotation"]
display(ma.pivot_table(index=["sample", "scene", "section", "lesion_section"], columns="compartment", values="n_pu1_raw", aggfunc="sum", fill_value=0).astype(int))
'''), md("## Per animal"), code('''
ba = pd.read_csv(coh / "capture_sites_by_animal.csv")
ba[ba.compartment != "all"].pivot_table(index=["animal", "lesion_section"], columns="compartment", values="n_pu1_eligible", aggfunc="sum", fill_value=0).astype(int)
'''), md("## Wells actually formed (target 3000 µm² per well)"), code('''
for r in runs:
    display(Markdown(f"### {r.name}"))
    w = r.wells
    display(w[w.well_group > 0][["section", "group", "well_group", "n_cells", "area_um2", "n_available", "note"]])
    short = w[(w.well_group == 0) | w.note.astype(str).str.startswith("short")]
    if len(short):
        display(Markdown(f"**Not filled / short:** {len(short)} of {len(w)} section×group combinations"))
'''), md("## Where the selected cells are"), code('''
for r in runs:
    les = r.sections[r.sections.is_lesion_section]
    n = len(les)
    if n == 0:
        continue
    fig, axes = plt.subplots(int(np.ceil(n / 2)), 2, figsize=(16, 8 * int(np.ceil(n / 2))), squeeze=False)
    for ax, sid in zip(axes.ravel(), les.section_id):
        R.plot_section(r, int(sid), ax=ax, scale=0.2, show_cells=False)
        c = r.cells[(r.cells.section_id == sid) & (r.cells.well_group > 0)]
        for grp, s in c.groupby(c.well_name.str.split("|").str[1]):
            ax.scatter(s.x_um, s.y_um, s=6, c=R.GROUP_COLORS.get(grp, "#ffffff"), linewidths=0, label=f"{grp} (n={len(s)})")
        ax.legend(markerscale=3, fontsize=8, loc="upper left")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{r.name}: cells selected for wells", y=1.0)
    plt.tight_layout(); plt.show()
'''), md("## Overlap with the collaborator's own selection (`cells_selected` in the table)"), code('''
for r in runs:
    c = r.cells
    theirs = c["cells_selected"].notna() & ~c["cells_selected"].astype(str).isin(["None", "nan"])
    display(Markdown(f"### {r.name}: collaborator selected {int(theirs.sum())} cells, this run selected {int((c.well_group > 0).sum())}; overlap {int((theirs & (c.well_group > 0)).sum())}"))
    display(pd.crosstab(c.loc[theirs, "manual_annotation"].fillna("ring (no manual label)"), c.loc[theirs, "zone"]))
'''), md("""
## LMD export

`lesionseg export-lmd` writes one Leica XML per scene with one well per `well_name`. It needs the
three calibration-mark coordinates in scene pixels; the demo below uses placeholders so the file
structure can be inspected.
"""), code('''
from lesionseg.export import export_lmd
r = runs[0]
calib = np.array([[500, 500], [r.log["scene_shape_px"][1] - 500, 500], [500, r.log["scene_shape_px"][0] - 500]], float)  # PLACEHOLDER
out = r.dir / "demo_lmd.xml"
counts = export_lmd(r.cells, out, calibration_points_px=calib, group_col="well_name", pixel_size_um=r.px, dilate_um=1.0)
print(f"{out}: {sum(counts.values())} shapes in {len(counts)} wells")
print(out.read_text()[:600])
''')]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--only", nargs="*", help="notebook name prefixes to (re)build")
    a = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    books = {"01_cohort_overview": nb_overview, "02_judge_lesions": nb_judge, "03_zones_relative_distance": nb_zones,
             "04_manual_vs_automatic": nb_manual, "05_capture_sites_and_wells": nb_wells}
    if a.only:
        books = {k: v for k, v in books.items() if any(k.startswith(o) for o in a.only)}
    for name, fn in books.items():
        nb = new_notebook(cells=fn(a.run_dir), metadata=KERNEL)
        nbf.write(nb, NB_DIR / f"{name}.ipynb")
        print("wrote", name)
    if not a.no_exec:
        for name in books:
            subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                            "--ExecutePreprocessor.timeout=3600", str(NB_DIR / f"{name}.ipynb")], check=True,
                           cwd=NB_DIR.parent)


if __name__ == "__main__":
    main()
