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
Meningeal cells and the buffer around them are never captured (see *Meninges, buffer and the edge
exclusion*).
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
'''), md("""
## Meninges, buffer and the edge exclusion

The meninges boundary is placed per location (`lesionseg.meninges`, `parenchyma.method: adaptive`):

* **meninges** – the outer 10 µm of the true surface (outermost nuclei), thin flaps / roots, and very
  compact surface-connected infiltrate (nuclei touching; nuclear area fraction above the 99.9th percentile
  of deep tissue), up to 80 µm deep;
* **buffer** – 10 µm inward of the meninges (and any cell within 6 µm of a meningeal cell): never
  collected, so meninges and lesion pools never touch;
* **manual lesion cores are protected** – nothing inside the collaborator's CORE polygons is ever
  meninges or buffer; where a core meets the meninges the buffer is carved from the meningeal side.
  Calibration on these scenes: without this, nuclear packing alone would call 7 % (99.9th percentile) to
  23 % (99th) of manual-core cells meninges – dense lymphocyte-rich subpial lesion looks like swollen
  meninges.

Lesion zones, wells and reactions are clipped to the parenchyma. On top of that, wells and reactions
skip cells within `edge_exclusion_um` of the section edge. That margin used to be 100 µm (the
collaborator's well protocol) and was the main thing keeping subpial lesion cells out; with meninges and
buffer excluded per cell it is now 40 µm, keeping off the damaged cut edge (25 µm let collected cells run
ragged up to the surface). The table below shows
what each margin would make available.
"""), code('''
surf = R.surface_summary(runs)
display(Markdown(f"**Manual-core cells in meninges or buffer: {int(surf.manual_core_cells_in_meninges_or_buffer.sum())}** (must be 0)"))
display(surf.groupby("lesion_section")[[c for c in surf.columns if c.startswith("pu1_")]].sum())
display(surf.set_index(["scene", "section"]))
'''), md("### What the edge exclusion costs: Pu.1⁺ parenchyma cells available per lesion compartment"), code('''
display(R.edge_exclusion_sweep(runs))
'''), md("### Where the meninges are thickest (raw image left, tiers right)"), code('''
for r in runs:
    spots = R.thickest_meninges_spots(r, n=2)
    if not spots:
        continue
    fig, axes = plt.subplots(len(spots), 2, figsize=(12, 6 * len(spots)), squeeze=False)
    for (x, y), row in zip(spots, axes):
        R.plot_surface_zoom(r, x, y, ax=row[0], raw=True)
        R.plot_surface_zoom(r, x, y, ax=row[1])
        row[0].set_title(f"{r.name} @ ({x:.0f}, {y:.0f}) µm · 250 µm", loc="left", fontsize=9)
    plt.tight_layout(); plt.show()
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


def nb_collection(run_dir):
    return [md("""
# 06 · Collection, section by section

For every section (animal × spinal level) on both replicate slides: **which cells are collected and
into which reaction**. Markers are collected cells coloured by compartment (core, rim, peri, deep, GM,
WM, VBO); grey dots are Pu.1⁺ cells in that section that are *not* collected (outside the eligible
compartments or within the edge exclusion). Every eligible cell is collected: ~250 per reaction is a
general target, not a cap (reactions below 80 % of it are flagged as shortfall).

**Meninges are kept apart from the lesion.** The meninges boundary follows the tissue per location
(`lesionseg.meninges`): the outer 10 µm of the surface plus very compact, surface-connected meningeal
infiltrate. Light-grey markers are Pu.1⁺ cells in the meninges, white rings Pu.1⁺ cells in the 10 µm
buffer inward of it; neither is ever collected, so meninges and lesion pools never touch. Solid grey line
= inner edge of the meninges, dashed white = inner edge of the buffer, dotted cyan = edge exclusion (no
cell outside it is collected); yellow = automatic lesion. Every line is in the legend. The
collaborator's manual lesion cores are protected: no cell inside them is ever meninges or buffer (checked
below). The table under each section lists its reactions and how the selected cells split between the two
slides. Aggregates at the end.
"""), code(SETUP.format(run_dir=run_dir)), code('''
plan = R.load_reaction_plan(RUN_DIR)
cells_by_scene = {r.name: R.attach_reactions(r) for r in runs}
budget = pd.read_csv(Path(RUN_DIR) / "cohort" / "reactions_budget.csv")
display(budget)
surf = R.surface_summary(runs)
sec_surf = surf.groupby("section")[["pu1_meninges", "pu1_buffer", "manual_core_cells_in_meninges_or_buffer"]].sum()
display(Markdown(f"**Manual-core cells in meninges or buffer: {int(surf.manual_core_cells_in_meninges_or_buffer.sum())}** "
                 "(must be 0)"))
sections = sorted({s for r in runs for s in r.sections.section_name.astype(str) if s not in ("unassigned", "nan")})
print(len(sections), "sections:", sections)
'''), md("## Sections"), code('''
for sec in sections:
    hits = [(r, int(r.sections.set_index("section_name").loc[sec, "section_id"])) for r in runs
            if sec in set(r.sections.section_name.astype(str))]
    if not hits:
        continue
    les = any(bool(r.sections.set_index("section_name").loc[sec, "is_lesion_section"]) for r, _ in hits)
    tab = R.section_collection_table(plan, sec, cells_by_scene)
    own = sum(int(((c.section_name.astype(str) == sec) & (c.reaction_id > 0)).sum()) for c in cells_by_scene.values())
    ms = sec_surf.loc[sec] if sec in sec_surf.index else None
    surf_txt = (f" · meninges {int(ms.pu1_meninges)} / buffer {int(ms.pu1_buffer)} Pu.1⁺ (not collected)"
                if ms is not None else "")
    display(Markdown(f"### {sec} — {'lesion section' if les else 'control section'} · {len(hits)} slide(s) · "
                     f"{own} cells collected from this section in {len(tab)} reaction(s){surf_txt}"))
    fig, axes = plt.subplots(1, len(hits), figsize=(10 * len(hits), 10), squeeze=False)
    for ax, (r, sid) in zip(axes.ravel(), hits):
        R.plot_section_collection(r, sid, ax=ax, cells=cells_by_scene[r.name])
    plt.tight_layout(); plt.show()
    if len(tab):
        display(tab.round(1))
'''), md("## Aggregates"), code('''
allc = pd.concat([c.assign(scene=k) for k, c in cells_by_scene.items()], ignore_index=True)
sel = allc[allc.reaction_id > 0]
display(Markdown(f"**{len(sel):,} cells collected in {sel.reaction_id.nunique()} reactions; "
                 f"{int(sel.pu1_pos.sum()):,} of them Pu.1⁺ ({100*sel.pu1_pos.mean():.0f} %).**"))
display(Markdown("### Per compartment (pooled over all sections)"))
agg = sel.groupby("compartment").agg(reactions=("reaction_id", "nunique"), cells=("cell_id", "size"),
                                      pu1_pos=("pu1_pos", "sum"), area_um2=("area_um2", "sum"),
                                      median_cell_area_um2=("area_um2", "median")).reindex(R.REACTION_ORDER).dropna(how="all")
display(agg.round(1))
display(Markdown("### Per section × compartment (cells collected)"))
display(sel.pivot_table(index="section_name", columns="compartment", values="cell_id", aggfunc="size", fill_value=0)
        .reindex(columns=[c for c in R.REACTION_ORDER if c in sel.compartment.unique()]).astype(int))
display(Markdown("### Per animal and level"))
sel = sel.assign(animal=sel.section_name.astype(str).str.replace(r"_[TCL]?$", "", regex=True),
                 level=sel.section_name.astype(str).str.extract(r"_([TCL])$")[0].fillna("?"))
display(sel.pivot_table(index=["animal", "level"], columns="compartment", values="cell_id", aggfunc="size", fill_value=0).astype(int))
display(Markdown("### Collected cells by manual annotation (sanity check)"))
display(pd.crosstab(sel.compartment, sel.manual_annotation.fillna("none")))
display(Markdown("### Surface tier of collected cells (must all be parenchyma, VBO excepted)"))
display(pd.crosstab(sel.compartment, sel.surface_tier.astype(str)))
display(Markdown("### Meninges and buffer per section (Pu.1⁺, never collected)"))
display(surf.set_index(["scene", "section"]))
'''), code('''
# cells per reaction vs the 250-cell target (a minimum to aim for, not a cap), and area per reaction
fig, axes = plt.subplots(1, 2, figsize=(16, 0.28 * len(plan) + 1.5))
d = plan.sort_values(["pool_type", "n_selected"])
col = [R.REACTION_COLORS.get(c, "#898781") for c in d.compartment]
axes[0].barh(d.reaction_name, d.n_selected, color=col, height=0.75)
axes[0].axvline(250, color="#898781", ls="--", lw=1); axes[0].set_title("cells selected per reaction (dashed = target 250)", loc="left")
axes[1].barh(d.reaction_name, d.area_selected_um2 / 1e3, color=col, height=0.75)
axes[1].set_title("nuclear area selected per reaction (×10³ µm²)", loc="left"); axes[1].set_yticklabels([])
for ax in axes:
    ax.tick_params(axis="y", labelsize=7)
plt.tight_layout()
'''), code('''
# size of collected cells per compartment
fig, ax = plt.subplots(figsize=(9, 4))
data = [sel.loc[sel.compartment == k, "area_um2"].dropna() for k in R.REACTION_ORDER if k in sel.compartment.unique()]
labels = [k for k in R.REACTION_ORDER if k in sel.compartment.unique()]
bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6)
ax.set_xticks(range(1, len(labels) + 1), labels)
for patch, k in zip(bp["boxes"], labels):
    patch.set_facecolor(R.REACTION_COLORS[k]); patch.set_alpha(0.85)
ax.set_ylabel("nuclear area (µm²)"); ax.set_title("size of collected cells per compartment", loc="left")
plt.tight_layout()
''')]


def nb_expression(run_dir):
    return [md("""
# 07 · Pu.1 expression by lesion proximity (exploratory)

Every Pu.1⁺ cell carries its own Pu.1 and Iba1 levels (nuclear mean intensity from the cell table), its
nuclear size / shape, its compartment and its signed distance to the lesion edge. Question: **do
Pu.1⁺ cells in the core express more Pu.1 than those in the rim, the peri-lesion area and further out?**

Compartments: `core`, `rim`, `peri`, `deep`, `distal` in lesion sections (parenchyma only), `meninges`,
and the manual `GM` / `WM` of control sections. Buffer and VBO cells are left out.

How to read it – four things that would otherwise mislead:

* **The unit is the section, not the cell.** With tens of thousands of cells any difference is
  "significant". Each comparison uses the median per section and slide, expressed as log2 ratio to the
  *same* section's distal Pu.1⁺ cells (paired, so scan and staining differences cancel); the two replicate
  slides of a section are averaged and a Wilcoxon signed-rank test runs over sections (n ≤ 8).
* **Crowding.** In packed tissue, signal from touching neighbours can raise a nucleus' mean intensity.
  Zones are compared again at matched local density (nuclei / Pu.1⁺ nuclei within 15 µm).
* **Truncation.** Pu.1⁺ cells were called by a threshold, so their intensities are a truncated
  distribution; the Pu.1⁺ fraction per compartment is shown alongside.
* **Composition.** A higher median can mean more Pu.1 per cell *or* a different mix of cells
  (resident microglia vs infiltrating monocyte-derived macrophages). Intensity alone cannot tell these apart.
"""), code(SETUP.format(run_dir=run_dir) + '''
from lesionseg import expression as E
df = E.expression_table(runs)
print(f"{len(df):,} cells with a compartment, {int(df.pu1_pos.sum()):,} Pu.1⁺")
df.groupby("compartment", observed=True).agg(cells=("pu1_pos", "size"), pu1_pos=("pu1_pos", "sum"),
                                             pu1_fraction=("pu1_pos", "mean")).round(3)
'''), md("## Paired per section: Pu.1⁺ cells vs the same section's distal Pu.1⁺ cells"), code('''
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
tests = {}
for ax, (v, label) in zip(axes.ravel(), E.MARKERS.items()):
    med = E.section_medians(df, v)
    E.plot_paired(med, ax, label)
    tests[label] = E.paired_tests(med)
plt.tight_layout(); plt.show()
for label, t in tests.items():
    display(Markdown(f"**{label}** – fold change = 2^median log2 over sections; Wilcoxon over sections (n ≤ 8, min p = 0.008)"))
    display(t.round(4))
'''), md("""
## Gradient: Pu.1 and Iba1 against distance to the lesion edge

Median of Pu.1⁺ cells per distance bin, one line per slide, normalised to the slide's control-section
Pu.1⁺ cells (1 = control level). Negative distances are inside the lesion.
"""), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
E.plot_gradient(E.gradient(df, "Pu1_mean_norm"), axes[0], "Pu.1 of Pu.1⁺ cells", "Pu.1 / control level")
E.plot_gradient(E.gradient(df, "Iba1_mean_norm"), axes[1], "Iba1 of Pu.1⁺ cells", "Iba1 / control level")
plt.tight_layout(); plt.show()
'''), md("""
## Crowding check: zones at matched local density

If the lesion effect were only signal bleeding in from packed neighbours, the zones would converge once
local density is matched. Bins are quartiles of the number of nuclei (left) or Pu.1⁺ nuclei (right)
within 15 µm, over Pu.1⁺ cells in lesion sections.
"""), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
t1 = E.density_matched(df, "Pu1_mean", "n_nb_15um")
t2 = E.density_matched(df, "Pu1_mean", "n_pu1_nb_15um")
E.plot_density_matched(t1, axes[0], "Pu.1 at matched nuclear density", "nuclei within 15 µm (quartile bin)")
E.plot_density_matched(t2, axes[1], "Pu.1 at matched Pu.1⁺ density", "Pu.1⁺ nuclei within 15 µm (bin)")
plt.tight_layout(); plt.show()
display(t1.round(2)); display(t2.round(2))
'''), md("## Distributions (cells pooled – descriptive only, not a test)"), code('''
pos = df[df.pu1_pos]
comps = [c for c in E.ORDER if c in set(pos.compartment.astype(str))]
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8))
for ax, v, label in ((axes[0], "Pu1_mean_norm", "Pu.1 / control level"), (axes[1], "Iba1_mean_norm", "Iba1 / control level")):
    data = [pos.loc[pos.compartment == c, v].dropna().to_numpy() for c in comps]
    bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.6, medianprops={"color": "#0b0b0b"})
    for patch, c in zip(bp["boxes"], comps):
        patch.set_facecolor(E.COLORS[c]); patch.set_alpha(0.85)
    ax.set_xticks(range(1, len(comps) + 1), comps); ax.axhline(1, color="#898781", lw=0.8, ls=":")
    ax.set_ylabel(label); ax.set_title(label.split(" /")[0] + " of Pu.1⁺ cells per compartment", loc="left")
plt.tight_layout(); plt.show()
'''), md("## Where the high-Pu.1 cells are: lesion sections on both slides"), code('''
for sec in ["R1_2_T", "P2_3_T", "P3_1_C"]:
    hits = [r for r in runs if sec in set(r.sections.section_name.astype(str))]
    fig, axes = plt.subplots(1, len(hits), figsize=(9 * len(hits), 9), squeeze=False)
    for ax, r in zip(axes.ravel(), hits):
        sc = E.plot_section_map(r, df, sec, ax)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="Pu.1 / control level")
    plt.show()
'''), md("""
## Reading

* Compare the paired panels first: they are the evidence. A compartment counts as different when most
  sections move the same way (`sections_higher`) – the p-values have little power with 8 sections and are
  not corrected for the four markers × five compartments tested.
* If Pu.1 is highest at the **rim** rather than the core, that fits an active edge (recruitment /
  activation at the border) – a hypothesis for the DVP data, which will measure the proteome directly.
* Smaller nuclei together with higher Pu.1 / Iba1 near lesions could mean activated or infiltrating
  myeloid cells; a marker that separates resident microglia from monocyte-derived macrophages
  (e.g. TMEM119 / P2RY12 vs CCR2) would be needed to separate composition from per-cell expression.
""")]


def nb_expression_split(run_dir):
    return [md("""
# 08 · Pu.1 by lesion proximity: per animal / spinal level, and per cell vs composition

Follow-up to notebook 07, which found Pu.1⁺ cells near lesions (rim > peri > core > deep) carrying more
Pu.1 than distal Pu.1⁺ cells of the same section. Two questions:

1. **Is it everywhere?** The same paired comparison for every section on both replicate slides, grouped by
   animal (`P2_3`, `P3_1`, `R1_2`, `R1_3`) and spinal level (C cervical, T thoracic, L lumbar).
2. **More Pu.1 per cell, or a different mix of cells?** Pu.1⁺ cells are split into data-driven types
   (Gaussian mixture on Iba1 level, nuclear area and eccentricity), then (a) the mix of types is compared
   between compartments and (b) the comparison is repeated *within* each type. A decomposition splits
   the compartment-vs-distal difference into a composition part and a per-cell part.

Values are log2(section median / same section's distal Pu.1⁺ median) unless stated; with 8 sections the
tests are descriptive (min Wilcoxon p = 0.008, not corrected for multiple comparisons).
"""), code(SETUP.format(run_dir=run_dir) + '''
from lesionseg import expression as E
df = E.expression_table(runs)
eff_pu1 = E.section_effects(df, "Pu1_mean")
eff_iba = E.section_effects(df, "Iba1_mean")
print(f"{len(eff_pu1)} slide × section combinations in lesion sections")
'''), md("## 1 · Per section, slide, animal and level"), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
im = E.plot_effect_heatmap(eff_pu1, axes[0], "Pu.1 of Pu.1⁺ cells, log2 vs distal")
E.plot_effect_heatmap(eff_iba, axes[1], "Iba1 of Pu.1⁺ cells, log2 vs distal", row_labels=False)
fig.colorbar(im, ax=axes, shrink=0.5, label="log2 (compartment / distal)")
plt.show()
display(Markdown(f"Rim above distal in **{int((eff_pu1.rim > 0).sum())} of {int(eff_pu1.rim.notna().sum())}** slide × section "
                 f"combinations; peri {int((eff_pu1.peri > 0).sum())} of {int(eff_pu1.peri.notna().sum())}; "
                 f"core {int((eff_pu1.core > 0).sum())} of {int(eff_pu1.core.notna().sum())}."))
'''), md("### By spinal level and animal (each dot = one section on one slide)"), code('''
levels = ["C", "T", "L"]
animals = sorted(eff_pu1.animal.unique())
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), sharey=True)
for ax, comp in zip(axes, ["core", "rim", "peri"]):
    for k, (an, col) in enumerate(zip(animals, E.SCENE_COLORS)):
        d = eff_pu1[eff_pu1.animal == an]
        x = d.level.map({l: i for i, l in enumerate(levels)}) + (k - 1.5) * 0.08
        ax.scatter(x, d[comp], s=50, color=col, edgecolors="white", linewidths=0.8, label=an, zorder=3)
    for i, l in enumerate(levels):
        v = eff_pu1.loc[eff_pu1.level == l, comp].dropna()
        if len(v):
            ax.plot([i - 0.3, i + 0.3], [v.median()] * 2, color="#0b0b0b", lw=2)
    ax.axhline(0, color="#898781", lw=0.8, ls=":")
    ax.set_xticks(range(3), ["cervical", "thoracic", "lumbar"]); ax.set_title(f"{comp} vs distal (Pu.1)", loc="left")
axes[0].set_ylabel("log2 (section median / distal)"); axes[-1].legend(fontsize=8, frameon=False, title="animal")
plt.tight_layout(); plt.show()
display(eff_pu1.groupby("level")[["core", "rim", "peri", "deep", "meninges"]].median().reindex(levels).round(2))
display(eff_pu1.groupby("animal")[["core", "rim", "peri", "deep", "meninges"]].median().round(2))
'''), md("### Replicate slides agree?"), code('''
long = eff_pu1.melt(id_vars=["scene", "section"], value_vars=["core", "rim", "peri", "deep", "meninges"],
                    var_name="compartment", value_name="log2").dropna()
long["slide"] = long.groupby(["section", "compartment"]).cumcount()
pairs = long.pivot_table(index=["section", "compartment"], columns="slide", values="log2").dropna()
fig, ax = plt.subplots(figsize=(5.5, 5.5))
for comp, d in pairs.groupby(level="compartment"):
    ax.scatter(d[0], d[1], s=40, color=E.COLORS[comp], edgecolors="white", linewidths=0.8, label=comp)
lim = [min(pairs.min().min(), 0) - 0.05, pairs.max().max() + 0.05]
ax.plot(lim, lim, color="#898781", lw=0.8, ls=":"); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("slide 1: log2 vs distal"); ax.set_ylabel("slide 2: log2 vs distal")
ax.set_title(f"replicate slides, Spearman r = {pairs[0].corr(pairs[1], method='spearman'):.2f}", loc="left")
ax.legend(fontsize=8, frameon=False); plt.show()
'''), md("### Gradient per animal and per level (slides pooled; Pu.1 relative to each slide's control level)"), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 4.8), sharey=True)
for ax, by, order in ((axes[0], "animal", animals), (axes[1], "level", levels)):
    g = E.gradient_by(df, "Pu1_mean_norm", by)
    for key, col in zip(order, E.SCENE_COLORS):
        d = g[g[by] == key].sort_values("mid_um")
        ax.plot(d.mid_um, d["median"], color=col, lw=2, marker="o", ms=4, label=key)
    ax.axvline(0, color="#0b0b0b", lw=0.8); ax.axvspan(g.mid_um.min() - 40, 0, color="#e66767", alpha=0.08, lw=0)
    ax.set_xlabel("signed distance to the lesion edge (µm; negative = inside)"); ax.set_title(f"by {by}", loc="left")
    ax.legend(fontsize=8, frameon=False)
axes[0].set_ylabel("Pu.1 / control level")
plt.tight_layout(); plt.show()
'''), md("""
## 2 · More Pu.1 per cell, or a different mix of cells?

Pu.1⁺ cells are split by a Gaussian mixture on log2 Iba1 (vs the slide's control level), log2 nuclear
area and eccentricity. BIC stops improving after three components; they are named from their means.
These are **morphological types, not validated cell identities** – the `small` type in particular may
partly be segmentation fragments.
"""), code('''
ct, prof, bic = E.cell_types(df)
display(bic.round(0).to_frame().T)
display(prof.round(2))
pos = ct[ct.cell_type.notna()].sample(20000, random_state=0)
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
for t in E.CELL_TYPE_ORDER:
    d = pos[pos.cell_type == t]
    axes[0].scatter(d.area_um2, d.eccentricity, s=3, alpha=0.35, color=E.CELL_TYPE_COLORS[t], linewidths=0, label=t)
    axes[1].hist(np.log2(d.Iba1_mean_norm.clip(lower=0.05)), bins=60, histtype="step", lw=2,
                 color=E.CELL_TYPE_COLORS[t], label=t, density=True)
axes[0].set_xlabel("nuclear area (µm²)"); axes[0].set_ylabel("eccentricity"); axes[0].set_title("types in shape space", loc="left")
axes[0].legend(markerscale=5, fontsize=8, frameon=False)
axes[1].set_xlabel("log2 Iba1 / control level"); axes[1].set_title("Iba1 per type", loc="left"); axes[1].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
'''), md("### (a) The mix of types per compartment (median share over slide × section)"), code('''
comp = E.composition(ct)
share = comp.groupby("compartment", observed=True)[E.CELL_TYPE_ORDER].median().reindex(E.LESION_ORDER)
fig, ax = plt.subplots(figsize=(9, 4))
left = np.zeros(len(share))
for t in E.CELL_TYPE_ORDER:
    ax.barh(share.index, share[t], left=left, color=E.CELL_TYPE_COLORS[t], edgecolor="white", linewidth=2, label=t)
    for i, (l, v) in enumerate(zip(left, share[t])):
        ax.text(l + v / 2, i, f"{100 * v:.0f} %", ha="center", va="center", fontsize=8, color="#0b0b0b")
    left += share[t].to_numpy()
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("share of Pu.1⁺ cells")
ax.legend(fontsize=8, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1))
ax.set_title("cell-type mix per compartment", loc="left"); plt.tight_layout(); plt.show()
display(share.round(3))
'''), md("### (b) Pu.1 within each type, paired per section"), code('''
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
within = {}
for ax, t in zip(axes, E.CELL_TYPE_ORDER):
    med = E.section_medians(ct[ct.cell_type == t], "Pu1_mean", min_cells=15)
    E.plot_paired(med, ax, f"Pu.1 within {t} cells")
    within[t] = E.paired_tests(med)
plt.tight_layout(); plt.show()
for t, tab in within.items():
    display(Markdown(f"**{t}**")); display(tab.round(4))
'''), md("""
### (c) Decomposition: how much of the difference is composition?

For each slide × section, the difference in *mean* Pu.1 between a compartment and distal is split into a
per-cell part (distal mix of types, compartment's per-type Pu.1) and a composition part (the rest).
`share_per_cell` ≈ 1 means the difference is per-cell expression, not a shift in which cells are there.
"""), code('''
summ, per = E.composition_vs_expression(ct)
display(summ.round(3))
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(summ))
ax.bar(x - 0.18, summ.per_cell, width=0.34, color="#2a78d6", label="per-cell expression")
ax.bar(x + 0.18, summ.composition, width=0.34, color="#eda100", label="cell-type composition")
ax.axhline(0, color="#898781", lw=0.8)
ax.set_xticks(x, summ.index); ax.set_ylabel("Δ mean Pu.1 vs distal (median over slide × section)")
ax.set_title("where the difference comes from", loc="left"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
'''), md("""
## Reading

* If every animal and level shows the rim above distal, and both replicate slides agree, the edge effect
  is a property of the lesions, not of one animal, one level or one scan.
* If `share_per_cell` is close to 1 and Pu.1 rises *within* each type, the higher Pu.1 near lesions is
  more Pu.1 per cell – an activation signal – rather than a different cell population moving in.
  Morphological types are a coarse proxy; a resident-vs-infiltrating marker (TMEM119 / P2RY12 vs CCR2)
  or the DVP proteomes themselves would test this directly.
""")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--no-exec", action="store_true")
    ap.add_argument("--only", nargs="*", help="notebook name prefixes to (re)build")
    a = ap.parse_args()
    NB_DIR.mkdir(exist_ok=True)
    books = {"01_cohort_overview": nb_overview, "02_judge_lesions": nb_judge, "03_zones_relative_distance": nb_zones,
             "04_manual_vs_automatic": nb_manual, "05_capture_sites_and_wells": nb_wells,
             "06_collection_by_section": nb_collection, "07_expression_by_zone": nb_expression,
             "08_expression_by_animal_and_cell_type": nb_expression_split}
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
