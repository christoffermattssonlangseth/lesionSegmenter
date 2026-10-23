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


def nb_elongation(run_dir):
    return [md("""
# 09 · Nuclear elongation: round vs elongated nuclei in core, rim, distal tissue and meninges

Every segmented nucleus has an **eccentricity**: that of the ellipse with the same second moments as the
Cellpose nucleus mask, √(1 − (b/a)²) – 0 is a circle, 1 a line. It describes the *nucleus*, not the
cell's processes. Classes used here:

| class | eccentricity | long : short axis |
|---|---|---|
| round | < 0.6 | < 1.25 |
| intermediate | 0.6 – 0.85 | 1.25 – 1.9 |
| elongated | ≥ 0.85 | ≥ 1.9 |

Outlines are drawn from the label image (every QC-passed nucleus, Pu.1⁺ and Pu.1⁻; faint nuclei that
Cellpose filtered out have no outline). Compartments as in notebook 07: `core`, `rim`, `peri`, `deep`,
`distal` (lesion sections, parenchyma), `meninges`, control `GM` / `WM`. In every zoom-in the nuclei of
the window's compartment are drawn at full strength and all others faded, with the inner edges of the
meninges (solid grey) and buffer (dashed white); the line above each panel counts the nuclei.

**`meninges` is a place, not a cell type.** The compartment is the outer surface layer plus the compact
infiltrate attached to it (notebook 05), so it holds true meningeal cells (flattened pial / arachnoid
fibroblasts) *and* immune cells piled up in the swollen subarachnoid space. There is no meningeal marker
in this panel (SytoG, Pu.1, Iba1, DAPI); Pu.1⁻ meningeal nuclei are the better proxy for meningeal cells,
Pu.1⁺ ones are infiltrating myeloid cells – both are shown separately below.
"""), code(SETUP.format(run_dir=run_dir) + '''
from lesionseg import expression as E, morphology as M
df = E.expression_table(runs)
cells_by_scene = {r.name: r.cells for r in runs}
# windows: surface meninges only (flaps / roots deeper than 80 µm are often sparse or artefact-ridden)
comp_by_scene = {r.name: M.window_compartments(r.cells, E.compartment(r.cells)) for r in runs}
SIZE = 120  # µm per zoom-in
'''), md("## Side by side: one typical window per compartment (same scale, 120 µm)"), code('''
order = ["core", "rim", "distal", "meninges"]
fig, axes = plt.subplots(2, 4, figsize=(22, 11.5))
for j, comp in enumerate(order):
    for r in runs:
        w = M.representative_windows(r.cells, comp_by_scene[r.name], comp, size_um=SIZE, n=1,
                                     min_frac=0.5 if comp == "meninges" else 0.8)
        if w:
            break
    x, y = w[0]
    M.plot_nuclei(axes[0, j], r, r.cells, x, y, SIZE, raw=True)
    st = M.plot_nuclei(axes[1, j], r, r.cells, x, y, SIZE, comp=comp_by_scene[r.name], focus=comp)
    axes[0, j].set_title(f"{comp} · {r.name} @ ({x:.0f}, {y:.0f}) µm", loc="left", fontsize=10)
    axes[1, j].set_title(M.stats_line(comp, st), loc="left", fontsize=9)
M.class_legend(axes[1, -1])
plt.tight_layout(); plt.show()
'''), md("""
## How common are elongated and round nuclei per compartment?

Share of each class per slide × section (median over slide × sections), for all nuclei, Pu.1⁺ and Pu.1⁻.
"""), code('''
comps = ["core", "rim", "peri", "deep", "distal", "meninges", "GM", "WM"]
fig, axes = plt.subplots(1, 3, figsize=(20, 4.8), sharey=True)
tabs = {}
for ax, (label, which) in zip(axes, (("all nuclei", "all"), ("Pu.1⁺", "pos"), ("Pu.1⁻", "neg"))):
    sh = M.class_shares(df, which).groupby("compartment", observed=True)[M.CLASS_ORDER].median().reindex(comps)
    sh = sh.div(sh.sum(axis=1), axis=0)
    tabs[label] = sh
    left = np.zeros(len(sh))
    for k in M.CLASS_ORDER:
        ax.barh(sh.index, sh[k], left=left, color=M.CLASS_COLORS[k], edgecolor="white", linewidth=2, label=k)
        for i, (l, v) in enumerate(zip(left, sh[k])):
            if v > 0.06:
                ax.text(l + v / 2, i, f"{100 * v:.0f} %", ha="center", va="center", fontsize=8, color="#0b0b0b")
        left += sh[k].to_numpy()
    ax.set_xlim(0, 1); ax.set_title(label, loc="left"); ax.set_xlabel("share of nuclei")
axes[0].invert_yaxis(); axes[-1].legend(fontsize=8, frameon=False, ncol=3, loc="lower right", bbox_to_anchor=(1, 1))
plt.tight_layout(); plt.show()
for label, sh in tabs.items():
    display(Markdown(f"**{label}**")); display((100 * sh).round(1))
'''), md("### Eccentricity distributions (all nuclei, pooled – descriptive)"), code('''
fig, axes = plt.subplots(1, 2, figsize=(16, 4.5), sharey=True)
bins = np.linspace(0, 1, 41)
for ax, group in ((axes[0], ["core", "rim", "peri", "distal", "meninges"]), (axes[1], ["distal", "GM", "WM", "meninges"])):
    for c in group:
        v = df.loc[df.compartment == c, "eccentricity"].dropna()
        ax.hist(v, bins=bins, histtype="step", lw=2, density=True, color=E.COLORS[c], label=f"{c} (median {v.median():.2f})")
    for t in (M.ROUND_MAX, M.ELONG_MIN):
        ax.axvline(t, color="#898781", lw=0.8, ls=":")
    ax.set_xlabel("eccentricity"); ax.legend(fontsize=8, frameon=False, loc="upper left")
axes[0].set_ylabel("density"); axes[0].set_title("lesion compartments vs meninges", loc="left"); axes[1].set_title("control tissue vs meninges", loc="left")
plt.tight_layout(); plt.show()
'''), md("### Elongated share vs the same section's distal tissue (paired; one line per section)"), code('''
sh = M.class_shares(df, "all")
ref = sh[sh.compartment == "distal"].set_index(["scene", "section"])["elongated"]
sh["d_elongated"] = sh["elongated"] - sh.set_index(["scene", "section"]).index.map(ref).to_numpy()
per = sh.groupby(["section", "compartment"], observed=True)["d_elongated"].mean().unstack("compartment")
cols = ["core", "rim", "peri", "deep", "distal", "meninges"]
per = per.reindex(columns=cols)
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(cols))
for _, row in per.iterrows():
    ax.plot(x, 100 * row.to_numpy(float), color="#c3c2b7", lw=1)
for i, c in enumerate(cols):
    v = 100 * per[c].dropna()
    ax.scatter(np.full(len(v), i), v, s=36, color=E.COLORS[c], edgecolors="white", linewidths=0.8, zorder=3)
    ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#0b0b0b", lw=2)
ax.axhline(0, color="#898781", lw=0.8, ls=":"); ax.set_xticks(x, cols)
ax.set_ylabel("elongated share − distal (percentage points)"); ax.set_title("elongated nuclei, paired per section", loc="left")
plt.tight_layout(); plt.show()
display((100 * per).round(1))
'''), md("""
### Meninges split: Pu.1⁻ (proxy for meningeal cells) vs Pu.1⁺ (infiltrating myeloid cells)

Elongated share per slide × section in the meninges, against the same section's distal tissue of the same
Pu.1 status (percentage points; one dot per section, replicate slides averaged).
"""), code('''
rows = []
for which, label in (("neg", "Pu.1⁻"), ("pos", "Pu.1⁺")):
    sh = M.class_shares(df, which)
    ref = sh[sh.compartment == "distal"].set_index(["scene", "section"])["elongated"]
    m = sh[sh.compartment == "meninges"].copy()
    m["distal"] = m.set_index(["scene", "section"]).index.map(ref).to_numpy()
    per = m.groupby("section")[["elongated", "distal"]].mean().dropna()
    per["delta_pp"] = 100 * (per["elongated"] - per["distal"])
    per["group"] = label
    rows.append(per)
men = pd.concat(rows)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for k, (label, d) in enumerate(men.groupby("group", sort=False)):
    ax.scatter(np.full(len(d), k) + np.linspace(-0.12, 0.12, len(d)), d.delta_pp, s=40, color="#c3c2b7",
               edgecolors="#0b0b0b", linewidths=0.5, zorder=3)
    ax.plot([k - 0.25, k + 0.25], [d.delta_pp.median()] * 2, color="#0b0b0b", lw=2)
ax.axhline(0, color="#898781", lw=0.8, ls=":"); ax.set_xticks([0, 1], ["Pu.1⁻ meninges", "Pu.1⁺ meninges"])
ax.set_ylabel("elongated share − distal (pp)"); ax.set_title("meninges vs distal, by Pu.1 status", loc="left")
plt.tight_layout(); plt.show()
display(men.groupby("group", sort=False).agg(meninges_elongated=("elongated", "median"), distal_elongated=("distal", "median"),
        delta_pp=("delta_pp", "median"), sections_higher=("delta_pp", lambda v: int((v > 0).sum())), n=("delta_pp", "size")).round(3))
'''), md("""
## Representative zoom-ins per compartment

Three typical windows per compartment, from different slides where possible: the compartment covers most
of the window (≥ 80 %, meninges ≥ 50 % since the band is thin), the window holds ≥ 40 nuclei, and its
median eccentricity is close to the compartment's own median (not the most extreme spots). Meninges
windows come from the surface meninges only, not from thin flaps or roots. Top = image, bottom = every nucleus filled by class; white edge = Pu.1⁺.
"""), code('''
for comp in ["core", "rim", "distal", "meninges"]:
    picks = []
    for r in runs:
        for w in M.representative_windows(r.cells, comp_by_scene[r.name], comp, size_um=SIZE, n=1,
                                          min_frac=0.5 if comp == "meninges" else 0.8):
            picks.append((r, w))
    picks = picks[:3]
    fig, axes = plt.subplots(2, len(picks), figsize=(6.5 * len(picks), 13), squeeze=False)
    for j, (r, (x, y)) in enumerate(picks):
        M.plot_nuclei(axes[0, j], r, r.cells, x, y, SIZE, raw=True)
        st = M.plot_nuclei(axes[1, j], r, r.cells, x, y, SIZE, comp=comp_by_scene[r.name], focus=comp)
        axes[0, j].set_title(f"{comp} · {r.name} @ ({x:.0f}, {y:.0f}) µm", loc="left", fontsize=10)
        axes[1, j].set_title(M.stats_line(comp, st), loc="left", fontsize=9)
    M.class_legend(axes[1, -1])
    fig.suptitle(comp, x=0.01, ha="left", fontsize=14)
    plt.tight_layout(); plt.show()
'''), md("""
## Reading

* The meningeal compartment carries the most elongated nuclei. Where it is a thin surface layer the
  window is mostly Pu.1⁻ flattened nuclei lying along the surface – the classic meningeal layer; where it
  is swollen it is mostly Pu.1⁺ infiltrate.
* Relative to their own distal baseline, Pu.1⁻ and Pu.1⁺ meningeal nuclei are elongated by a similar
  margin (~10 percentage points each). Pu.1⁻ nuclei are more elongated in absolute terms everywhere, so
  raw shares overstate the difference between the two. Identifying meningeal cells properly needs a marker
  (e.g. a fibroblast / leptomeningeal marker).
* Lesion compartments differ little from distal tissue in nuclear shape; what changes near lesions is
  Pu.1 / Iba1 level and nuclear size (notebooks 07–08).
* Eccentricity says nothing about orientation; a next step could measure whether elongated nuclei lie
  parallel to the surface (meninges) or along fibre tracts / vessels (white matter, perivascular cuffs).
""")]


def nb_meninges_reliability(run_dir):
    return [md("""
# 10 · Are meningeal nuclei really more elongated? Random examples and statistics

Notebook 09 showed *typical* windows – still a choice. Here the evidence is built so that it does not
depend on choices:

1. **Statistics with the section as the unit.** Every section has meninges, so all sections are used
   (lesion and control, 17 sections from 6 animals), each compared with its *own* parenchyma (distal in
   lesion sections, GM / WM in control sections); replicate slides are averaged. Tests: sign test and
   Wilcoxon over sections, and a **hierarchical bootstrap** (resample animals, then sections) for a 95 %
   confidence interval that respects that sections from one animal are not independent.
2. **Robustness checks.** The effect has to survive other cut-offs, the continuous median eccentricity,
   well-segmented nuclei only (solidity ≥ 0.9: merged nuclei look elongated), size-matched nuclei,
   Pu.1⁻ and Pu.1⁺ separately, and an **edge-artefact control** – parenchyma 20–80 µm under the surface
   instead of distal tissue (nuclei at a cut edge can be squashed or partly cut).
3. **Depth profile.** Elongated share against depth below the surface, per slide – no compartment
   boundary involved.
4. **Random galleries.** Windows drawn at random (fixed seed) among surface-meninges nuclei and, for
   comparison, distal nuclei – nothing selected on elongation – with a tally of how many random meninges
   windows beat their slide's parenchyma.

Meninges here = surface meninges (≤ 80 µm deep; flaps / roots excluded). As in notebook 09, *meninges
is a place*: meningeal cells plus infiltrate.
"""), code(SETUP.format(run_dir=run_dir) + '''
from lesionseg import expression as E, morphology as M
df = E.expression_table(runs)
rob = M.robustness_table(df)
rob.round(4)
'''), md("## Effect under every check (mean Δ meninges − parenchyma, 95 % hierarchical-bootstrap CI)"), code('''
fig, ax = plt.subplots(figsize=(10, 5.2))
y = np.arange(len(rob))[::-1]
ax.hlines(y, rob.ci95_low, rob.ci95_high, color="#2a78d6", lw=3)
ax.scatter(rob.mean_delta, y, s=60, color="#0b0b0b", zorder=3)
for yi, r in zip(y, rob.itertuples()):
    ax.text(r.ci95_high + 0.4, yi, f"{r.sections_higher}/{r.sections} sections higher", va="center", fontsize=8, color="#52514e")
ax.axvline(0, color="#898781", lw=0.8, ls=":")
ax.set_yticks(y, rob.check, fontsize=9)
ax.set_xlabel("meninges − parenchyma (percentage points; median eccentricity ×100)")
ax.set_title("meningeal elongation: one row per check", loc="left")
ax.set_xlim(left=min(-1, rob.ci95_low.min() - 1), right=rob.ci95_high.max() + 9)
plt.tight_layout(); plt.show()
'''), md("## Per section: meninges vs the section's own parenchyma (replicate slides averaged)"), code('''
per = M.meninges_effect(df).sort_values(["animal", "section"])
animals = sorted(per.animal.unique())
acol = dict(zip(animals, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]))
fig, ax = plt.subplots(figsize=(13, 4.8))
x = np.arange(len(per))
for xi, r in zip(x, per.itertuples()):
    ax.plot([xi, xi], [r.parenchyma, r.meninges], color="#c3c2b7", lw=2, zorder=1)
ax.scatter(x, per.parenchyma, s=40, color="#898781", label="parenchyma", zorder=2)
ax.scatter(x, per.meninges, s=60, color=[acol[a] for a in per.animal], edgecolors="#0b0b0b", linewidths=0.5,
           label="meninges (colour = animal)", zorder=3)
ax.set_xticks(x, per.section, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("elongated nuclei (%)"); ax.set_title("elongated share per section", loc="left")
ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
display(per.round(2))
'''), md("## Depth profile: elongated share against depth below the surface"), code('''
dp = M.depth_profile(df)
fig, ax = plt.subplots(figsize=(10, 4.5))
for (scene, d), col in zip(dp.groupby("scene"), E.SCENE_COLORS):
    ax.plot(d.mid_um, d["mean"], color=col, lw=2, marker="o", ms=4, label=scene)
ax.axvspan(0, 10, color="#c3c2b7", alpha=0.3, lw=0)
ax.text(1, ax.get_ylim()[1], "outer 10 µm", va="top", fontsize=8, color="#52514e")
ax.set_xlabel("depth below the surface (µm)"); ax.set_ylabel("elongated nuclei (%)")
ax.set_title("elongation is a surface property", loc="left"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
'''), md("""
## Random gallery: meninges (6 random windows per slide, 120 µm)

Window centres are random surface-meninges nuclei (seed 0), kept if the window holds ≥ 15 meningeal
nuclei and lies ≥ 300 µm from earlier draws. Meningeal nuclei at full strength, all others faded; the line
above each panel counts the meningeal nuclei.
"""), code('''
SIZE, N = 120, 6
comp_by_scene = {r.name: M.window_compartments(r.cells, E.compartment(r.cells)) for r in runs}
ref_share = (df[M.reference_compartment(df) == "parenchyma"].assign(el=lambda d: d.eccentricity >= M.ELONG_MIN)
             .groupby("scene")["el"].mean())
tally = []
def gallery(target, n, seed=0):
    rows = []
    for r in runs:
        for w in M.random_windows(r.cells, comp_by_scene[r.name], target, n, size_um=SIZE, seed=seed):
            rows.append((r, w))
    ncol = 6
    nrow = int(np.ceil(len(rows) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 4.3 * nrow), squeeze=False)
    for ax, (r, (x, y)) in zip(axes.ravel(), rows):
        st = M.plot_nuclei(ax, r, r.cells, x, y, SIZE, comp=comp_by_scene[r.name], focus=target)
        ax.set_title(f"{r.name.replace(' scene ', ' s')} · {st['n_focus']} nuclei · {100 * st['elongated_focus']:.0f} % elong · {100 * st['pu1_focus']:.0f} % Pu.1⁺",
                     loc="left", fontsize=8)
        tally.append({"target": target, "scene": r.name, "x": x, "y": y, **st,
                      "slide_parenchyma_elongated": ref_share[r.name]})
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    M.class_legend(axes.ravel()[len(rows) - 1])
    plt.tight_layout(); plt.show()
gallery("meninges", N)
'''), md("## Random gallery: distal parenchyma for comparison (3 random windows per slide)"), code('''
gallery("distal", 3)
'''), md("## Tally: random windows vs their slide's parenchyma"), code('''
t = pd.DataFrame(tally)
t["higher_than_parenchyma"] = t.elongated_focus > t.slide_parenchyma_elongated
display(t.groupby("target").agg(windows=("x", "size"), median_elongated=("elongated_focus", "median"),
                                 slide_parenchyma=("slide_parenchyma_elongated", "median"),
                                 windows_higher=("higher_than_parenchyma", "sum")).round(3))
display(t.round(3))
'''), md("""
### Why single random windows disagree: the thin surface layer vs swollen meninges

One 120 µm window holds 20–130 meningeal nuclei – too few for a stable share (± ~8 points at 30 nuclei) –
and the meninges are not uniform. The section-level test below splits the meningeal nuclei by depth below
the surface: the thin outer layer against the deeper, swollen part.
"""), code('''
from scipy.stats import wilcoxon
lay = M.meninges_layers(df)
el = lay.pivot_table(index="section", columns="layer", values="el", observed=True)
pu = lay.pivot_table(index="section", columns="layer", values="pu1", observed=True)
cols = list(el.columns)
fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
for ax, tab, lab in ((axes[0], el, "elongated nuclei (%)"), (axes[1], pu, "Pu.1⁺ nuclei (%)")):
    x = np.arange(len(cols))
    for _, row in tab.iterrows():
        ax.plot(x, row.to_numpy(float), color="#c3c2b7", lw=1)
    for i, c in enumerate(cols):
        v = tab[c].dropna()
        ax.scatter(np.full(len(v), i), v, s=36, color="#eb6834" if tab is el else "#c98500", edgecolors="white", linewidths=0.8, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#0b0b0b", lw=2)
    ax.set_xticks(x, cols); ax.set_xlabel("depth within the meninges"); ax.set_ylabel(lab)
axes[0].axhline(100 * ref_share.median(), color="#898781", lw=1, ls=":")
axes[0].text(len(cols) - 1, 100 * ref_share.median(), " parenchyma", va="bottom", fontsize=8, color="#52514e")
axes[0].set_title("elongation by meningeal layer (one line per section)", loc="left")
axes[1].set_title("Pu.1⁺ share by meningeal layer", loc="left")
plt.tight_layout(); plt.show()
rows = []
for c in cols[1:]:
    dd = (el[cols[0]] - el[c]).dropna()
    rows.append({"outer layer vs": c, "sections": len(dd), "median_delta_pp": dd.median(),
                 "sections_outer_higher": int((dd > 0).sum()), "wilcoxon_p": wilcoxon(dd).pvalue})
display(pd.DataFrame(rows).round(4))
display(pd.concat({"elongated %": el.median(), "Pu.1⁺ %": pu.median()}, axis=1).round(1))
'''), md("""
## Reading

* **Reliable, at the right unit.** Pooled per section, meninges carry more elongated nuclei than their own
  parenchyma in 16 of 17 sections from 6 animals (+13 points, 95 % hierarchical-bootstrap CI ~10–17), and
  the effect survives every check – including the edge control: parenchyma right under the surface is
  not equally elongated, so this is not a cut-edge or squashing artefact.
* **It is the thin surface layer.** Split by depth, the outer ≤ 10 µm of the meninges is strongly
  elongated and ~80 % Pu.1⁻ (flattened nuclei along the pia – the best proxy here for meningeal cells),
  while deeper meninges fall back to parenchyma levels of elongation. Their Pu.1⁺ share varies a lot
  between sections (median ~25 %, up to 70–85 % where the meninges are swollen with infiltrate), so
  "deeper = infiltrate" holds only in the swollen sections. Single random windows mix the layers, which
  is why many do *not* beat the parenchyma – the tally shows that heterogeneity, it does not contradict
  the section-level result.
* Cell identity still needs a marker (fibroblast / leptomeningeal); this panel can only say *where* the
  elongated nuclei are.
* What this cannot establish: cell identity. The compartment mixes meningeal cells and infiltrate; a
  fibroblast / leptomeningeal marker would be needed to say *which* cells are elongated.
""")]


def nb_swelling(run_dir):
    return [md("""
# 11 · Meningeal swelling along the surface

Each section's outer surface is traced and cut into **100 µm segments**. Every surface-meninges nucleus
(≤ 80 µm deep; flaps / roots excluded) is assigned to its nearest surface point, giving per segment:

* **cellularity** – meningeal nuclei per 100 µm of surface (the main swelling measure: a swollen
  meninges has more cells stacked over the same stretch of surface);
* **Pu.1⁺ per 100 µm** – the myeloid part of it;
* **thickness** – 90th percentile depth of its meningeal nuclei. Secondary: the meninges boundary always
  includes the outer 10 µm and only grows into very compact tissue, so thickness has a floor at ~10 µm
  and responds less than cellularity;
* **distance to the nearest automatic lesion** from the segment midpoint.

A segment counts as **swollen** when its cellularity exceeds the 95th percentile of control-section
segments (what an uninflamed surface looks like). Statistics use the section as the unit (replicate slides
averaged).

**Caveat – manual cores cap the meninges.** The meninges boundary never enters a manual lesion core
(notebook 05). Where an annotated core reaches the surface there are no meningeal nuclei by construction,
so swelling next to annotated lesions is *under*-estimated (e.g. R1_3_L below).
"""), code(SETUP.format(run_dir=run_dir) + '''
from scipy.stats import mannwhitneyu, wilcoxon
from lesionseg import expression as E, morphology as M, swelling as S
seg = pd.concat([S.surface_segments(r) for r in runs], ignore_index=True)
seg["cls"] = S.segment_class(seg)
THR = float(np.percentile(seg.loc[~seg.lesion_section, "cells_per_100um"], 95))
seg["swollen"] = seg.cells_per_100um > THR
print(f"{len(seg):,} segments of 100 µm in {seg.groupby(['scene', 'section']).ngroups} slide × sections; "
      f"swollen = more than {THR:.1f} meningeal nuclei per 100 µm (control p95)")
CLS = ["control section", "far from lesion", "near lesion"]
CCOL = {"control section": "#898781", "far from lesion": "#3987e5", "near lesion": "#e66767"}
summ = seg.groupby("cls").agg(segments=("segment", "size"), cells_per_100um=("cells_per_100um", "median"),
                              pu1_per_100um=("pu1_per_100um", "median"), thickness_um=("thickness_um", "median"),
                              swollen_pct=("swollen", lambda v: 100 * v.mean())).reindex(CLS)
summ.round(2)
'''), md("## Cellularity against distance to the nearest lesion (lesion sections; median per slide)"), code('''
les = seg[seg.lesion_section].copy()
edges = [0, 50, 100, 200, 400, 800, 1600]
les["db"] = pd.cut(les.dist_to_lesion_um.clip(lower=0), edges, include_lowest=True)
g = les.groupby(["scene", "db"], observed=True).cells_per_100um.median().reset_index()
g["mid"] = g.db.map(lambda iv: iv.mid).astype(float)
ctrl = seg[~seg.lesion_section].cells_per_100um.median()
fig, axes = plt.subplots(1, 2, figsize=(16, 4.6))
for (scene, d), col in zip(g.groupby("scene"), E.SCENE_COLORS):
    axes[0].plot(d.mid, d.cells_per_100um, color=col, lw=2, marker="o", ms=4, label=scene)
axes[0].axhline(ctrl, color="#898781", lw=1, ls=":"); axes[0].text(edges[-1], ctrl, " control sections", va="bottom", ha="right", fontsize=8, color="#52514e")
axes[0].set_xscale("symlog", linthresh=100); axes[0].set_xlabel("distance from segment to nearest lesion (µm)")
axes[0].set_ylabel("meningeal nuclei per 100 µm"); axes[0].set_title("cellularity falls with distance from lesions", loc="left")
axes[0].legend(fontsize=8, frameon=False)
sw = seg.groupby("cls").swollen.mean().reindex(CLS) * 100
axes[1].barh(CLS, sw, color=[CCOL[c] for c in CLS], height=0.6)
for i, v in enumerate(sw):
    axes[1].text(v + 0.5, i, f"{v:.0f} %", va="center", fontsize=9)
axes[1].set_xlabel("swollen segments (%)"); axes[1].set_title("share of swollen surface", loc="left"); axes[1].invert_yaxis()
plt.tight_layout(); plt.show()
'''), md("## Per section: near vs far from lesions (paired), lesion vs control sections"), code('''
fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))
tests = []
for ax, v, lab in zip(axes, ["cells_per_100um", "pu1_per_100um", "thickness_um"],
                      ["meningeal nuclei / 100 µm", "Pu.1⁺ meningeal nuclei / 100 µm", "thickness (µm)"]):
    p = S.paired_near_far(seg, v)
    for _, r in p.iterrows():
        ax.plot([0, 1], [r["far from lesion"], r["near lesion"]], color="#c3c2b7", lw=1)
    ax.scatter(np.zeros(len(p)), p["far from lesion"], color=CCOL["far from lesion"], s=40, zorder=3)
    ax.scatter(np.ones(len(p)), p["near lesion"], color=CCOL["near lesion"], s=40, zorder=3)
    sec = seg.groupby(["scene", "section", "lesion_section"])[v].median().groupby(["section", "lesion_section"]).mean().reset_index()
    cv = sec[~sec.lesion_section][v]
    ax.scatter(np.full(len(cv), 2) + np.linspace(-0.1, 0.1, len(cv)), cv, color=CCOL["control section"], s=40, zorder=3)
    ax.set_xticks([0, 1, 2], ["far from lesion", "near lesion", "control sections"]); ax.set_ylabel(lab)
    ax.set_title(lab, loc="left", fontsize=10)
    lv = sec[sec.lesion_section][v]
    tests.append({"measure": lab, "sections_paired": len(p), "near_higher": int((p.delta > 0).sum()),
                  "median_near_minus_far": p.delta.median(), "wilcoxon_p_near_far": wilcoxon(p.delta).pvalue if len(p) >= 6 else np.nan,
                  "lesion_sections_median": lv.median(), "control_sections_median": cv.median(),
                  "mannwhitney_p_lesion_vs_control": mannwhitneyu(lv, cv).pvalue})
plt.tight_layout(); plt.show()
pd.DataFrame(tests).round(4)
'''), md("## Per section, animal and level"), code('''
st = S.section_table(seg.assign(cells_per_100um=seg.cells_per_100um), THR)
sw_sec = seg.groupby(["scene", "section"]).swollen.mean().mul(100).rename("swollen_cellularity_pct")
st = st.merge(sw_sec.reset_index(), on=["scene", "section"]).drop(columns="swollen_pct")
display(st.round(2))
display(st.groupby(["animal"])[["cells_per_100um", "pu1_per_100um", "swollen_cellularity_pct"]].median().round(2))
display(st.groupby(["level", "lesion_section"])[["cells_per_100um", "pu1_per_100um", "swollen_cellularity_pct"]].median().round(2))
'''), md("## Maps: surface segments coloured by cellularity (light = more meningeal nuclei)"), code('''
vmax = float(np.nanpercentile(seg.cells_per_100um, 98))
for sec in ["R1_2_T", "P2_3_T", "P3_1_C", "OS1_2_C"]:
    hits = [r for r in runs if sec in set(seg[seg.scene == r.name].section)]
    if not hits:
        continue
    fig, axes = plt.subplots(1, len(hits), figsize=(9 * len(hits), 9), squeeze=False)
    for ax, r in zip(axes.ravel(), hits):
        sc = S.plot_section_swelling(ax, r, seg, sec, "cells_per_100um", vmax=vmax)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, label="meningeal nuclei per 100 µm")
    plt.show()
'''), md("""
## The most swollen spots (150 µm zoom-ins)

The six segments with the highest cellularity, at most one per section and slide. Meningeal nuclei at full
strength (filled by elongation class as in notebook 09), everything else faded; meninges / buffer edges
drawn.
"""), code('''
top = seg.sort_values("cells_per_100um", ascending=False).drop_duplicates(["scene", "section"]).head(6)
byname = {r.name: r for r in runs}
fig, axes = plt.subplots(2, 3, figsize=(20, 13.5))
for ax, row in zip(axes.ravel(), top.itertuples()):
    r = byname[row.scene]
    comp = M.window_compartments(r.cells, E.compartment(r.cells))
    st_ = M.plot_nuclei(ax, r, r.cells, row.x_um, row.y_um, 150, comp=comp, focus="meninges")
    ax.set_title(f"{row.section} · {row.scene.replace(' scene ', ' s')} · {row.cells_per_100um:.0f} nuclei/100 µm · "
                 f"{100 * st_['pu1_focus']:.0f} % Pu.1⁺ · {row.dist_to_lesion_um:.0f} µm to lesion", loc="left", fontsize=9)
M.class_legend(axes.ravel()[-1])
plt.tight_layout(); plt.show()
display(top[["scene", "section", "cells_per_100um", "pu1_per_100um", "thickness_um", "dist_to_lesion_um"]].round(1))
'''), md("""
## Reading

* Meningeal cellularity is a usable swelling measure: it rises toward lesions, is about twice as high in
  lesion sections as in control sections, and a much larger share of the near-lesion surface is swollen.
* With 8 lesion sections the paired near-vs-far test has little power; the consistency over sections and
  slides is the stronger evidence. Sections where a manual core reaches the surface (R1_3_L) show *less*
  meninges near the lesion – that is the protection rule, not biology.
* By animal, swelling separates disease from control: the adjuvant-only (CFA_L2) and OS1_2 animals have
  almost no swollen surface, the EAE animals 6–18 % – including EAE sections without lesions of their own
  (e.g. R1_2_L), so meningeal swelling is not confined to lesion sections.
* Thickness responds less because of how the boundary is built (10 µm floor, very strict growth); a
  dedicated meninges marker or annotation (option 1 earlier) would make thickness itself measurable.
* The traced surface follows tears in the section (e.g. CML_metal scene 1, R1_2_T); segments along a tear
  have little meninges and slightly dilute the far-from-lesion values.
""")]


def nb_orientation(run_dir):
    return [md("""
# 12 · Do the elongated meningeal nuclei lie along the surface?

Notebooks 09–10 showed that the outer ≤ 10 µm of the meninges is rich in elongated, mostly Pu.1⁻ nuclei.
If these are flattened pial / arachnoid cells, their long axis should run **parallel to the tissue
surface**; squashed or cut nuclei at the edge, or cells migrating inward, would point any way or radially.

* **Long axis** of every nucleus from the second moments of its mask in the Cellpose label image (the
  same ellipse as the cell table's eccentricity – they agree exactly).
* **Surface direction** from the fine tissue raster (gradient of the smoothed depth, σ = 5 µm).
* **Surface angle** 0–90°: 0 = along the surface, 90 = pointing into the tissue. Random axes give a
  uniform angle (mean 45°, 1/3 within 30° of the surface).
* **Order parameter S** = mean cos(2 · angle): +1 all parallel, 0 random, −1 all radial.

Only **elongated nuclei** (eccentricity ≥ 0.85) are analysed – a round nucleus has no meaningful axis.
Groups by depth: meninges 0–10 µm (the outer layer), meninges 10–80 µm, parenchyma 20–80 µm under the
surface (**edge control**: same surface, no meninges) and parenchyma 150–300 µm (deep). The section is
the unit (replicate slides averaged); a **shuffle null** pairs each nucleus with the surface direction of
a random other nucleus of the same slide (keeps any slide-wide preferred direction, e.g. from cutting).
"""), code(SETUP.format(run_dir=run_dir) + '''
from matplotlib.colors import LinearSegmentedColormap
from lesionseg import expression as E, morphology as M, orientation as O
ex = E.expression_table(runs)
df = O.orientation_table(runs, ex)
df["group"] = O.depth_groups(df)
el = df[(df.eccentricity >= M.ELONG_MIN) & df.group.notna()]
GCOL = dict(zip(O.GROUPS, ["#eb6834", "#eda100", "#3987e5", "#898781"]))
rows = []
for g in O.GROUPS:
    a = el[el.group == g]
    nul = O.shuffled_null(a, 200)
    rows.append({"group": g, "elongated_nuclei": len(a), "S": O.order_parameter(a.surface_angle),
                 "within_30deg_pct": 100 * (a.surface_angle < 30).mean(), "mean_angle_deg": a.surface_angle.mean(),
                 "shuffle_S_low": np.percentile(nul, 2.5), "shuffle_S_high": np.percentile(nul, 97.5)})
summ = pd.DataFrame(rows)
summ.round(3)
'''), md("## Angle to the surface, elongated nuclei (dotted = random)"), code('''
fig, axes = plt.subplots(1, 4, figsize=(18, 3.8), sharey=True)
bins = np.arange(0, 91, 10)
for ax, g in zip(axes, O.GROUPS):
    a = el[el.group == g].surface_angle
    ax.hist(a, bins=bins, weights=np.full(len(a), 100 / max(len(a), 1)), color=GCOL[g], edgecolor="white")
    ax.axhline(100 / 9, color="#52514e", lw=1, ls=":")
    ax.set_title(f"{g}\\nS = {O.order_parameter(a):.2f} · {len(a):,} nuclei", loc="left", fontsize=10)
    ax.set_xlabel("angle to surface (°)  0 = parallel"); ax.set_xticks([0, 30, 60, 90])
axes[0].set_ylabel("nuclei (%)")
plt.tight_layout(); plt.show()
'''), md("## Alignment against depth below the surface (elongated nuclei, any compartment; per slide)"), code('''
dp = O.depth_alignment(df)
fig, ax = plt.subplots(figsize=(10, 4.5))
for (scene, d), col in zip(dp.groupby("scene"), E.SCENE_COLORS):
    ax.plot(d.mid_um, d["mean"], color=col, lw=2, marker="o", ms=4, label=scene)
ax.axhline(0, color="#898781", lw=0.8, ls=":"); ax.axvspan(0, 10, color="#c3c2b7", alpha=0.3, lw=0)
ax.set_xscale("symlog", linthresh=50)
ax.set_xlabel("depth below the surface (µm)"); ax.set_ylabel("order parameter S (+1 parallel, −1 radial)")
ax.set_title("alignment with the surface fades within ~100 µm; deeper, slightly radial", loc="left"); ax.legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
'''), md("## Per section (replicate slides averaged)"), code('''
per = O.section_alignment(df)
t = per.pivot_table(index=["section", "animal"], columns="group", values="S").reindex(columns=O.GROUPS)
fig, ax = plt.subplots(figsize=(10, 4.8))
x = np.arange(len(O.GROUPS))
for _, row in t.iterrows():
    ax.plot(x, row.to_numpy(float), color="#c3c2b7", lw=1)
for i, g in enumerate(O.GROUPS):
    v = t[g].dropna()
    ax.scatter(np.full(len(v), i), v, s=36, color=GCOL[g], edgecolors="white", linewidths=0.8, zorder=3)
    ax.plot([i - 0.25, i + 0.25], [v.median()] * 2, color="#0b0b0b", lw=2)
ax.axhline(0, color="#898781", lw=0.8, ls=":")
ax.set_xticks(x, O.GROUPS); ax.set_ylabel("order parameter S"); ax.set_title("one line per section", loc="left")
plt.tight_layout(); plt.show()
display(pd.DataFrame([O.paired_test(per, O.GROUPS[0], b) for b in O.GROUPS[1:]]).round(4))
display(t.round(2))
'''), md("""
## Robustness: outer meninges vs the edge control under other choices

Each row recomputes S of the outer meningeal layer (0–10 µm) and of parenchyma 20–80 µm under the
surface, per section, and tests the paired difference.
"""), code('''
def check(d, ecc=M.ELONG_MIN, **kw):
    p = O.section_alignment(d, ecc_min=ecc, **kw)
    r = O.paired_test(p, O.GROUPS[0], O.GROUPS[2])
    s = p.groupby("group").S.median()
    return {"S_outer_meninges": s.get(O.GROUPS[0]), "S_parenchyma_20_80": s.get(O.GROUPS[2]),
            **{k: r[k] for k in ("sections", "median_delta", "ci95_low", "ci95_high", "sections_higher", "wilcoxon_p")}}
rows = [("default (ecc ≥ 0.85, σ = 5 µm)", check(df))]
rows += [(f"eccentricity ≥ {e}", check(df, ecc=e)) for e in (0.8, 0.9)]
for s in (2, 10, 20):
    rows.append((f"surface smoothing σ = {s} µm", check(O.orientation_table(runs, ex, sigma_um=s).assign(
        group=lambda d: O.depth_groups(d)))))
rows += [("well-segmented only (solidity ≥ 0.9)", check(df[df.solidity >= 0.9])),
         ("Pu.1⁻ nuclei only", check(df[~df.pu1_pos])), ("Pu.1⁺ nuclei only", check(df[df.pu1_pos], min_cells=5)),
         ("lesion sections only", check(df[df.section.isin(ex[ex.compartment.isin(["core", "rim"])].section.unique())])),
         ("control sections only", check(df[~df.section.isin(ex[ex.compartment.isin(["core", "rim"])].section.unique())]))]
rob = pd.DataFrame([{"check": n, **r} for n, r in rows])
rob.round(4)
'''), md("""
## Random examples (outer meninges, 140 µm windows, seed 1)

Window centres are random nuclei of the outer meningeal layer (≤ 10 µm deep), kept if the window holds
≥ 8 of them and lies ≥ 300 µm from earlier draws – nothing selected on orientation.
Every elongated nucleus is drawn as a short line along its long axis, coloured by its angle to the
surface: **green = along the surface, pink = pointing into the tissue**, grey = in between. Meninges /
buffer edges as before.
"""), code('''
cmap = LinearSegmentedColormap.from_list("angle", ["#1baf7a", "#c3c2b7", "#e87ba4"])
angs = {r.name: O.cell_surface_angles(r) for r in runs}
SIZE = 140
picks = []
for r in runs:
    outer = (r.cells.surface_tier.astype(str) == "meninges") & (r.cells.depth_um <= 10)
    comp = pd.Series(np.where(outer, "outer", None), index=r.cells.index)
    picks += [(r, w) for w in M.random_windows(r.cells, comp, "outer", 3, size_um=SIZE, min_target=8, seed=1)]
fig, axes = plt.subplots(3, 4, figsize=(22, 17.5), layout="constrained")
for ax, (r, (x, y)) in zip(axes.ravel(), picks):
    rgb, ext = R.read_crop_um(r, x, y, SIZE, scale=1.0)
    if rgb is not None:
        ax.imshow(np.clip(rgb * 0.8, 0, 1), extent=ext)
    view = (x - SIZE / 2, x + SIZE / 2, y - SIZE / 2, y + SIZE / 2)
    R.draw_surface(ax, r, view, lw=1.2)
    lc = O.draw_axes(ax, r.cells, angs[r.name], view, cmap=cmap)
    a = angs[r.name].merge(r.cells[["label", "x_um", "y_um", "surface_tier", "depth_um"]], on="label")
    a = a[a.x_um.between(*view[:2]) & a.y_um.between(*view[2:]) & (a.ecc_mom >= M.ELONG_MIN)
          & (a.surface_tier == "meninges") & (a.depth_um <= 10)]
    ax.set_xlim(view[0], view[1]); ax.set_ylim(view[3], view[2]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{r.name.replace(' scene ', ' s')} · outer layer: {len(a)} elong., S = {O.order_parameter(a.surface_angle):.2f}",
                 loc="left", fontsize=9)
for ax in axes.ravel()[len(picks):]:
    ax.axis("off")
fig.colorbar(lc, ax=axes.ravel().tolist(), shrink=0.5, label="angle to surface (°)")
plt.show()
'''), md("""
## Reading

* **Yes – the elongated nuclei of the outer meninges lie along the surface.** In the outer 10 µm,
  S = 0.61 and ~74 % of elongated nuclei are within 30° of the surface (random: 33 %); the shuffle null
  is ≈ 0, so this is the *local* surface, not a slide-wide cutting or scan direction. Higher than the
  section's own parenchyma 20–80 µm under the surface in **17 of 17 sections** (6 animals; median ΔS
  +0.39, 95 % hierarchical-bootstrap CI ~0.37–0.57), and under every check: eccentricity cut-off, surface
  smoothing, well-segmented nuclei, Pu.1⁻ and Pu.1⁺ separately, lesion and control sections.
* **It is a surface property that fades with depth.** Deeper meninges (10–80 µm) are less aligned
  (S ≈ 0.4, lower than the outer layer in 10/12 sections), parenchyma right under the surface only weakly
  (S ≈ 0.2), and from ~100 µm on the alignment is gone or slightly *radial* (S ≈ −0.13 at 150–300 µm,
  beyond its shuffle null) – plausibly radial glial / astrocytic septa of the white matter; not followed up here.
* **Fits flattened pial / arachnoid cells.** Together with notebooks 09–10 (thin, Pu.1⁻-rich, elongated
  outer layer), the orientation is what lining cells stretched along the pia would look like. The
  Pu.1⁺ elongated nuclei of the outer layer are aligned just as strongly – flattened macrophages along the
  surface (meningeal / perivascular macrophages are a candidate) rather than cells migrating inward.
* **Where it fails in the examples:** narrow strands and flaps, where "the surface" lies on both sides and
  its direction is ill-defined, and swollen infiltrate, whose nuclei are round anyway.
* Still no identity marker; orientation is evidence on shape and position only.
""")]


def nb_meninges_weak(run_dir):
    return [md("""
# 13 · Can a model find the meninges without hand annotations?

The meninges are placed by a rule (outer 10 µm of the surface plus compact infiltrate connected to it).
Until the collaborator annotates the boxes in `results/meninges_annotation/`, the only labels are the
rule's own. This notebook asks how far those take a model – and where they stop.

* **Confident labels only.** 1 = meninges in the outer ≤ 10 µm (meninges by any definition), 0 =
  parenchyma ≥ 150 µm deep, outside manual lesion cores. Everything in between is left out of training,
  so what a model says there is its own opinion.
* **No position.** The models never see depth or distance to the edge. Feature sets: nuclear **shape**
  (eccentricity, area, solidity, alignment of neighbours), **markers** (Pu.1, Iba1, nuclear intensity),
  **image** (texture and background of the 1.3 µm overview around the cell, normalised to the tissue) and
  **density** (cells per mm² of tissue around the cell).
* **Held out by animal.** Each animal is predicted by a model trained on the other five.

Three tests, as AUC (0.5 = chance, 1 = perfect):

| test | positives | negatives | what it asks |
|---|---|---|---|
| vs deep | outer meninges | deep parenchyma | sanity check |
| vs **virtual cut** | outer meninges | cells ≤ 10 µm from an artificial edge | did the model learn "edge" instead of meninges? |
| vs **core at surface** | outer meninges | manual lesion-core cells ≤ 30 µm deep | meninges vs subpial lesion – decides what is collected |

The **virtual cut**: in deep parenchyma (≥ 300 µm from any surface), all cells on one side of a random line
are deleted and every feature is recomputed as if the line were a surface (25 cuts per slide). There are no
meninges there, so a model that calls these cells meninges is an edge detector.

Computed by `scripts/meninges_weak_labels.py` → `results/meninges_weak/` (this notebook only reads it).
"""), code(SETUP.format(run_dir=run_dir) + '''
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score
OUT = Path("results/meninges_weak")
ev = pd.read_csv(OUT / "evaluation.csv")
pr = pd.read_parquet(OUT / "cell_predictions.parquet")
SETS = list(ev.features)
SCOL = dict(zip(SETS, ["#c3c2b7", "#898781", "#eb6834", "#3987e5", "#9085e9"]))
BEST, EDGE = "shape + markers + image", "shape + markers + density"
outer = pr.y == 1
core_s = pr.core & (pr.depth <= 30)
print(f"{len(pr):,} cells in sections · confident: {outer.sum():,} outer meninges, {(pr.y == 0).sum():,} deep parenchyma"
      f" · {core_s.sum():,} manual-core cells ≤ 30 µm deep · {pr.animal.nunique()} animals")
ev.set_index("features").round(3)
'''), md("## The three tests per feature set"), code('''
tests = [("outer_vs_deep_auc", "vs deep parenchyma"), ("outer_vs_cut_edge_auc", "vs virtual-cut edge"),
         ("outer_vs_core_at_surface_auc", "vs lesion core at the surface")]
fig, axes = plt.subplots(1, 4, figsize=(19, 3.6), sharey=True, gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
yy = np.arange(len(SETS))[::-1]
for ax, (col, title) in zip(axes, tests):
    ax.barh(yy, ev[col] - 0.5, left=0.5, color=[SCOL[s] for s in SETS], height=0.6)
    for y, v in zip(yy, ev[col]):
        ax.text(max(v, 0.5) + 0.01, y, f"{v:.2f}", va="center", fontsize=9)
    ax.axvline(0.5, color="#52514e", lw=1, ls=":"); ax.set_xlim(0.3, 1.08); ax.set_title(title, loc="left", fontsize=10)
    ax.set_xlabel("AUC (0.5 = chance)")
ax = axes[3]
w = 0.27
for i, (col, lab, c) in enumerate([("outer_called_pct", "outer meninges", "#eb6834"),
                                  ("cut_edge_called_pct", "virtual-cut edge", "#52514e"),
                                  ("core_at_surface_called_pct", "core at surface", "#e87ba4")]):
    ax.barh(yy + (1 - i) * w, ev[col], height=w, color=c, label=lab)
ax.set_xlim(0, 100); ax.set_xlabel("% called meninges (p ≥ 0.5)")
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("how often each group is called meninges", loc="left", fontsize=10)
axes[0].set_yticks(yy, SETS)
plt.tight_layout(); plt.show()
'''), md("""
## Virtual cut: p(meninges) against distance to an artificial edge

Mean held-out probability of cells near the 25 virtual cuts per slide, by distance to the cut line.
Dotted: mean p of real outer meninges for the same model. A model that learned meninges stays flat and low;
an edge detector rises towards the cut.
"""), code('''
cp = pd.read_csv(OUT / "virtual_cut_profile.csv")
cp["left"] = cp.db.str.extract(r"\\(([-\\d.]+),")[0].astype(float).clip(lower=0)
cp["right"] = cp.db.str.extract(r",\\s*([\\d.]+)\\]")[0].astype(float)
cp["mid"] = (cp.left + cp.right) / 2
fig, ax = plt.subplots(figsize=(10, 4.3))
for s in SETS:
    d = cp[cp.features == s].sort_values("mid")
    ax.plot(d.mid, d["mean"], color=SCOL[s], lw=2.2, marker="o", ms=4, label=s)
    ax.axhline(pr.loc[outer, f"p_{s}"].mean(), color=SCOL[s], lw=1, ls=":")
ax.set_xlabel("distance to the virtual cut (µm)"); ax.set_ylabel("mean p(meninges)"); ax.set_ylim(0, 1)
ax.set_title("density models light up at any edge; the position-free models barely notice it", loc="left")
ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(); plt.show()
display(cp.pivot_table(index="mid", columns="features", values="mean").reindex(columns=SETS).round(2))
'''), md("""
## p(meninges) against depth below the real surface

Held-out probability of every cell in the sections (also the unlabelled ones between 10 and 150 µm),
by depth and lesion context. Manual lesion core = inside the collaborator's CORE polygons (any section).
"""), code('''
GROUPS = {"control section": "#898781", "far from lesion": "#3987e5", "≤ 100 µm from lesion": "#eda100",
          "manual lesion core": "#e87ba4"}
dbins = [0, 10, 20, 30, 40, 60, 80, 120, 160, 250, 400]
pr["group"] = np.where(pr.core, "manual lesion core", pr.ctx)
pr["db"] = pd.cut(pr.depth, dbins, include_lowest=True)
mid = {iv: (iv.left + iv.right) / 2 for iv in pr.db.cat.categories}
fig, axes = plt.subplots(1, 2, figsize=(17, 4.3), sharey=True)
for ax, s in zip(axes, [BEST, EDGE]):
    t = pr.groupby(["group", "db"], observed=True)[f"p_{s}"].agg(["mean", "size"]).reset_index()
    for g, col in GROUPS.items():
        d = t[(t.group == g) & (t["size"] >= 20)]
        ax.plot(d.db.map(mid).astype(float), d["mean"], color=col, lw=2.2, marker="o", ms=4, label=g)
    ax.axvspan(0, 10, color="#c3c2b7", alpha=0.3, lw=0); ax.axvspan(150, 400, color="#c3c2b7", alpha=0.15, lw=0)
    ax.set_xscale("symlog", linthresh=40); ax.set_xlabel("depth below the surface (µm)")
    ax.set_title(s, loc="left", fontsize=10)
axes[0].set_ylabel("mean p(meninges)"); axes[0].set_ylim(0, 1); axes[0].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
print("grey bands: the confident training labels (outer ≤ 10 µm = 1, ≥ 150 µm = 0)")
'''), md("## Per held-out animal (position-free model)"), code('''
def per_animal(s):
    rows = []
    for a, d in pr.groupby("animal"):
        o, conf, cs = d.y == 1, d.y >= 0, d.core & (d.depth <= 30)
        p = d[f"p_{s}"]
        row = {"animal": a, "outer_cells": int(o.sum()), "core_at_surface_cells": int(cs.sum()),
               "outer_vs_deep_auc": roc_auc_score(d.y[conf], p[conf]) if d.y[conf].nunique() == 2 else np.nan,
               "outer_called_pct": 100 * (p[o] >= 0.5).mean()}
        if cs.sum() >= 10:
            row["outer_vs_core_auc"] = roc_auc_score(np.r_[np.ones(o.sum()), np.zeros(cs.sum())], np.r_[p[o], p[cs]])
            row["core_called_pct"] = 100 * (p[cs] >= 0.5).mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("animal")
per_animal(BEST).round(3)
'''), md("""
## Examples: the annotation boxes where a lesion core meets the surface

The six `core_at_surface` boxes from `results/meninges_annotation/` (300 µm). Left: raw image with the
rule's meninges edge (solid) and buffer (dashed) and the **manual CORE** (dashed, collaborator). Middle and
right: every nucleus as a dot coloured by its held-out p(meninges), position-free model and density model.
"""), code('''
cmap = LinearSegmentedColormap.from_list("p", ["#3987e5", "#e9e8e2", "#eb6834"])
tg = pd.read_csv("results/meninges_annotation/targets.csv")
byname = {r.name: r for r in runs}
SIZE = 300

def box_row(axes, t):
    r = byname[t.scene]
    view = (t.x_um - SIZE / 2, t.x_um + SIZE / 2, t.y_um - SIZE / 2, t.y_um + SIZE / 2)
    rgb, ext = R.read_crop_um(r, t.x_um, t.y_um, SIZE)
    d = pr[(pr.scene == t.scene) & pr.x_um.between(*view[:2]) & pr.y_um.between(*view[2:])]
    for j, ax in enumerate(axes):
        if rgb is not None:
            ax.imshow(np.clip(rgb * (0.8 if j == 0 else 0.35), 0, 1), extent=ext)
        R.draw_surface(ax, r, view, lw=1.2)
        R.draw_outlines(ax, r, lesion=False, core=False, manual=True, lw=1.4)
        if j:
            s = [BEST, EDGE][j - 1]
            sc = ax.scatter(d.x_um, d.y_um, c=d[f"p_{s}"], cmap=cmap, vmin=0, vmax=1, s=16, lw=0)
            ax.set_title(f"{s}", loc="left", fontsize=9)
        else:
            ax.set_title(f"{t.target_id} · {t.kind} · {t.why}", loc="left", fontsize=9)
        ax.set_xlim(view[0], view[1]); ax.set_ylim(view[3], view[2]); ax.set_xticks([]); ax.set_yticks([])
    return sc

def gallery(rows):
    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 5.3 * len(rows)), layout="constrained", squeeze=False)
    for ax3, t in zip(axes, rows.itertuples()):
        sc = box_row(ax3, t)
    fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.3, label="p(meninges), held out by animal")
    plt.show()

gallery(tg[tg.kind == "core_at_surface"])
'''), md("""
## Examples: the other kinds of box (one each)

`swollen` (most cellular meninges of the section), `internal` (meninges-like tissue > 80 µm deep), `bare`
(surface without meningeal nuclei) and `typical`.
"""), code('''
gallery(tg[tg.kind != "core_at_surface"].drop_duplicates("kind"))
'''), md("""
## Reading

* **Without annotations, the best honest model recognises meninges but cannot separate them from lesion
  at the surface.** Shape + markers + image separates outer meninges from deep parenchyma well (AUC 0.96)
  and is not fooled by an artificial edge (0.91; only ~20 % of virtual-cut edge cells called meninges).
  Against manual lesion-core cells at the surface it is at chance (0.50) and calls most of them meninges:
  subpial lesion infiltrate looks like meningeal infiltrate to it, as it does to the rule.
* **What it picks up is "near the surface or in infiltrate", not meninges as such.** In the depth plot the
  position-free model still gives p ≈ 0.6–0.7 to parenchyma 30–60 µm under the surface of *control* sections,
  and ≈ 0.5 to lesion-core cells 100–200 µm deep. At the cores (examples below) it paints the subpial part
  of the core as meninges, continuous with the real lining; per held-out animal its AUC against these cells is
  0.45–0.62 in all four animals that have them.
* **Density features are edge detectors.** Their near-perfect scores (0.998) are a leak: at a virtual cut
  they call ~96 % of cells meninges, even with density counted per tissue area, because the tissue outline
  is drawn around the cells themselves. Their better score against the cores (0.74) is the same leak –
  core cells sit a little deeper than the outer layer. They are marked position-dependent in
  `lesionseg/meninges_model.py` and should not be used.
* **The deciding labels are the hand annotations at the cores.** The confident labels contain no example of
  "lesion at the surface", so no model trained on them can learn that border. The six `core_at_surface`
  boxes above are the ones to annotate first if the collaborator's time is short.
* Caveat: "manual lesion core ≤ 30 µm deep" is the only independent negative at the surface, and the CORE
  polygons were drawn for lesions, not for the meningeal border – some of those cells may be meninges.
""")]


def nb_xenium_meninges(run_dir):
    return [md("""
# 14 · Where are the meninges? A gene-based answer from the Xenium EAE atlas

Our images (nuclei, Pu.1, Iba1) cannot separate meninges from subpial lesion (notebook 13). The RRMAP2 Xenium 5k
atlas of EAE spinal cord can: **157 sections**, CFA controls and every EAE phase, 5,000 genes and the raw
transcripts. Two gene layers mark the border:

* **meningeal fibroblast genes** – Slc47a1, Foxc2, Crabp2, Osr1, Slc13a4, Smoc2, Wnt6, Klf5, Fmod, Cdh1: chosen
  from the atlas as the genes detected in meningeal fibroblasts but rarely in any other cell (table below);
  Ptgds and the collagens are *not* specific here (Ptgds spreads into many cells; collagens are shared with the
  lesion fibroblasts);
* **glia limitans genes** – Aqp4 and Gfap: the astrocyte endfeet on the parenchymal side of the pia.

**Method** (`lesionseg/xenium_meninges.py`; `scripts/xenium_transcript_rasters.py` on the remote;
`scripts/xenium_meninges.py`):

1. All transcripts of a capture region binned at 5 µm. **Tissue** = transcript density above half the section's
   median (the edge at half maximum; a looser mask sat 15–20 µm outside the cells). **Depth** = distance to the
   tissue edge – the same definition the image pipeline uses on its own tissue mask.
2. Every tissue pixel is assigned to its nearest edge point; edge points are grouped into **50 µm stretches**.
3. Per stretch, 5 µm depth rings (3 pooled), the **pia** placed two ways with opposite biases:
   * **balance** – the run of rings from the surface where the meningeal share of transcripts beats the
     glia-limitans share (each scaled between the section's surface and deep levels) and is ≥ 10 % of the way to
     its surface level. Immune transcripts dilute the meningeal *share* in swollen meninges → leans "subpial".
   * **glia** – the run of rings from the surface where the glia-limitans *density* is below half its deep level
     (no astrocyte endfeet yet). Astrocyte loss in lesions → leans "meningeal".

   Stretches that never reach parenchyma (flaps, tears, oblique cuts) get no pia and are left out, as are false
   edges: tissue whose nearest outside pixel lies in a patch with *no* transcripts at all (a missing field of
   view – a real edge always has a halo of diffused transcripts).
   A stretch whose pia is at 0 µm has **no resolved meningeal layer** – mostly meninges thinner than one 5 µm
   ring (a single cell), sometimes stripped. A third view, **present**, asks the clean question only on
   stretches that do have a layer: where meninges exist, which side of them is the infiltrate on?
4. Atlas cells (curated cell types) are placed **above** (meninges) or **below** (parenchyma) the pia of their
   stretch.

Unit for comparisons: the section. Atlas cells ≠ our cells – the depth scales of the two datasets are not
calibrated against each other (both measure from a tissue edge, but from different masks).
"""), code(SETUP.format(run_dir=run_dir) + '''
from lesionseg import xenium_meninges as X
OUT, WORK = Path("results/xenium_meninges"), Path("outputs/xenium_meninges")
sh = pd.read_csv("../ImmunoTransformer/docs/sample_sheet/sections.csv")
sh["group"] = np.where(sh.condition == "CONTROL", "CFA control", sh.phase)
prof_g = pd.read_csv(OUT / "depth_profiles.csv")
ab = pd.read_csv(OUT / "above_pia_by_depth.csv")
abs_ = pd.read_csv(OUT / "immune_above_pia_by_section.csv")
aba = pd.read_csv(OUT / "immune_above_pia_by_animal.csv")
pia = pd.read_parquet(WORK / "pia.parquet")
cov = pd.read_csv(OUT / "section_coverage.csv")
PH = [p for p in X.PHASES if p in set(sh.group)]
EAE = [g for g in PH if g != "CFA control"]
PCOL = dict(zip(PH, ["#898781", "#c3c2b7", "#eda100", "#e87ba4", "#3987e5", "#9085e9", "#199e70", "#0b6e4f"]))
BINS = ["(0, 10]", "(10, 20]", "(20, 30]", "(30, 50]", "(50, 80]", "(80, 150]"]
MID = dict(zip(BINS, [5, 15, 25, 40, 65, 115]))
print(f"{sh.section.nunique()} sections in the sheet · {pia.section.nunique()} with a pia · {len(pia):,} stretches")
print("share of a section's edge stretches with a pia (median, 5th pct):", cov.stretches_with_pia.median().round(2),
      cov.stretches_with_pia.quantile(.05).round(2))
sh.groupby("group").agg(sections=("section", "size"), animals=("animal", "nunique")).reindex(PH)
'''), md("## Which genes mark the meningeal fibroblasts (detection rate: fraction of cells with ≥ 1 transcript)"),
        code('''
det = pd.read_csv(OUT / "fb_gene_detection.csv", index_col=0)
chosen = ["Slc47a1", "Foxc2", "Crabp2", "Osr1", "Slc13a4", "Smoc2", "Wnt6", "Klf5", "Fmod", "Cdh1"]
other = ["Ptgds", "Col1a1", "Col1a2", "Lum", "Igf2", "Aqp4", "Gfap", "Ptprc", "Cd74", "Mrc1"]
print("first 10 rows: the meningeal group; then for comparison")
det.loc[chosen + other].round(2)
'''), md("""
## Gene layers against depth (all sections of a phase pooled; per 1000 transcripts)

The meningeal genes peak in the outer 5–15 µm; the glia limitans dips there and rises from ~20–30 µm; immune
transcripts in EAE are highest at 15–40 µm – right where the two layers meet.
"""), code('''
fig, axes = plt.subplots(1, 4, figsize=(21, 4.2), sharex=True)
for ax, k, t in zip(axes, ["meningeal", "glia_limitans", "immune", "collagen"],
                    ["meningeal FB genes", "glia limitans (Aqp4, Gfap)", "immune genes", "collagens (Col1a1/2, Lum, Pdgfra)"]):
    for g in PH:
        d = prof_g[(prof_g.group == g) & (prof_g.depth <= 150)]
        ax.plot(d.depth, d[f"{k}_per_1000"], color=PCOL[g], lw=2, label=g)
    ax.set_title(t, loc="left", fontsize=10); ax.set_xlabel("depth below the tissue edge (µm)")
axes[0].set_ylabel("per 1000 transcripts"); axes[-1].legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
plt.tight_layout(); plt.show()
'''), md("""
## Examples: gene layers and the detected pia

Red = glia limitans, green = meningeal fibroblast genes, blue = immune genes. **White** = pia (balance),
**yellow dashed** = pia (glia); tissue between edge (grey) and pia is meninges by the genes. One section of each of
our five animals' Xenium twins (see the end), plus a control.
"""), code('''
cells_atlas = pd.read_parquet(WORK / "atlas_cells.parquet").rename(columns={"meta_sample_id": "section"})
TWIN = {a: f"RR_{a}" for a in ["OS1_2", "P2_3", "P3_1", "R1_2", "R1_3"]}
ex = (sh[sh.animal.isin(TWIN.values())].sort_values(["animal", "spinal_level"]).groupby("animal").head(1).section.tolist()
      + sh[sh.group == "CFA control"].section.iloc[[3]].tolist())
fig, axes = plt.subplots(2, 3, figsize=(21, 14))
for ax, s in zip(axes.ravel(), ex):
    r = sh[sh.section == s].iloc[0]
    c = cells_atlas[cells_atlas.section == s]
    rgb, men, tis, ext = X.section_view("outputs/xenium_rasters", r, c, pia)
    men_g = X.section_view("outputs/xenium_rasters", r, c, pia, col="pia_glia_um")[1]
    ax.imshow(rgb, extent=ext)
    ax.contour(men, levels=[0.5], colors="white", linewidths=0.8, extent=ext, origin="upper")
    ax.contour(men_g, levels=[0.5], colors="#eda100", linewidths=0.8, linestyles="--", extent=ext, origin="upper")
    ax.contour(tis, levels=[0.5], colors="#898781", linewidths=0.5, extent=ext, origin="upper")
    ax.set_title(f"{s} · {r.animal} · {r.group} · level {r.spinal_level}", loc="left", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
plt.tight_layout(); plt.show()
'''), md("## How deep is the pia? (per 50 µm stretch)"), code('''
fig, axes = plt.subplots(1, 2, figsize=(17, 4.2))
ax = axes[0]
for g in PH:
    for col, ls in (("pia_depth_um", "-"), ("pia_glia_um", "--")):
        v = np.sort(pia[pia.group == g][col])
        ax.plot(v, 1 - np.arange(len(v)) / len(v), color=PCOL[g], lw=1.8, ls=ls, label=g if ls == "-" else None)
ax.set_xlim(0, 60); ax.set_xlabel("pia depth (µm)"); ax.set_ylabel("share of stretches with a deeper pia")
ax.legend(fontsize=8, frameon=False); ax.set_title("solid = balance, dashed = glia: meninges are thin", loc="left", fontsize=10)
ax = axes[1]
e = pia[pia.group != "CFA control"].copy()
e["load"] = pd.qcut(e.immune_frac_0_80.rank(method="first"), 5, labels=["1 low", "2", "3", "4", "5 high"])
for col, ls in (("pia_depth_um", "-"), ("pia_glia_um", "--")):
    t = e.groupby("load", observed=True)[col].describe(percentiles=[.5, .75, .9])
    for q, c in (("50%", "#0b0b0b"), ("75%", "#898781"), ("90%", "#c3c2b7")):
        ax.plot(t.index.astype(str), t[q], marker="o", color=c, ls=ls, label=f"{q[:-1]}th pct" if ls == "-" else None)
ax.set_xlabel("immune load of the stretch (quintile, EAE)"); ax.set_ylabel("pia depth (µm)"); ax.legend(fontsize=8, frameon=False)
ax.set_title("infiltrated stretches have thicker meninges", loc="left", fontsize=10)
plt.tight_layout(); plt.show()
summ = pd.concat({c: pia.groupby("group")[c].describe(percentiles=[.5, .75, .9])[["count", "mean", "50%", "75%", "90%"]]
                  for c in ("pia_depth_um", "pia_glia_um")}, axis=1)
summ[("", "no meningeal layer (balance)")] = pia.groupby("group").pia_depth_um.apply(lambda v: (v == 0).mean())
summ.reindex(PH).round(2)
'''), md("""
## Are immune cells near the surface meningeal or subpial?

Share of atlas cells **above** the pia of their stretch, by depth; solid = balance, dashed = glia; middle panel
= only stretches that have a meningeal layer (present). Meningeal
fibroblasts are the method's control: ~87 % of them at 0–10 µm land above the pia (5 µm rings and cell
positions set a floor of ~13 % misplacement).
"""), code('''
fig, axes = plt.subplots(1, 3, figsize=(24, 4.4), sharey=True)
for ax, (cls, dfns, title) in zip(axes, [("immune", (("balance", "-"), ("glia", "--")), "immune cells, all stretches"),
                                         ("immune", (("present", "-"),), "immune cells, stretches with a meningeal layer"),
                                         ("meningeal FB", (("balance", "-"), ("glia", "--")), "meningeal FB cells (control)")]):
    for g in PH:
        for dfn, ls in dfns:
            d = ab[(ab.definition == dfn) & (ab.cell_class == cls) & (ab.group == g) & ab.depth_bin.isin(BINS)]
            d = d.assign(x=d.depth_bin.map(MID)).sort_values("x")
            d = d[d.cells >= 30]
            ax.plot(d.x, 100 * d.above_pia, color=PCOL[g], lw=1.8, ls=ls, marker="o", ms=3,
                    label=g if ls == "-" else None)
    ax.axhline(50, color="#52514e", lw=0.8, ls=":"); ax.set_xscale("symlog", linthresh=30)
    ax.set_xlabel("depth of the cell (µm)"); ax.set_title(title, loc="left", fontsize=10)
axes[0].set_ylabel("% above the pia (= in the meninges)"); axes[2].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
t = (ab[(ab.cell_class == "immune") & ab.depth_bin.isin(BINS)]
     .pivot_table(index=["definition", "depth_bin"], columns="group", values="above_pia"))
(100 * t.reindex(columns=PH)).reindex(BINS, level=1).round(0)
'''), md("""
### Sensitivity: the balance definition's minimum meningeal level

The fibroblast control hardly moves with the threshold, so it cannot pick it; the immune shares at 0–20 µm do
move (the default is 0.1). The conclusion below 20–30 µm does not depend on it.
"""), code('''
sens = pd.read_csv(OUT / "sensitivity_min_meningeal.csv")
(100 * sens.set_index(["cells", "min_meningeal"])[BINS]).round(0)
'''), md("## Per section (immune cells, balance definition; sections with ≥ 20 immune cells in the bin)"), code('''
fig, ax = plt.subplots(figsize=(14, 4.5))
b_ = abs_[abs_.definition == "balance"]
for i, b in enumerate(BINS[:5]):
    for j, g in enumerate(EAE):
        v = b_[(b_.depth_bin == b) & (b_.group == g) & (b_.cells >= 20)].above_pia * 100
        x = i + (j - len(EAE) / 2) * 0.11
        ax.scatter(np.full(len(v), x), v, s=10, color=PCOL[g], alpha=0.7, label=g if i == 0 else None)
        if len(v):
            ax.plot([x - 0.04, x + 0.04], [v.median()] * 2, color="#0b0b0b", lw=1.5)
ax.axhline(50, color="#52514e", lw=0.8, ls=":")
ax.set_xticks(range(5), [f"{b} µm" for b in BINS[:5]]); ax.set_ylabel("% immune cells above the pia")
ax.legend(fontsize=8, frameon=False, ncol=4); ax.set_title("each dot a section, black = median", loc="left")
plt.tight_layout(); plt.show()
'''), md("""
## Zoomed examples at the surface (300 µm windows)

Background: gene layers as above (dimmed). **White line** = pia (balance), yellow dashed = pia (glia), grey =
tissue edge. Every atlas cell is a dot: **green** meningeal fibroblast, **blue** immune (myeloid, T, B, DC –
not microglia), **orange** astrocyte, **pink** microglia, grey other. **Filled** = above the pia (meninges),
**hollow** = below (parenchyma); small grey × = outside the tissue mask or on a stretch without a pia. Windows are centred on a 50 µm surface stretch picked by the criterion in the
title – four per kind, from different sections.
"""), code('''
cdepth = pd.read_parquet(WORK / "cells_depth.parquet").merge(pia[["section", "seg", "pia_depth_um"]], on=["section", "seg"], how="left")
cdepth["above"] = cdepth.depth <= cdepth.pia_depth_um
KIND = [(cdepth.Anno_L2.isin(X.MENINGEAL_FB), "#1baf7a", "meningeal FB"),
        (cdepth.Anno_L1_curated.isin(X.IMMUNE_L1), "#3987e5", "immune"),
        (cdepth.Anno_L1_curated == "Astrocyte", "#eda100", "astrocyte"),
        (cdepth.Anno_L2 == "Microglia", "#e87ba4", "microglia")]
cdepth["col"] = np.select([k[0] for k in KIND], [k[1] for k in KIND], "#898781")
_views = {}

def view(s):
    if s not in _views:
        r = sh[sh.section == s].iloc[0]
        c = cells_atlas[cells_atlas.section == s]
        rgb, men, tis, ext = X.section_view("outputs/xenium_rasters", r, c, pia)
        _views[s] = (r, rgb, men, X.section_view("outputs/xenium_rasters", r, c, pia, col="pia_glia_um")[1], tis, ext)
    return _views[s]

def zoom(ax, s, seg, size=300, note=""):
    r, rgb, men, men_g, tis, ext = view(s)
    cx, cy = (seg % 100000 + 0.5) * X.SEG_UM, (seg // 100000 + 0.5) * X.SEG_UM
    ax.imshow(np.clip(rgb * 0.55, 0, 1), extent=ext)
    ax.contour(men, levels=[0.5], colors="white", linewidths=1.4, extent=ext, origin="upper")
    ax.contour(men_g, levels=[0.5], colors="#eda100", linewidths=1, linestyles="--", extent=ext, origin="upper")
    ax.contour(tis, levels=[0.5], colors="#898781", linewidths=0.8, extent=ext, origin="upper")
    d = cdepth[(cdepth.section == s) & cdepth.x_centroid.between(cx - size / 2, cx + size / 2)
               & cdepth.y_centroid.between(cy - size / 2, cy + size / 2)]
    out_ = d.depth.isna() | d.pia_depth_um.isna()
    ax.scatter(d.x_centroid[out_], d.y_centroid[out_], s=10, marker="x", c="#52514e", lw=0.6)
    d = d[~out_]
    up = d.above.astype(bool)
    ax.scatter(d.x_centroid[up], d.y_centroid[up], s=16, c=d.col[up], lw=0)
    ax.scatter(d.x_centroid[~up], d.y_centroid[~up], s=16, facecolors="none", edgecolors=d.col[~up], lw=0.9)
    ax.set_xlim(cx - size / 2, cx + size / 2); ax.set_ylim(cy + size / 2, cy - size / 2)
    ax.set_xticks([]); ax.set_yticks([])
    p = pia[(pia.section == s) & (pia.seg == seg)].iloc[0]
    imm = d[d.col == "#3987e5"]
    ax.set_title(f"{s} · {r.animal} · {r.group}\\npia {p.pia_depth_um:.0f} µm (glia {p.pia_glia_um:.0f}) · immune above/below: "
                 f"{int(imm.above.fillna(False).sum())}/{int((~imm.above.fillna(False).astype(bool)).sum())} {note}",
                 loc="left", fontsize=8.5)

def gallery(picks, ncol=4):
    picks = list(picks)
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 5.6 * nrow), squeeze=False)
    for ax, (s, seg) in zip(axes.ravel(), picks):
        zoom(ax, s, seg)
    for ax in axes.ravel()[len(picks):]:
        ax.axis("off")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ls="", color=c, label=l) for _, c, l in KIND] + [
        Line2D([], [], marker="o", ls="", color="#898781", label="other"),
        Line2D([], [], marker="o", ls="", mfc="none", mec="#52514e", label="hollow = below the pia"),
        Line2D([], [], color="white", lw=2, label="pia (balance)"), Line2D([], [], color="#eda100", ls="--", label="pia (glia)")]
    fig.legend(handles=h, loc="lower center", ncol=8, fontsize=9, frameon=False, facecolor="#52514e")
    plt.tight_layout(rect=(0, 0.03, 1, 1)); plt.show()

def pick(df, n=4, seed=None, by=None):
    df = df.sample(frac=1, random_state=seed) if seed is not None else df.sort_values(by, ascending=False)
    return list(df.drop_duplicates("section").head(n)[["section", "seg"]].itertuples(index=False, name=None))
pia["imm_above_dens"] = pia.immune_above_pia / pia.total
pia["imm_below_dens"] = pia.immune_below_pia_0_50 / pia.total
'''), md("### Normal meninges – CFA controls, random stretches with a pia"), code('''
gallery(pick(pia[pia.group == "CFA control"], seed=1))
'''), md("### Swollen meninges: immune infiltrate **above** the pia (most immune transcripts above, pia ≥ 20 µm)"),
code('''
gallery(pick(pia[(pia.group != "CFA control") & (pia.pia_depth_um >= 20)], by="imm_above_dens"))
'''), md("### Subpial infiltrate: immune infiltrate **below** an existing pia (layer 5–15 µm, most immune transcripts below)"),
code('''
gallery(pick(pia[(pia.group != "CFA control") & pia.pia_depth_um.between(5, 15)], by="imm_below_dens"))
'''), md("""### No resolved meningeal layer (pia at 0 µm), infiltrated EAE stretches – random

What the "all stretches" numbers count as subpial although no meningeal layer is visible: thin meninges, lost
meninges or infiltrate right at the edge."""), code('''
e0 = pia[(pia.group != "CFA control") & (pia.pia_depth_um == 0)]
gallery(pick(e0[e0.immune_frac_0_80 > e0.immune_frac_0_80.median()], seed=3))
'''), md("### Unselected: random infiltrated EAE stretches (immune load above the EAE median)"), code('''
e = pia[pia.group != "CFA control"]
gallery(pick(e[e.immune_frac_0_80 > e.immune_frac_0_80.median()], n=8, seed=2))
'''), md("### Our five animals' Xenium twins: the most infiltrated surface stretch of each section"), code('''
tw = pia[pia.section.isin(sh[sh.animal.isin(TWIN.values())].section)].copy()
tw["imm"] = tw.immune_above_pia + tw.immune_below_pia_0_50
picks = list(tw.sort_values("imm", ascending=False).drop_duplicates("section")
             .merge(sh[["section", "spinal_level"]], on="section").sort_values(["animal", "spinal_level"])
             [["section", "seg"]].itertuples(index=False, name=None))
gallery(picks, ncol=3)
'''), md("""
## What it means for our rule

Our rule calls the outer 10 µm meninges and extends the meninges into compact infiltrate connected to it, down to
80 µm. Below: how deep our rule's Pu.1⁺ meninges cells are, and – applying the Xenium share of immune cells
above the pia in the **same animal** – how many of them would be subpial. *present* (stretches with a layer)
is the lower bound, *balance* / *glia* over all stretches the upper.

**Assumption to confirm:** our animals OS1_2, P2_3, P3_1, R1_2 and R1_3 are the RRMAP2 animals RR_OS1_2,
RR_P2_3, RR_P3_1, RR_R1_2 and RR_R1_3 (same names, same C/T/L levels). CFA_L2 has no match and uses the pooled
CFA controls. Xenium sections of an animal are pooled over levels (few immune cells per bin otherwise); bins
with < 20 Xenium immune cells fall back to the animal's phase.
"""), code('''
rows = []
for r in runs:
    m = r.cells[(r.cells.section_id > 0) & (r.cells.surface_tier.astype(str) == "meninges") & r.cells.pu1_pos]
    m = m.assign(animal=m.section_name.astype(str).str.rsplit("_", n=1).str[0],
                 depth_bin=pd.cut(m.depth_um, [0, 10, 20, 30, 50, 80, 150], labels=BINS, include_lowest=True).astype(str))
    rows.append(m.groupby(["animal", "depth_bin"]).size().rename("pu1_meninges_cells").reset_index())
ours_t = pd.concat(rows).groupby(["animal", "depth_bin"], as_index=False).pu1_meninges_cells.sum()
ours_t = ours_t[ours_t.depth_bin.isin(BINS)]
grp_of = sh.drop_duplicates("animal").set_index("animal").group
phase_rate = (pd.read_csv(OUT / "above_pia_by_depth.csv").query("cell_class == 'immune'")
              .set_index(["definition", "group", "depth_bin"]).above_pia)
est = []
for dfn in ("present", "balance", "glia"):
    a = aba[(aba.definition == dfn)].set_index(["animal", "depth_bin"])
    t = ours_t.copy()
    t["xenium_animal"] = t.animal.map(TWIN)
    t["grp"] = t.xenium_animal.map(grp_of).fillna("CFA control")

    def rate(row):
        k = (row.xenium_animal, row.depth_bin)
        if isinstance(row.xenium_animal, str) and k in a.index and a.loc[k, "cells"] >= 20:
            return a.loc[k, "above_pia"]
        return phase_rate.get((dfn, row.grp, row.depth_bin), np.nan)
    t["xenium_above"] = t.apply(rate, axis=1)
    t["expected_subpial"] = t.pu1_meninges_cells * (1 - t.xenium_above)
    est.append(t.assign(definition=dfn))
est = pd.concat(est)
tab = est.pivot_table(index="depth_bin", columns="definition", values="expected_subpial", aggfunc="sum").reindex(BINS)
tab.insert(0, "pu1_meninges_cells", est[est.definition == "balance"].groupby("depth_bin").pu1_meninges_cells.sum())
display(tab.round(0))
tot = tab.pu1_meninges_cells.sum()
for dfn in ("present", "balance", "glia"):
    print(f"{dfn:8s}: of {tot:,.0f} Pu.1+ cells our rule calls meninges, expected subpial {tab[dfn].sum():,.0f} "
          f"({100 * tab[dfn].sum() / tot:.0f} %); deeper than 20 µm: {tab.loc[BINS[2:], dfn].sum():,.0f} "
          f"of {tab.loc[BINS[2:], 'pu1_meninges_cells'].sum():,.0f}")
display((100 * est.pivot_table(index=["definition", "animal"], columns="depth_bin", values="xenium_above")
         .reindex(columns=BINS)).round(0))
'''), md("""
## Reading

* **The meninges are thin.** In ~40–55 % of EAE surface stretches and ~70 % in CFA controls no meningeal layer
  is resolved at 5 µm (a single-cell pia–arachnoid, sometimes stripped). Where there is one, it is 5–30 µm deep
  and thicker where the surface is infiltrated.
* **Where meninges exist, the outer 10 µm is meningeal and deeper infiltrate is subpial.** On stretches with a
  meningeal layer, immune cells in EAE are ~89 % above the pia at 0–10 µm, ~47 % at 10–20 µm, ~20 % at
  20–30 µm, ~9 % at 30–50 µm and ~4 % at 50–80 µm. Over all stretches (counting edges without a layer as
  subpial) the shares are lower, but every definition agrees beyond ~20 µm: **most immune cells there are below
  the pia.**
* **For our rule:** the outer ≤ 10 µm layer is fine where meninges exist; the extension of the meninges into
  compact infiltrate down to 80 µm takes in mostly subpial lesion. A cap at ~20 µm would match the genes.
  Expected subpial share of our rule's Pu.1⁺ meninges cells: ~39 % (present, lower bound) to ~57–68 % (all
  stretches) overall, but **~78–89 % of those deeper than 20 µm** (1,630–1,860 of 2,090 cells).
* **The truly ambiguous case** is a dense immune band at the surface with no glia limitans and only faint
  fibroblast genes (see the subpial gallery): *balance* puts the pia at the edge, *glia* 25–30 µm deep. That is
  what the six `core_at_surface` annotation boxes are for.
* **It explains notebook 13:** meninges and lesion interleave at 10–30 µm, so a model trained on "outer ≤ 10 µm =
  meninges" learns infiltrate, and no image feature separates them.
* **Which pia to trust:** by the fibroblast control, *balance* (≈ 87 % of meningeal fibroblasts at 0–10 µm above
  its pia) beats *glia* (60–75 % in EAE, ~40 % in controls); glia is the bracket with the opposite bias.
* **Caveats.** Different cells, segmentation and tissue mask than our images: the depth scales are not
  calibrated (treat µm values as ±5–10 µm). The pia is resolved in 5 µm rings. ~30 % of stretches (tears,
  flaps, oblique cuts) have none; false edges at missing fields of view are removed, but low-density patches
  inside sections still count as edge. The Xenium sections of "our" animals are probably not the same sections
  (assumption above).
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
             "08_expression_by_animal_and_cell_type": nb_expression_split, "09_nuclear_elongation": nb_elongation,
             "10_meninges_elongation_reliability": nb_meninges_reliability, "11_meningeal_swelling": nb_swelling,
             "12_nuclear_orientation": nb_orientation,
             "13_meninges_model_without_annotations": nb_meninges_weak,
             "14_xenium_meninges": nb_xenium_meninges}
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
