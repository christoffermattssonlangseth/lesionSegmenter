# Status – lesionSegmenter (handoff, 2026-10-15)

Read this first when resuming. `README.md` documents the pipeline; `important-info.md` holds the
reaction rules agreed with Ting. Last commit at handoff: `e6b8755` (main, pushed).

## Where things stand

The pipeline runs end to end on the 4 scenes (CML_1, CML_2_rescan, CML_metal scene 0/1), the mass-spec
reaction plan is regenerated, and notebooks 01–12 are executed (`notebooks/`, copies in
`results/notebooks/`, summary in `results/REPORT.md`).

### Decisions taken in this session (all in `configs/eae_dvp_sdata.yaml`)

| topic | decision |
|---|---|
| cells per reaction | `reactions.target_cells: 250` is a **target, not a cap**; `max_cells: null` = collect every eligible cell (set a number to cap) |
| VBO | always every cell inside the polygons (Pu.1⁺ and Pu.1⁻), ignores edge exclusion |
| meninges boundary | `parenchyma.method: adaptive` (`lesionseg/meninges.py`): outer 10 µm of the true surface + very compact surface-connected infiltrate (`compact_q: 99.9`, ≤ 80 µm) + flaps; 10 µm **buffer** inward that is never collected. `method: band` restores the old fixed 30 µm band |
| manual cores | the collaborator's CORE polygons are **protected**: never meninges or buffer; meninges cannot grow through them; buffer carved from the meningeal side (restriction from the user: meninges must never enter the manual lesion annotations). Check: 0 manual-core cells in meninges/buffer |
| edge exclusion | wells and reactions: **40 µm** (was 100 µm in the collaborator's protocol; 25 µm looked ragged at the surface) |
| reaction plan | 57 reactions (≤ 60), 36,988 cells; 22 shortfalls (15 are VBO, the rest R1_3_L / R1_3_T with few lesion cells) |

Every cell now carries `surface_tier` (meninges | buffer | parenchyma | outside) and `depth_um`;
`maps/surface_tiers.tif` is the ~1.3 µm raster.

### Notebooks

| nb | content |
|---|---|
| 01–04 | cohort overview, lesion judging, zones, manual vs automatic (unchanged in content, regenerated) |
| 05 | capture sites / wells + **meninges, buffer and edge exclusion** (manual-core check, edge-exclusion sweep, close-ups with legend) |
| 06 | collection section by section; meninges (grey) / buffer (white ring) markers, meninges / buffer / **edge-exclusion line** (dotted cyan), full legend; header counts this section's own cells |
| 07 | Pu.1 / Iba1 of Pu.1⁺ cells by compartment, paired per section, gradient, crowding check |
| 08 | same per slide × section / animal / level; morphological cell types (GMM) and per-cell vs composition decomposition |
| 09 | round vs elongated nuclei, zoom-ins with every nucleus outline (from the Cellpose label image) |
| 10 | is meningeal elongation reliable: section-level stats, hierarchical bootstrap, robustness checks, random galleries, layer split |
| 11 | meningeal swelling along the surface in 100 µm segments |
| 12 | orientation of nuclei vs the local surface (`lesionseg/orientation.py`; long axis from label-image moments, cached as `nucleus_orientation.parquet` per scene) |

## Findings so far (exploratory; unit = section, 8 lesion sections / 17 sections / 6 animals)

* **Pu.1 near lesions** (nb 07/08): Pu.1 per Pu.1⁺ cell vs same-section distal: rim 1.25×, peri 1.18×,
  core 1.15×, back to baseline ~250 µm out; Iba1 ~1.3×; nuclei ~10 % smaller. Rim > distal in 16/16
  slide × sections, every animal and level. Holds at matched crowding. Per-cell, not composition:
  morphological type mix barely shifts, per-cell part explains ~100 %; within every type Pu.1 rises at the rim.
* **Nuclear shape** (nb 09/10): lesion compartments ≈ distal. Meninges +13 pp elongated vs own parenchyma
  (95 % CI ~10–17), 16/17 sections, survives all checks incl. an edge-artefact control. It is the **thin
  outer layer**: outer ≤ 10 µm 31 % elongated (~80 % Pu.1⁻) vs 13 % at 30–80 µm (= parenchyma), 17/18
  sections. Fits flattened pial / arachnoid cells; swollen infiltrate is round. Random 120 µm windows are
  too small to show it individually.
* **Meningeal swelling** (nb 11): cellularity (meningeal nuclei / 100 µm surface) 7.5 near lesions vs 4.2
  far vs 3.2 in controls; 30 % of near-lesion surface swollen vs 10 % / 5 %; control animals (CFA_L2,
  OS1_2) 1–3 % swollen, EAE animals 6–18 % incl. lesion-free EAE sections. Thickness barely responds
  (10 µm floor by construction).
* **Orientation** (nb 12): elongated outer-meningeal nuclei lie along the surface: order parameter
  S = 0.61 (+1 parallel, 0 random), 74 % within 30° (random 33 %); > parenchyma 20–80 µm in 17/17 sections,
  shuffle null ≈ 0, holds for Pu.1⁻ and Pu.1⁺, all robustness checks. Fades by ~100 µm; deep parenchyma
  slightly radial (S ≈ −0.13). Supports flattened pial / arachnoid lining cells.
* **Correction made along the way:** the old 100 µm edge exclusion, not the 30 µm meninges band, was
  what kept subpial lesion out of the reactions; only ~50 collected cells were in the new meninges/buffer.

## Open items / next steps

1. **Ask the collaborator** whether 40 µm edge exclusion is fine for the LMD wells (her protocol: 100 µm).
2. **Meninges annotation – ready to send.** `results/meninges_annotation/` (8 sections, every animal,
   31 boxes of 300 µm: core_at_surface / swollen / internal / bare / typical; per-slide ROI GeoJSON, raw-image
   previews) + protocol `docs/meninges_annotation.md` (napari, GeoJSON back). When the files come back:
   `scripts/train_meninges_model.py --annot-dir <dir>` → rule vs model per box kind, held out by animal
   (`lesionseg/meninges_model.py`). Plumbing dry-run on rule-derived fake labels worked (~7k labelled
   cells). Feasibility (2026-10-23): shape alone predicts the *rule's* meninges only weakly (AUC 0.72);
   geometry dominates – real labels are the bottleneck. Open question for her: perivascular cuffs = meninges?
   **Without annotations** (`scripts/meninges_weak_labels.py` → `results/meninges_weak/`, notebook 13, 2026-10-23): train on
   confident labels only (outer ≤ 10 µm vs parenchyma ≥ 150 µm, cores left out), no position features, held out
   by animal. Position-free shape + markers + image: AUC 0.96 vs deep parenchyma, 0.91 vs a *virtual cut* edge
   (not an edge detector), but **0.50 vs manual lesion core at the surface** – it cannot tell meninges from subpial
   lesion, the case that matters for collection. Density features are edge detectors (96 % of virtual-cut cells
   called meninges) despite tissue normalisation. → The core_at_surface boxes are the decisive annotations.
   Also found: 348 manual-core cells (59–137 per slide) are per-cell parenchyma (rule holds) but the coarse
   grid `zone` layer still says meninges, so they get no core/rim zone; 36 of them (16 Pu.1⁺) are beyond the
   40 µm edge exclusion and lost from lesion reactions. Not fixed.
3. ~~Orientation analysis~~ done (nb 12).
4. Meninges as its own **reaction compartment** (discussed, not decided): budget is 57/60, needs pooling.
5. Cell identity needs markers (resident vs infiltrating myeloid: TMEM119/P2RY12 vs CCR2; meningeal
   fibroblasts: PDGFRβ / collagen I; CD31 for endothelium).
6. From README: section threshold for CFA_L2_C, LMD calibration marks, scans 2023-1 and CML_2 lack Cellpose output.

## How to regenerate (env `lesionseg`, ~15 min)

```bash
conda run -n lesionseg lesionseg run configs/eae_dvp_sdata.yaml                     # ~1–1.5 min / scene
conda run -n lesionseg python scripts/summarize_cohort.py --run-dir outputs/sdata --config configs/eae_dvp_sdata.yaml
conda run -n lesionseg python scripts/publish_results.py --run-dir outputs/sdata --out results
conda run -n lesionseg python scripts/make_report_notebooks.py          # all, or --only 05 06 …
cp notebooks/*.ipynb results/notebooks/
conda run -n lesionseg python scripts/make_report.py --run-dir outputs/sdata --out results/REPORT.md
conda run -n lesionseg pytest -q && conda run -n lesionseg ruff check lesionseg scripts tests
```

## Gotchas

* `summarize_cohort.py` **needs `--config`**; without it the reaction settings come from the old run log.
* Notebooks are **generated** by `scripts/make_report_notebooks.py` – edit the generator, not the .ipynb.
  Open notebooks in an editor show stale outputs until reloaded from disk.
* Cell table: `label` = mask value in the Cellpose label image; `cell_id` is just the row number.
* Pu.1⁺ contours (`contour_wkt`) exist only for Pu.1⁺ cells; outlines of all nuclei come from the label
  zarr (`run.log["cells"]["labels"]`), read crop by crop (`lesionseg/morphology.py`).
* Some image crops show blocky orange compression artefacts in the Pu.1 channel – not segmentation.
* `scripts/meninges_tiers_diagnostic.py` + `results/meninges_tiers/` are the exploration behind the
  adaptive meninges (no manual-core protection, looser `compact_q`); the pipeline module supersedes it.
* User preference in this project: commit and push after each finished change; regenerate notebooks
  whenever results change.
