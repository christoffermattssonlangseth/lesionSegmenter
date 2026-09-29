# lesionSegmenter

Outline **EAE lesions** in whole-slide scans from two things we can measure reliably:
the density of segmented nuclei and the density of **Pu.1⁺ myeloid cells**. Every lesion is
split into a **core**, a **rim** and a **peri-lesion** band, and every Pu.1⁺ cell is given a
zone plus a signed distance to the lesion edge. The cell outlines are then exported for
**Deep Visual Proteomics** (laser microdissection, LMD) so that myeloid metabolic signatures
can be stratified by lesion proximity.

```
inputs                          lesionseg                                         outputs
──────────────────────────────  ────────────────────────────────────────────────  ─────────────────────────────
cells label mask  (all cells)   ┐ cell table (centroid, area, contour)             cells.parquet / .csv
Pu.1⁺ label mask                ┘ Pu.1 match ───► Pu.1⁺ flag                       zone_summary.csv
[image .czi – optional]           intensities per cell, overview figure            lesions.csv
                                  density grid (10 µm bins, σ 40 µm)               dense_nonmyeloid_regions.csv
                                  ├ nuclei / mm²                                   maps/*.tif  (grid layers)
                                  ├ Pu.1⁺ / mm²   ─► lesion score (robust z)       zones_px.geojson (QuPath)
                                  └ Pu.1⁺ fraction ─► myeloid gate                 cells_pu1_px.geojson (QuPath)
                                  lesion mask ─► core | rim | peri | distal        overview.png, distance_profile.*
                                  per-cell zone + signed distance                  LMD XML via py-lmd
```

## Zone definitions

All distances are measured from the lesion boundary in µm and are configurable.

| zone | definition | default |
|---|---|---|
| **core** | inside lesion, deeper than `rim_width_um` from the edge (or, `core_method: score`, bins above `core_threshold`) | > 50 µm inside |
| **rim** | inside lesion, within `rim_width_um` of the edge | 0–50 µm inside |
| **peri** | outside lesion, within `peri_width_um` of the edge | 0–150 µm outside |
| **deep** | outside the peri band, a further `deep_width_um` outward (the "additional step" captured for DVP) | 150–300 µm |
| **distal** | everything farther out, and all tissue of lesion-free control sections | > 300 µm |

Slides carry several cross-sections (nine per slide here). Sections come from the curated
`Sample_category` polygons when present, otherwise connected tissue pieces are labelled
automatically (`section_id`, `sections` layer in `maps/`). Every cell and lesion carries its
section, so infiltration can be located per section (`section_summary.csv`).

**Lesion vs control sections are decided from the data** (`sections.focus: auto`): a section is a
lesion section when the automatically detected lesions cover ≥ `lesion_min_frac` (2 %) of it and
≥ `lesion_min_area_mm2` (0.05). Control sections get no lesion context at all (all cells
`distal`, `dist_to_lesion_um = inf`) and only supply GM / WM control wells. The manual CORE
polygons are never used to define lesions or sections – they only feed the validation files and
the `has_manual_core` column, so the two views can be compared. Each cell additionally gets `dist_to_lesion_um` (negative inside) and an optional `dist_bin`
so zones can be re-binned later without re-running anything (`lesionseg run --from-cells`).

### What counts as a lesion – and what does not

The lesion **score is Pu.1⁺ density**, not nuclear density: a robust z-score of the smoothed
Pu.1⁺ cells/mm² relative to all tissue bins, thresholded at `lesion.threshold` (default z > 2.5).
Hypercellular regions that are *not* myeloid – **central canal ependyma, grey matter, meninges** –
are explicitly excluded by a **myeloid gate**:

* a bin needs `pu1_fraction ≥ min_pu1_fraction` (default 0.15) and `pu1_density ≥ min_pu1_density`;
* a connected lesion needs a mean Pu.1⁺ fraction ≥ `min_lesion_pu1_fraction` (default 0.2);
* regions with nuclear-density z ≥ `dense_nonmyeloid_z` that fail the gate are kept separately as
  **`dense_nonmyeloid`** (mask layer, `dense_nonmyeloid_regions.csv`, blue outlines in the figure
  and in `zones_px.geojson`) so they can be reviewed rather than silently dropped.

Alternative scores (`pu1_fraction`, `nuclei_density`, `combined` with weights) are available in
`lesion.score` for comparison.

## Install

```bash
mamba env create -f environment.yml
conda activate lesionseg
pip install -e ".[lmd,dev]"          # + ".[cellpose]" if you want to segment from images
pytest                                # synthetic end-to-end tests (~20 s)
```

## Input routes

### 1. SpatialData store + cell table (what the collaborator pipeline exports – primary)

`cellpose_output.zip` from the Cellpose/SpatialData pipeline unpacks to one `<slide>.zarr`
per scene plus `cell_table*.h5ad`. The pipeline reads

* `labels/<name>_masks_filtered/s0` – int label image, **label == `cell_id`**, scene-pixel frame;
* the AnnData table – centroids, area, channel means (`SytoG, Pu1, Iba1, Dapi`), `Pu1_class`
  (Pu.1 positivity), `Sample_category` (section = animal + spinal level), `manual_annotation`;
* `shapes/<name>_shapes` – exact Cellpose outlines (used as contours for Pu.1⁺ cells);
* `shapes/Sample_category` – curated section outlines (used as section identity);
* `shapes/CORE`, `NA_GW_WM`, `VBO` – manual annotations, **used for validation only**.

```yaml
# configs/eae_dvp_sdata.yaml (excerpt)
table: ../data/masks/cellpose_output/cell_table_full_annotated_v2.h5ad
samples:
  - name: CML_1
    image: ../data/raw/20260917_CML_1.czi                        # optional (figures)
    sdata: ../data/masks/cellpose_output/20260917_CML_1.zarr
    slide_name: 20260917_CML_1
```

```bash
lesionseg run configs/eae_dvp_sdata.yaml            # 4 scenes, ~20–30 s each on a laptop
```

### 2. Plain segmentation masks

Two label images in the **same full-resolution pixel frame** as the scan (and as the LMD
calibration marks): one with every cell/nucleus, one with only the Pu.1⁺ cells (e.g. from BIAS,
Cellpose or QuPath). Formats: `.tif` (any int dtype), 16-bit `.png`, `.npy`, Cellpose `*_seg.npy`.
Binary masks are automatically relabelled by connected components.

Pu.1⁺ objects are matched to cells by pixel overlap (`masks.min_overlap`, default 30 % of the Pu.1
object). Pu.1⁺ objects hitting no cell are **kept** as extra rows (`source = pu1_mask`, negative
`label`) so no myeloid cell is lost for DVP. Contours (WKT, µm) are extracted for Pu.1⁺ cells by
default (`masks.contours: pu1 | all | none`).

If the `.czi` is also given, per-cell mean intensity and local background are measured for every
channel (exact, tiled) and the overview figure uses the real image.

```yaml
# configs/eae_dvp.yaml (excerpt)
samples:
  - name: CML_1
    image: data/raw/20260917_CML_1.czi      # optional
    cells_mask: data/masks/CML_1_cells.tif
    pu1_mask:   data/masks/CML_1_pu1.tif
    pixel_size_um: 0.3250972                # required only when no image is given
    scenes: [0]
```

```bash
lesionseg run configs/eae_dvp.yaml
lesionseg run configs/eae_dvp.yaml --only CML_1 --from-cells   # re-tune lesion/zone params in seconds
```

### 3. Raw image (fallback / quick look)

Without masks, nuclei are segmented from the nuclear channel (`classical` watershed on CPU,
or `cellpose` / `stardist`) tile-by-tile straight from the CZI, and Pu.1⁺ is called from
background-corrected nuclear intensity (GMM / Otsu / quantile / absolute).
`configs/quicklook_image.yaml` does this at half resolution.

```bash
lesionseg info data/raw/20260917_CML_1.czi
lesionseg overview data/raw/20260917_CML_1.czi --out overview.png
lesionseg run configs/quicklook_image.yaml
```

## Outputs (per sample / scene, `outputs/<name>/scene<i>/`)

| file | content |
|---|---|
| `cells.parquet` / `.csv` | one row per cell: `x_px, y_px` (scene px), `x_um, y_um`, area, shape, `<ch>_mean/_bg`, `pu1_pos`, `zone`, `zone_code`, `lesion_id`, `dist_to_lesion_um`, `dist_bin`, `contour_wkt` (µm) |
| `lesions.csv` | per lesion: area, centroid, core/rim area, mean Pu.1⁺ density & fraction, score |
| `dense_nonmyeloid_regions.csv` | hypercellular Pu.1-poor regions (canal, grey matter) |
| `zone_summary.csv`, `per_lesion_zone_counts.csv` | cell / Pu.1⁺ counts per zone (and per section × lesion) |
| `section_summary.csv` | per tissue piece (e.g. each spinal-cord cross-section on the slide): area, cells, Pu.1⁺, lesion count / area / fraction, Pu.1⁺ per zone |
| `distance_profile.csv/.png` | Pu.1⁺ counts & fraction vs signed distance to lesion edge |
| `maps/*.tif` + `maps.json` | grid layers: `nuclei_density`, `pu1_density`, `pu1_fraction`, `z_*`, `lesion_score`, `lesion_mask`, `lesion_labels`, `zones`, `signed_distance_um`, `dense_nonmyeloid`, `tissue` |
| `zones_px.geojson` / `zones_um.geojson`, `zone_polygons.csv` | lesion, core, rim, peri, deep and dense_nonmyeloid polygons per lesion / section (QuPath-ready GeoJSON; WKT table in µm). `lesionseg export-zones-lmd` cuts these regions on the LMD |
| `cells_pu1_px.geojson` | Pu.1⁺ cell outlines as QuPath detections classified by zone |
| `wells.csv` | LMD well selection: per section × group (core, rim, rings, GM, WM) the selected cells, area, shortfall |
| `validation_manual_cores.csv`, `validation_cell_confusion.csv`, `manual_annotations_px.geojson` | agreement with the manual CORE / GM / WM annotations (SpatialData route) |
| `overview.png`, `pu1_histogram.png` | QC figures (green = manual CORE outlines, blue = dense non-myeloid) |
| `run_log.json` | all parameters, thresholds and counts |

## DVP / LMD hand-off

### Well selection (`lesionseg.wells`)

Reproduces the collaborator's protocol (`docs/collaborator/code/2_2_Annotation.ipynb`) on the
data-driven zones: per lesion section, a fixed **target area of Pu.1⁺ cells** (3000 µm²) is picked
for each group – `core`, `rim`, rings `0–100`, `100–200`, `200–400` µm outside the lesion edge –
and GM / WM control groups from the manual grey/white-matter polygons in any section. Cells within
100 µm of the section edge or inside VBO polygons are excluded, cells far from the group's median
size are dropped, and cells are taken in spatial order (`order: spatial | random | central`).
Rings measured from the *manual* core boundary are available via `manual_dist_um` filters for
direct comparison with the collaborator's wells. Result: `well_group` / `well_name` per cell and
`wells.csv`.

### LMD export

`export-lmd` writes a Leica LMD XML with [py-lmd](https://github.com/MannLabs/py-lmd); by default
one well per `well_name` with automatic plate positions (A1, A2, …). Calibration points must be the three physical marks in the **same scene-pixel frame**
as the masks (see `x_px, y_px`).

```bash
lesionseg export-lmd outputs/sdata/CML_1/scene0/cells.parquet outputs/sdata/CML_1/scene0/CML_1_lmd.xml \
    --calibration "1200,900,22800,950,1250,26500" \
    --pixel-size-um 0.3250972 --dilate-um 1.0          # add --group-col zone --wells "core=A1,..." for zone wells
```

Or from Python (`lesionseg.export.export_lmd`) with `only=`, `group_col="dist_bin"` etc. for
finer distance stratification.

## Validation on the 20260917 scans

Against the collaborator's hand-drawn lesion cores (never used by the pipeline):

| scene | manual cores | detected (>50 % inside auto lesion) | manual core area inside auto lesion |
|---|---|---|---|
| CML_1 | 34 | 97 % | 98 % |
| CML_2_rescan | 25 | 100 % | 95 % |
| CML_metal scene0 | 33 | 88 % | 93 % |
| CML_metal scene1 | 20 | 100 % | 100 % |

The automatic lesion (core + rim) is larger than the manual core by design; 98 % of cells the
annotator labelled `core` fall in the automatic core or rim, grey-matter cells stay distal.
The data-driven section classification also flags infiltrated sections without manual cores
(e.g. R1_3_C, R1_2_L, P3_1_L).

Compute: the SpatialData route needs no GPU and runs in 20–30 s per scene on a laptop; only
image-based Cellpose segmentation of full scans would justify the remote machine.

## Tuning notes

* `density.sigma_um` (40) sets how smooth the lesion outline is; `density.bin_um` (10) the grid.
* Lesion `threshold` accepts `{type: zscore|otsu|quantile|absolute, value}`. Use `absolute`
  (cells/mm²) once a Pu.1⁺ density cut-off has been calibrated across samples.
* Rim/peri widths: 50 / 150 µm defaults; `assign.distance_bins_um` gives finer bins.
* The tissue mask for the density grid is derived from cell positions (`tissue.source: cells`),
  so sparse white matter is not lost; `image` or `both` are alternatives.

## Layout

```
lesionseg/    io.py (CZI/TIFF readers)  sdata.py (SpatialData route)  masks.py (mask route)
              segment.py (image route)  features.py (Pu.1 calls)  density.py
              lesion.py (score, myeloid gate, zones)  assign.py (zones, sections)  wells.py (LMD wells)
              validate.py (vs manual annotations)  export.py (parquet/GeoJSON/TIFF/LMD)  viz.py
              pipeline.py  cli.py  config.py
configs/      eae_dvp_sdata.yaml (primary), eae_dvp.yaml (plain masks), quicklook_image.yaml
docs/collaborator/  the collaborator's annotation + well-selection notebook (reference)
data/masks/cellpose_output -> unpacked cellpose_output.zip (not committed)
tests/        synthetic end-to-end tests for both routes + LMD export
data/raw ->   ../DVP/data/OneDrive_1_9-18-2026 (symlink, not committed)
```

## Open questions

* Section threshold: 2 % lesion area calls CFA_L2_C (adjuvant-only control, CML_metal scene1) a
  lesion section at 2.2 % – raise `lesion_min_frac` or add a Pu.1⁺-fraction criterion?
* Calibration-mark coordinates for the LMD export.
* Scans 2023-1 and CML_2 have no Cellpose output yet.
