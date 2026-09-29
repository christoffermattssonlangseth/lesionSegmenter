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
| **distal** | remaining tissue (normal-appearing tissue / control) | > 150 µm |

Each cell additionally gets `dist_to_lesion_um` (negative inside) and an optional `dist_bin`
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

### 1. Segmentation masks (primary)

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

### 2. Raw image (fallback / quick look)

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
| `zone_summary.csv`, `per_lesion_zone_counts.csv` | cell / Pu.1⁺ counts per zone (and per lesion) |
| `distance_profile.csv/.png` | Pu.1⁺ counts & fraction vs signed distance to lesion edge |
| `maps/*.tif` + `maps.json` | grid layers: `nuclei_density`, `pu1_density`, `pu1_fraction`, `z_*`, `lesion_score`, `lesion_mask`, `lesion_labels`, `zones`, `signed_distance_um`, `dense_nonmyeloid`, `tissue` |
| `zones_px.geojson` / `zones_um.geojson` | lesion, core, rim, peri and dense_nonmyeloid polygons (QuPath-ready, classified) |
| `cells_pu1_px.geojson` | Pu.1⁺ cell outlines as QuPath detections classified by zone |
| `overview.png`, `pu1_histogram.png` | QC figures |
| `run_log.json` | all parameters, thresholds and counts |

## DVP / LMD hand-off

`export-lmd` writes a Leica LMD XML with [py-lmd](https://github.com/MannLabs/py-lmd), one well
per zone. Calibration points must be the three physical marks in the **same scene-pixel frame**
as the masks (see `x_px, y_px`).

```bash
lesionseg export-lmd outputs/CML_1/scene0/cells.parquet outputs/CML_1/scene0/CML_1_lmd.xml \
    --calibration "1200,900,22800,950,1250,26500" \
    --wells "core=A1,rim=A2,peri=A3,distal=A4" \
    --pixel-size-um 0.3250972 --dilate-um 1.0
```

Or from Python (`lesionseg.export.export_lmd`) with `only=`, `group_col="dist_bin"` etc. for
finer distance stratification.

## Tuning notes

* `density.sigma_um` (40) sets how smooth the lesion outline is; `density.bin_um` (10) the grid.
* Lesion `threshold` accepts `{type: zscore|otsu|quantile|absolute, value}`. Use `absolute`
  (cells/mm²) once a Pu.1⁺ density cut-off has been calibrated across samples.
* Rim/peri widths: 50 / 150 µm defaults; `assign.distance_bins_um` gives finer bins.
* The tissue mask for the density grid is derived from cell positions (`tissue.source: cells`),
  so sparse white matter is not lost; `image` or `both` are alternatives.

## Layout

```
lesionseg/    io.py (CZI/TIFF readers)  masks.py (mask route)  segment.py (image route)
              features.py (Pu.1 calls)  density.py  lesion.py (score, gate, zones)  assign.py
              export.py (parquet/GeoJSON/TIFF/LMD)  viz.py  pipeline.py  cli.py  config.py
configs/      eae_dvp.yaml (masks + images), quicklook_image.yaml
tests/        synthetic end-to-end tests for both routes + LMD export
data/raw ->   ../DVP/data/OneDrive_1_9-18-2026 (symlink, not committed)
```

## Open questions

* Which secondary carries Pu.1 – AF647 or AF555? (`channels.pu1`, only matters for intensities/figures)
* Mask format/coordinate frame from the segmentation pipeline (full scene vs. cropped region → `offset_px`).
* Calibration-mark coordinates for the LMD export.
* Do we want a white/grey-matter annotation as an extra gate, or is the Pu.1 fraction gate enough?
