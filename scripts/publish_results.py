"""Copy the curated, small outputs of a run into the committed ``results/`` folder.

    python scripts/publish_results.py --run-dir outputs/sdata --out results

Figures (overview.png, distance_profile.png), per-scene tables (sections, lesions, wells, capture
sites, Pu.1 counts, validation) and the cohort tables are copied; parquet / TIFF / GeoJSON are not.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

KEEP = ["overview.png", "lesion_cells.png", "distance_profile.png", "section_summary.csv", "lesions.csv", "wells.csv",
        "capture_sites.csv", "zones_px.geojson", "zones_um.geojson", "zone_polygons.csv",
        "pu1_counts.csv", "zone_summary.csv", "per_lesion_zone_counts.csv", "validation_manual_cores.csv",
        "validation_cell_confusion.csv", "distance_profile.csv", "dense_nonmyeloid_regions.csv", "run_log.json"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    root, out = Path(a.run_dir), Path(a.out)
    n = 0
    for logf in sorted(root.glob("*/scene*/run_log.json")):
        d = logf.parent
        dest = out / d.parent.name / d.name
        dest.mkdir(parents=True, exist_ok=True)
        extra = [p.name for pat in ("section_cells_*.png", "zone_polygons*.png") for p in d.glob(pat)]
        for f in KEEP + extra:
            if (d / f).exists():
                shutil.copy2(d / f, dest / f)
                n += 1
    if (root / "cohort").exists():
        (out / "cohort").mkdir(parents=True, exist_ok=True)
        for f in (root / "cohort").glob("*.csv"):
            shutil.copy2(f, out / "cohort" / f.name)
            n += 1
    for f in Path("models").glob("training_report.csv"):
        shutil.copy2(f, out / "model_training_report.csv")
    for extra in ("lesion_threshold_grid.csv", "lesion_calibration_table.csv"):
        if (root.parent / extra).exists():
            shutil.copy2(root.parent / extra, out / extra)
    print(f"copied {n} files to {out}/")


if __name__ == "__main__":
    main()
