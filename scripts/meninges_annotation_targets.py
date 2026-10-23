"""Pick sections and boxes for manual meninges annotation and write the material for the annotator.

    python scripts/meninges_annotation_targets.py --run-dir outputs/sdata --out results/meninges_annotation

Writes ``sections.csv``, ``targets.csv``, one ``<slide>_MENINGES_ROI.geojson`` per slide (boxes in scene px,
to load into napari as the ``MENINGES_ROI`` layer), a preview PNG per section (raw image only – the current
boundary is deliberately not shown) and ``README.md``. Protocol: ``docs/meninges_annotation.md``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from lesionseg import report as R  # noqa: E402
from lesionseg.meninges_targets import annotation_targets  # noqa: E402

KIND_COLORS = {"core_at_surface": "#e66767", "swollen": "#eda100", "internal": "#4a3aa7", "bare": "#3987e5",
               "typical": "#1baf7a"}


def write_geojson(t, path: Path) -> None:
    feats = [{"type": "Feature", "properties": {"target_id": r.target_id, "kind": r.kind, "section": r.section},
              "geometry": {"type": "Polygon", "coordinates": [[[r.x0_px, r.y0_px], [r.x1_px, r.y0_px],
                                                               [r.x1_px, r.y1_px], [r.x0_px, r.y1_px],
                                                               [r.x0_px, r.y0_px]]]}}
             for r in t.itertuples()]
    path.write_text(json.dumps({"type": "FeatureCollection", "features": feats}))


def preview(run, t, section: str, roi_um: float, path: Path) -> None:
    from matplotlib.patches import Rectangle

    sid = int(run.sections.loc[run.sections["section_name"].astype(str) == section, "section_id"].iloc[0])
    x0, x1, y1, y0 = R.section_bbox_um(run, sid)
    n = len(t)
    fig = plt.figure(figsize=(8 + 4 * min(n, 3), 8 if n <= 3 else 12), layout="constrained")
    gs = fig.add_gridspec(2 if n > 3 else 1, 1 + min(n, 3), width_ratios=[2] + [1] * min(n, 3))
    ax = fig.add_subplot(gs[:, 0])
    cx, cy, size = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
    rgb, ext = R.read_crop_um(run, cx, cy, size, scale=0.2)
    if rgb is not None:
        ax.imshow(np.clip(rgb * 1.2, 0, 1), extent=ext)
    for r in t.itertuples():
        col = KIND_COLORS[r.kind]
        ax.add_patch(Rectangle((r.x_um - roi_um / 2, r.y_um - roi_um / 2), roi_um, roi_um, fill=False, ec=col, lw=2))
        ax.text(r.x_um + roi_um / 2, r.y_um - roi_um / 2, r.target_id.rsplit("-", 1)[1], color=col, fontsize=12,
                fontweight="bold", va="bottom")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{section} · {run.name}", loc="left")
    for k, r in enumerate(t.itertuples()):
        a = fig.add_subplot(gs[k // 3, 1 + k % 3])
        rgb, ext = R.read_crop_um(run, r.x_um, r.y_um, roi_um, scale=1.0)
        if rgb is not None:
            a.imshow(np.clip(rgb * 1.3, 0, 1), extent=ext)
        a.set_xlim(r.x_um - roi_um / 2, r.x_um + roi_um / 2)
        a.set_ylim(r.y_um + roi_um / 2, r.y_um - roi_um / 2)
        a.set_xticks([])
        a.set_yticks([])
        for s in a.spines.values():
            s.set_edgecolor(KIND_COLORS[r.kind])
            s.set_linewidth(3)
        a.set_title(f"{r.target_id} · {r.kind}\n{r.why}", loc="left", fontsize=9)
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/sdata")
    ap.add_argument("--out", default="results/meninges_annotation")
    ap.add_argument("--sections", type=int, default=8)
    ap.add_argument("--roi-um", type=float, default=300.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = R.load_runs(a.run_dir)
    secs, t = annotation_targets(runs, a.sections, a.roi_um)
    secs.to_csv(out / "sections.csv", index=False)
    t.to_csv(out / "targets.csv", index=False)
    byname = {r.name: r for r in runs}
    for slide, ts in t.groupby("slide"):
        write_geojson(ts, out / f"{slide}_MENINGES_ROI.geojson")
    lines = ["# Meninges annotation targets", "",
             "Protocol: `docs/meninges_annotation.md`. Boxes per slide: `<slide>_MENINGES_ROI.geojson` (scene px).", "",
             f"{len(secs)} sections, {len(t)} boxes of {a.roi_um:.0f} µm, every animal covered.", "",
             "| section | slide | animal | boxes | kinds |", "|---|---|---|---|---|"]
    for s in secs.itertuples():
        ts = t[t["section"] == s.section]
        preview(byname[s.scene], ts, s.section, a.roi_um, out / f"preview_{s.section}.png")
        lines.append(f"| {s.section} | {ts['slide'].iloc[0] if len(ts) else ''} | {s.animal} | {len(ts)} | "
                     f"{', '.join(ts['kind'])} |")
    lines += ["", "## Previews (raw image; the current boundary is deliberately not shown)", ""]
    lines += [f"![{s}](preview_{s}.png)" for s in secs["section"]]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(secs.to_string(index=False))
    print(t.groupby("kind").size().to_string())


if __name__ == "__main__":
    main()
