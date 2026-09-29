"""Command-line interface: ``lesionseg --help``."""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Outline EAE lesions from nuclei + Pu.1 and export cells for DVP.")


@app.command()
def info(path: Path):
    """Print channels, pixel size and scenes of a slide."""
    from .io import open_slide

    r = open_slide(path)
    d = r.info() if hasattr(r, "info") else {"channels": r.channel_names, "pixel_size_um": r.pixel_size_um}
    typer.echo(json.dumps(d, indent=2))


@app.command()
def overview(path: Path, out: Path = typer.Option(Path("overview.png")), scale: float = 0.05,
             scene: int = 0):
    """Save a quick per-channel overview PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .io import open_slide
    from .viz import autoscale

    r = open_slide(path)
    stack = r.read_overview_stack(scene, scale, range(len(r.channel_names)))
    fig, axes = plt.subplots(1, len(stack), figsize=(6 * len(stack), 6))
    for ax, img, nm in zip(axes, stack, r.channel_names, strict=True):
        ax.imshow(autoscale(img), cmap="gray")
        ax.set_title(nm)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    typer.echo(f"wrote {out}")


@app.command()
def run(config: Path, only: list[str] | None = typer.Option(None, help="sample name(s) to run"),
        from_cells: bool = typer.Option(False, help="reuse existing cells.parquet, skip segmentation"),
        quiet: bool = False):
    """Run the full pipeline for all samples in a YAML config."""
    from .config import load_config
    from .pipeline import run_config

    cfg = load_config(config)
    logs = run_config(cfg, only=only, from_cells=from_cells, progress=not quiet)
    for lg in logs:
        typer.echo(f"{lg['sample']} scene {lg['scene']}: {lg['segmentation']['n_cells']} nuclei, "
                   f"{lg['pu1']['n_pos']} Pu.1+, {lg['lesion']['n_lesions']} lesion(s), {lg['seconds_total']} s")


@app.command("export-lmd")
def export_lmd_cmd(cells: Path, out: Path, calibration: str = typer.Option(..., help="x1,y1,x2,y2,x3,y3 in scene px"),
                   wells: str = typer.Option("core=A1,rim=A2,peri=A3,distal=A4"),
                   pixel_size_um: float = typer.Option(..., help="µm per full-res pixel"),
                   pu1_only: bool = True, dilate_um: float = 1.0):
    """Write LMD XML (py-lmd) of cell contours grouped by zone into wells."""
    import numpy as np
    import pandas as pd

    from .export import export_lmd

    df = pd.read_parquet(cells)
    calib = np.array([float(v) for v in calibration.split(",")]).reshape(3, 2)
    wmap = dict(kv.split("=") for kv in wells.split(","))
    only = df["pu1_pos"] if pu1_only else None
    counts = export_lmd(df, out, calibration_points_px=calib, wells=wmap, pixel_size_um=pixel_size_um,
                        only=only, dilate_um=dilate_um)
    typer.echo(json.dumps(counts))


if __name__ == "__main__":  # pragma: no cover
    app()
