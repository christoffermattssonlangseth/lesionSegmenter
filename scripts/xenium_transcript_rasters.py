"""Bin Xenium transcripts of every capture region into 5 µm rasters (all genes + meninges / border gene groups).

Runs where the raw Xenium output is (the remote, env ``sq``: numpy, pandas, pyarrow – no lesionseg needed):

    python xenium_transcript_rasters.py --regions capture_regions.csv --root /Volumes/moldiassd --out rasters/ -j 4

One ``<xenium_output>.npz`` per region: ``total`` (all decoded transcripts, qv ≥ 20) and one raster per gene group,
uint16 counts per bin, plus ``origin_um`` / ``res_um`` to map region µm (the frame of ``cells.parquet``) to bins.
Gene groups were chosen from the RRMAP2 atlas (detection in meningeal fibroblasts vs all other cells and vs
lesion fibroblasts; see notebook 14).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

GROUPS = {
    "meningeal": ["Slc47a1", "Foxc2", "Crabp2", "Osr1", "Slc13a4", "Smoc2", "Wnt6", "Klf5", "Fmod", "Cdh1"],
    "collagen": ["Col1a1", "Col1a2", "Lum", "Pdgfra"],
    "ptgds": ["Ptgds"],
    "glia_limitans": ["Aqp4", "Gfap"],
    "immune": ["Ptprc", "Cd74", "Spi1", "Cd68", "Cd3e", "Ms4a1"],
    "cam": ["Mrc1"],
    "aif1": ["Aif1"],
    "parenchyma": ["Mbp", "Snap25"],
}
RES_UM = 5.0


def rasterize(path: Path, out: Path) -> str:
    dst = out / f"{path.name}.npz"
    if dst.exists():
        return f"skip {path.name}"
    t = pq.read_table(path / "transcripts.parquet", columns=["feature_name", "x_location", "y_location", "qv"],
                      filters=[("qv", ">=", 20)]).to_pandas()
    gene = t["feature_name"].astype(str)
    x, y = t["x_location"].to_numpy(np.float32), t["y_location"].to_numpy(np.float32)
    x0, y0 = float(np.floor(x.min())), float(np.floor(y.min()))
    j, i = ((x - x0) / RES_UM).astype(np.int32), ((y - y0) / RES_UM).astype(np.int32)
    shape = (int(i.max()) + 1, int(j.max()) + 1)

    def hist(m):
        return np.bincount(i[m] * shape[1] + j[m], minlength=shape[0] * shape[1]).reshape(shape).clip(0, 65535) \
            .astype(np.uint16)

    real = ~gene.str.startswith(("NegControl", "Unassigned", "Deprecated", "BLANK", "Intergenic")).to_numpy()
    arrs = {"total": hist(real)}
    for g, genes in GROUPS.items():
        arrs[g] = hist(gene.isin(genes).to_numpy())
    np.savez_compressed(dst, origin_um=np.array([x0, y0]), res_um=np.array(RES_UM), **arrs)
    return f"done {path.name} {shape} {int(real.sum()):,} transcripts"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", required=True, help="capture_regions.csv of the RRMAP2 sample sheet")
    ap.add_argument("--root", default="/Volumes/moldiassd")
    ap.add_argument("--out", default="rasters")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = pd.read_csv(a.regions)
    reg = reg[reg["in_atlas"].astype(str).str.lower().isin(["true", "1", "yes"])]
    paths = [Path(a.root) / r.run_folder / r.xenium_output for r in reg.itertuples()]
    with ProcessPoolExecutor(a.j) as ex:
        for msg in ex.map(rasterize, paths, [out] * len(paths)):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
