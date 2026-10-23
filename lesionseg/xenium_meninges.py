"""Where are the meninges? An independent, gene-based answer from the RRMAP2 Xenium 5k EAE spinal cord atlas.

The imaging data of this project (nuclei, Pu.1, Iba1) cannot tell meninges from subpial lesion. The Xenium sections
(157 sections, CFA controls and every EAE phase) can: meningeal fibroblasts express genes nothing else does, and the
glia limitans (Aqp4, Gfap) marks the parenchymal side of the pia. Inputs:

* per capture region, 5 µm transcript rasters (``scripts/xenium_transcript_rasters.py``, run where the raw data are);
* the atlas cells (``cell_id``, section, cell type, niche) and the RRMAP2 sample sheet (``sections.csv``).

Per section the tissue is the area with transcripts (not with called cells – the atlas keeps only QC-passed cells),
depth is the distance to its edge, exactly as the image pipeline measures it from its own tissue mask. Every tissue
pixel is assigned to the nearest edge point, edge points are grouped into ``SEG_UM`` stretches, and per stretch the
depth profiles of the gene groups give the **pia depth** = the deepest depth at which meningeal genes still exceed
the glia-limitans genes (both as fractions of all transcripts, relative to the section's own surface levels).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

GROUPS = ["total", "meningeal", "collagen", "ptgds", "glia_limitans", "immune", "cam", "aif1", "parenchyma"]
MENINGEAL_FB = ["Meningeal FBs", "Meningeal EAE FBs", "Meningeal/Perivascular FBs"]
IMMUNE_L1 = ["Myeloid", "T cell", "B cell", "DC", "NK/DC", "Neutrophil"]
SEG_UM, MAX_DEPTH_UM, PAD_UM = 50.0, 200.0, 100.0
MIN_MENINGEAL = 0.1   # meningeal share ≥ 10 % of the way from deep to surface level (sweep in notebook 14)
PHASES = ["CFA control", "pre-symptomatic", "onset", "peak", "remission", "late", "early chronic (day 16)",
          "late chronic (day 30)"]


def load_region(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def tissue_mask(total: np.ndarray, res: float, frac: float = 0.5, edge_um: float = 20.0) -> np.ndarray:
    """Tissue: a solid body from heavily smoothed transcript density (σ 15 µm, > ¼ of the median tissue level,
    holes filled – white matter has fewer transcripts and a fine threshold turns it into lace with false inner
    edges), whose outer ``edge_um`` is refined at half the median density (σ 5 µm) so the edge sits at half
    maximum. A single looser mask sat 15–20 µm outside the outermost cells."""
    from skimage.filters import threshold_otsu

    t = total.astype(np.float32)
    fine = ndi.gaussian_filter(t, 1.0)
    coarse = ndi.gaussian_filter(t, 3.0)
    rough = np.log1p(fine) > threshold_otsu(np.log1p(fine[fine > 0]))
    med = np.median(fine[rough])
    body = ndi.binary_fill_holes(ndi.binary_closing(coarse > 0.25 * med, iterations=2))
    inner = ndi.distance_transform_edt(body) * res > edge_um
    m = body & (inner | (fine > frac * med))
    m = ndi.binary_opening(m, iterations=1)
    return ndi.binary_fill_holes(m)


def section_frame(reg: dict, row, cells: pd.DataFrame):
    """Crop of the region rasters around one section; tissue components holding this section's cells only."""
    x0, y0 = reg["origin_um"]
    res = float(reg["res_um"])
    H, W = reg["total"].shape
    c0, c1 = max(int((row.x_min_um - PAD_UM - x0) / res), 0), min(int((row.x_max_um + PAD_UM - x0) / res), W)
    r0, r1 = max(int((row.y_min_um - PAD_UM - y0) / res), 0), min(int((row.y_max_um + PAD_UM - y0) / res), H)
    crop = {k: reg[k][r0:r1, c0:c1] for k in GROUPS}
    tis = tissue_mask(crop["total"], res)
    lab, _ = ndi.label(tis)
    ci = ((cells["y_centroid"].to_numpy() - y0) / res).astype(int) - r0
    cj = ((cells["x_centroid"].to_numpy() - x0) / res).astype(int) - c0
    ok = (ci >= 0) & (ci < lab.shape[0]) & (cj >= 0) & (cj < lab.shape[1])
    hit = np.bincount(lab[ci[ok], cj[ok]], minlength=lab.max() + 1)
    # components of this section: hold ≥ 1 % of its cells (drops neighbouring sections and debris)
    keep = np.flatnonzero(hit >= max(0.01 * ok.sum(), 5))
    keep = keep[keep > 0]
    tis = np.isin(lab, keep)
    origin = (x0 + c0 * res, y0 + r0 * res)
    return crop, tis, origin, res


def dead_zone(total: np.ndarray, size_px: int = 5) -> np.ndarray:
    """Pixels in a patch with no transcripts at all (≥ size_px × size_px zeros, grown by 2 px): a missing field of
    view or a crop limit. A real tissue edge is followed by a halo of diffused transcripts, never by exact zeros."""
    z = ndi.uniform_filter((total == 0).astype(np.float32), size_px, mode="constant", cval=1.0) > 0.999
    return ndi.binary_dilation(z, iterations=2)


def depth_and_segments(tis: np.ndarray, res: float, origin, total: np.ndarray | None = None):
    """Depth (µm) per pixel and the id of the SEG_UM edge stretch its nearest edge point falls in. False edges –
    tissue cut by the frame, or whose nearest outside pixel lies in a transcript-free patch (``dead_zone`` of
    ``total``) – get stretch id −1 and are left out."""
    pad = np.pad(tis, 1)
    dist, (ii, jj) = ndi.distance_transform_edt(pad, return_indices=True)
    bad = (ii == 0) | (jj == 0) | (ii == pad.shape[0] - 1) | (jj == pad.shape[1] - 1)
    if total is not None:
        dead = np.pad(dead_zone(total), 1, constant_values=True)
        bad |= dead[ii, jj]
    dist, ii, jj, bad = (a[1:-1, 1:-1] for a in (dist, ii - 1, jj - 1, bad))
    depth = dist * res
    sx = (origin[0] + jj * res) // SEG_UM
    sy = (origin[1] + ii * res) // SEG_UM
    seg = np.where(bad, -1, sy * 100000 + sx).astype(np.int64)
    return depth, seg


def section_profiles(crop: dict, tis, depth, seg, step: float = 5.0) -> pd.DataFrame:
    """Counts per (edge stretch, depth bin) for every gene group, pixels ≤ MAX_DEPTH_UM deep."""
    m = tis & (depth <= MAX_DEPTH_UM) & (seg >= 0)
    db = (depth[m] // step * step).astype(np.float32)
    df = pd.DataFrame({"seg": seg[m], "depth": db, **{k: crop[k][m].astype(np.int32) for k in GROUPS}})
    df["pixels"] = 1
    return df.groupby(["seg", "depth"], as_index=False).sum()


def cell_depth(cells: pd.DataFrame, tis, depth, seg, origin, res) -> pd.DataFrame:
    i = ((cells["y_centroid"].to_numpy() - origin[1]) / res).astype(int)
    j = ((cells["x_centroid"].to_numpy() - origin[0]) / res).astype(int)
    ok = (i >= 0) & (i < tis.shape[0]) & (j >= 0) & (j < tis.shape[1])
    d = np.full(len(cells), np.nan)
    s = np.full(len(cells), -1, np.int64)
    inside = ok.copy()
    inside[ok] = tis[i[ok], j[ok]]
    d[inside] = depth[i[inside], j[inside]]
    s[inside] = seg[i[inside], j[inside]]
    return cells.assign(depth=d, seg=s)


def _last_true_from_surface(flag: np.ndarray, dep: np.ndarray) -> float:
    """Depth of the deepest ring of the unbroken run of True from the surface (0 if the first ring is False)."""
    k = int(np.argmin(flag)) if not flag.all() else len(flag)
    return float(dep[k - 1]) if k > 0 else 0.0     # a ring labelled d µm spans (d − 5, d]


def pia_depth(prof: pd.DataFrame, ref: dict, min_total: int = 400) -> pd.DataFrame:
    """Per edge stretch, the pia placed two ways (5 µm rings, 3 pooled – a ring of a 50 µm stretch holds ~2,000
    transcripts but only a handful of meningeal ones):

    * ``pia_depth_um`` (**balance**): the unbroken run of rings from the surface where the meningeal share of
      transcripts is ≥ MIN_MENINGEAL and beats the glia-limitans share, both scaled between the section's own
      surface (0–10 µm) and deep (50–150 µm) levels. Immune transcripts dilute the meningeal *share* in swollen
      meninges, so this leans towards "subpial".
    * ``pia_glia_um`` (**glia**): the unbroken run of rings from the surface where the glia-limitans *density*
      (counts per area, not diluted) is below half its deep level – no astrocyte endfeet yet. Astrocyte loss in a
      lesion makes this lean towards "meningeal".

    Stretches whose glia-limitans share never reaches half its deep level within 150 µm are dropped (no parenchyma
    under them: flaps, tears, oblique cuts). Also the stretch's immune load above and below the (balance) pia."""
    out = []
    for (sec, sg), d in prof.groupby(["section", "seg"], sort=False):
        d = d.sort_values("depth")
        if d["total"].sum() < min_total or d["depth"].min() > 5:   # the outermost pixel ring is 5 µm deep
            continue
        sm = d[["meningeal", "glia_limitans", "total", "pixels"]].rolling(3, center=True, min_periods=1).sum()
        men = (sm["meningeal"] / sm["total"].clip(lower=1)).to_numpy()
        gl = (sm["glia_limitans"] / sm["total"].clip(lower=1)).to_numpy()
        gd = (sm["glia_limitans"] / sm["pixels"].clip(lower=1)).to_numpy()
        r = ref[sec]
        mz = (men - r["men_deep"]) / max(r["men_surf"] - r["men_deep"], 1e-6)
        gz = (gl - r["gl_surf"]) / max(r["gl_deep"] - r["gl_surf"], 1e-6)
        dep = d["depth"].to_numpy()
        if not (gz[dep <= 150] >= 0.5).any():
            continue
        pia = _last_true_from_surface((mz > gz) & (mz >= MIN_MENINGEAL), dep)
        pia_g = _last_true_from_surface(gd < 0.5 * r["gl_deep_density"], dep)
        imm = d["immune"].to_numpy()
        tot = d["total"].to_numpy()
        out.append({"section": sec, "seg": sg, "pia_depth_um": pia, "pia_glia_um": pia_g, "total": int(tot.sum()),
                    "immune_above_pia": int(imm[dep <= pia].sum()), "immune_below_pia_0_50": int(
                        imm[(dep > pia) & (dep <= pia + 50)].sum()),
                    "immune_frac_0_80": imm[dep <= 80].sum() / max(tot[dep <= 80].sum(), 1)})
    return pd.DataFrame(out)


def section_refs(prof: pd.DataFrame) -> dict:
    """Section-level references: surface (0–10 µm) and deep (50–150 µm) shares of meningeal and glia-limitans
    genes, and the deep glia-limitans density (counts per 5 µm pixel)."""
    ref = {}
    for sec, d in prof.groupby("section"):
        s, dp = d[d["depth"] < 10], d[(d["depth"] >= 50) & (d["depth"] < 150)]

        def f(x, k):
            return x[k].sum() / max(x["total"].sum(), 1)
        ref[sec] = {"men_surf": f(s, "meningeal"), "men_deep": f(dp, "meningeal"),
                    "gl_surf": f(s, "glia_limitans"), "gl_deep": f(dp, "glia_limitans"),
                    "gl_deep_density": dp["glia_limitans"].sum() / max(dp["pixels"].sum(), 1)}
    return ref


def run(raster_dir: Path, cells: pd.DataFrame, sections: pd.DataFrame):
    """(profiles per stretch × depth, atlas cells with depth / stretch) for every section."""
    profs, cl = [], []
    for xo, rows in sections.groupby("xenium_output"):
        f = Path(raster_dir) / f"{xo}.npz"
        if not f.exists():
            continue
        reg = load_region(f)
        for row in rows.itertuples():
            c = cells[cells["section"] == row.section]
            if c.empty:
                continue
            crop, tis, origin, res = section_frame(reg, row, c)
            depth, seg = depth_and_segments(tis, res, origin, crop["total"])
            profs.append(section_profiles(crop, tis, depth, seg).assign(section=row.section))
            cl.append(cell_depth(c, tis, depth, seg, origin, res))
    return pd.concat(profs, ignore_index=True), pd.concat(cl, ignore_index=True)


def section_view(raster_dir: Path, row, cells: pd.DataFrame, pia: pd.DataFrame, col: str = "pia_depth_um"):
    """(rgb of meningeal / glia-limitans / immune transcripts, gene-defined meninges mask, tissue, extent µm) for
    plotting. Meninges = tissue pixels no deeper than their edge stretch's pia (column ``col`` of ``pia``)."""
    reg = load_region(Path(raster_dir) / f"{row.xenium_output}.npz")
    crop, tis, origin, res = section_frame(reg, row, cells)
    depth, seg = depth_and_segments(tis, res, origin, crop["total"])
    pd_map = pia[pia["section"] == row.section].set_index("seg")[col]
    lim = pd.Series(seg.ravel()).map(pd_map).to_numpy().reshape(seg.shape)
    men = tis & (depth <= np.nan_to_num(lim, nan=-1))

    def ch(k, q=99.5):
        a = ndi.gaussian_filter(crop[k].astype(np.float32), 1.0)
        return np.clip(a / max(np.percentile(a[tis], q), 1e-6), 0, 1)
    rgb = np.stack([ch("glia_limitans"), ch("meningeal"), ch("immune")], -1)   # glia red, meningeal green, immune blue
    rgb = np.clip(rgb * 1.2 + 0.08 * tis[..., None], 0, 1)
    ext = (origin[0], origin[0] + tis.shape[1] * res, origin[1] + tis.shape[0] * res, origin[1])
    return rgb, men, tis, ext


# ---- meningeal regions: one row per surface stretch ------------------------------------------------------------
CELL_GROUPS = {
    "T_CD4": ["CD4+ T Cell"], "T_CD8": ["CD8+ T Cell"], "T_other": ["T Cell", "T cell_LowQuality"],
    "B": ["B Cell", "B cell_LowQuality"], "MDM": ["MDM"], "CAM": ["CAM"], "DC": ["GMP-DC", "CLP-DC", "NK/DC"],
    "neutrophil": ["Neutrophil"], "microglia": ["Microglia"], "meningeal_FB": MENINGEAL_FB,
    "EAE_FB": ["EAE - Associated FBs"], "perineural_FB": ["Perineural FBs"], "schwann": ["Schwann cell"],
    "endothelial": ["Endothelial"], "VSMC": ["VSMC"],
    "astrocyte": ["Homo_AST", "Reactive_AST", "Myelin_AST", "Neurovascular_AST"],
    "oligodendroglia": ["MOL", "NFOL", "DAO", "OPC/COP"],
}
IMMUNE_GROUPS = ["T_CD4", "T_CD8", "T_other", "B", "MDM", "CAM", "DC", "neutrophil"]
ANATOMY = {"Dorsal funiculus": "dorsal", "DorsalHorn_sensory": "dorsal", "Lateral funiculus": "lateral",
           "Ventral funiculus": "ventral", "VentralHorn_motor": "ventral", "DRG": "DRG"}


def stretch_table(cells: pd.DataFrame, pia: pd.DataFrame, prof: pd.DataFrame, col: str = "pia_depth_um",
                  subpial_um: float = 50.0) -> pd.DataFrame:
    """One row per surface stretch with a pia: surface length, pia depth, cell counts by group in the meninges
    (above the pia), in the subpial band (pia … pia + ``subpial_um``) and deeper (… 200 µm), and the anatomy of
    the parenchyma under it (majority physiological niche of cells 50–200 µm below the pia: dorsal / lateral /
    ventral / DRG). Atlas cells must carry ``section``, ``seg``, ``depth``, ``Anno_L2``, ``Physiological_niche``."""
    c = cells.merge(pia[["section", "seg", col]], on=["section", "seg"])
    rel = c["depth"] - c[col]
    c["comp"] = np.select([rel <= 0, rel <= subpial_um, rel <= 200], ["men", "sub", "deep"], "")
    c = c[c["comp"] != ""]
    lab = c["Anno_L2"].astype(str)
    c["cg"] = "other"
    for g, types in CELL_GROUPS.items():
        c.loc[lab.isin(types), "cg"] = g
    cnt = c.groupby(["section", "seg", "comp", "cg"]).size().unstack(["comp", "cg"], fill_value=0)
    cnt.columns = [f"{a}_{b}" for a, b in cnt.columns]
    under = c[(c["comp"] != "men") & (rel > 50)]
    anat = (under["Physiological_niche"].astype(str).map(ANATOMY).dropna().groupby([under["section"], under["seg"]])
            .agg(lambda v: v.value_counts().idxmax()).rename("anatomy"))
    length = (prof[prof["depth"] == prof["depth"].min()].groupby(["section", "seg"])["pixels"].sum()
              * float(prof["depth"].min())).rename("surface_um")
    t = (pia.set_index(["section", "seg"]).join([cnt, anat, length], how="left").reset_index())
    num = [k for k in t.columns if k.startswith(("men_", "sub_", "deep_"))]
    t[num] = t[num].fillna(0).astype(int)
    for comp in ("men", "sub"):
        t[f"{comp}_immune"] = t[[f"{comp}_{g}" for g in IMMUNE_GROUPS if f"{comp}_{g}" in t]].sum(1)
    t["men_cells"] = t[[k for k in t.columns if k.startswith("men_") and k != "men_immune"]].sum(1)
    t["anatomy"] = t["anatomy"].fillna("unassigned")
    t["root_zone"] = (t.get("men_schwann", 0) + t.get("men_perineural_FB", 0) + t.get("sub_schwann", 0)) >= 2
    t["layer"] = t[col] > 0
    return t
