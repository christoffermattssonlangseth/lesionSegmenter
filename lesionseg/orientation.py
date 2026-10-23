"""Nuclear orientation relative to the tissue surface (report notebook 12).

Are the elongated nuclei of the outer meninges lying *along* the surface, as flattened pial / arachnoid
cells would, or at random / radially, as squashed or migrating cells might?

* **Long axis per nucleus** from the second moments of its mask in the Cellpose label image (the same
  ellipse that defines the cell table's ``eccentricity``). The moments are sums over pixels, so the label
  image is read chunk by chunk and summed exactly – every nucleus, no crops. Cached per scene in
  ``nucleus_orientation.parquet`` next to ``cells.parquet``.
* **Local surface direction** from the fine tissue raster (``maps/surface_tiers.tif``, tier > 0): the
  gradient of the smoothed depth below the surface points inward, along the surface normal.
* **surface_angle** = angle between the long axis and the surface tangent, 0–90°: 0 = lying along the
  surface, 90 = pointing into the tissue. Random axes give a uniform angle (mean 45°, 1/3 below 30°).
* **Order parameter** S = mean cos(2 · surface_angle): +1 all parallel, 0 random, −1 all radial.

Only elongated nuclei have a meaningful axis; round ones are kept but analysed separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

PARALLEL_MAX = 30.0   # degrees: "along the surface"
CACHE = "nucleus_orientation.parquet"


def label_moments(labels, ref_xy: np.ndarray, chunk: int = 4096) -> pd.DataFrame:
    """Second-moment ellipse of every label in a (lazy) 2-D label array.

    ``ref_xy`` (n_max + 1 × 2, scene px) is a reference point per label (its centroid from the cell table);
    coordinates are taken relative to it so the sums stay small. Returns per label (> 0, with pixels):
    ``n_px``, ``orient_rad`` (long axis, image coordinates: x right, y down, in (−π/2, π/2]) and
    ``ecc_mom``."""
    H, W = labels.shape
    nmax = len(ref_xy) - 1
    acc = np.zeros((6, nmax + 1))
    for r0 in range(0, H, chunk):
        for c0 in range(0, W, chunk):
            lab = np.asarray(labels[r0:r0 + chunk, c0:c0 + chunk])
            yy, xx = np.nonzero(lab)
            if not len(yy):
                continue
            ll = lab[yy, xx]
            keep = ll <= nmax
            ll, yy, xx = ll[keep], yy[keep], xx[keep]
            x = xx + c0 - ref_xy[ll, 0]
            y = yy + r0 - ref_xy[ll, 1]
            for k, w in enumerate((None, x, y, x * x, y * y, x * y)):
                acc[k] += np.bincount(ll, weights=w, minlength=nmax + 1)
    n, sx, sy, sxx, syy, sxy = acc
    ok = n > 0
    ok[0] = False
    n = n[ok]
    mx, my = sx[ok] / n, sy[ok] / n
    mu20, mu02, mu11 = sxx[ok] / n - mx ** 2, syy[ok] / n - my ** 2, sxy[ok] / n - mx * my
    common = np.sqrt(((mu20 - mu02) / 2) ** 2 + mu11 ** 2)
    l1, l2 = (mu20 + mu02) / 2 + common, (mu20 + mu02) / 2 - common
    ecc = np.sqrt(np.clip(1 - l2 / np.where(l1 > 0, l1, np.nan), 0, 1))
    return pd.DataFrame({"label": np.flatnonzero(ok), "n_px": n.astype(int),
                         "orient_rad": 0.5 * np.arctan2(2 * mu11, mu20 - mu02), "ecc_mom": ecc})


def nucleus_orientation(run, refresh: bool = False) -> pd.DataFrame:
    """Per-nucleus long axis for one scene (cached in the scene's run directory)."""
    f = run.dir / CACHE
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    import zarr

    arr = zarr.open(str(run.log["cells"]["labels"]), mode="r")
    c = run.cells
    lab = c["label"].to_numpy(int)
    ref = np.zeros((int(lab.max()) + 1, 2))
    ref[lab] = c[["x_px", "y_px"]].to_numpy(float)
    out = label_moments(arr, ref)
    out.to_parquet(f)
    return out


def surface_normal(tissue: np.ndarray, res_um: float, sigma_um: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Unit inward normal (nx, ny per pixel, image coordinates) of a tissue raster: the gradient of the
    smoothed distance to the tissue edge."""
    depth = ndi.distance_transform_edt(tissue) * res_um
    depth = ndi.gaussian_filter(depth.astype(np.float32), sigma_um / res_um)
    gy, gx = np.gradient(depth)
    norm = np.hypot(gx, gy)
    norm[norm == 0] = np.nan
    return gx / norm, gy / norm


def surface_angle(orient_rad, nx, ny) -> np.ndarray:
    """Angle (degrees, 0–90) between a long axis and the surface tangent: 0 = along the surface."""
    ux, uy = np.cos(orient_rad), np.sin(orient_rad)
    return np.degrees(np.arcsin(np.clip(np.abs(ux * nx + uy * ny), 0, 1)))


def cell_surface_angles(run, orient: pd.DataFrame | None = None, sigma_um: float = 5.0) -> pd.DataFrame:
    """``label``, ``orient_rad``, ``ecc_mom``, ``normal_x/y`` and ``surface_angle`` for every cell of
    ``run`` that lies in the tissue raster."""
    from .report import surface_tiers

    tiers, res = surface_tiers(run)
    if tiers is None:
        return pd.DataFrame()
    orient = nucleus_orientation(run) if orient is None else orient
    nx, ny = surface_normal(tiers > 0, res, sigma_um)
    c = run.cells[["label", "x_um", "y_um"]].merge(orient, on="label", how="inner")
    iy = (c["y_um"] / res).astype(int).clip(0, tiers.shape[0] - 1).to_numpy()
    ix = (c["x_um"] / res).astype(int).clip(0, tiers.shape[1] - 1).to_numpy()
    c["normal_x"], c["normal_y"] = nx[iy, ix], ny[iy, ix]
    c["surface_angle"] = surface_angle(c["orient_rad"].to_numpy(), c["normal_x"].to_numpy(), c["normal_y"].to_numpy())
    return c.drop(columns=["x_um", "y_um"])


def orientation_table(runs, df: pd.DataFrame, sigma_um: float = 5.0) -> pd.DataFrame:
    """``df`` (``expression.expression_table``) with the surface angle per nucleus added (inner join);
    ``sigma_um`` smooths the surface before its direction is taken."""
    frames = [cell_surface_angles(r, sigma_um=sigma_um).assign(scene=r.name) for r in runs]
    ang = pd.concat(frames, ignore_index=True)
    out = df.merge(ang, on=["scene", "label"], how="inner")
    return out[out["surface_angle"].notna()]


def order_parameter(angle_deg) -> float:
    """S = mean cos(2θ): +1 all along the surface, 0 random, −1 all pointing into the tissue."""
    a = np.radians(np.asarray(angle_deg, float))
    return float(np.mean(np.cos(2 * a))) if a.size else np.nan


def depth_groups(df: pd.DataFrame) -> pd.Series:
    """Per nucleus: ``meninges 0–10 µm`` / ``meninges 10–80 µm`` (surface meninges), ``parenchyma 20–80 µm``
    (the edge control: same surface, no meninges), ``parenchyma 150–300 µm`` (deep; far enough that the
    surface direction should not matter); else None."""
    comp = df["compartment"].astype(str).to_numpy()
    d = df["depth_um"].to_numpy(float)
    men = comp == "meninges"
    par = np.isin(comp, ["core", "rim", "peri", "deep", "distal", "GM", "WM"])
    out = np.full(len(df), None, dtype=object)
    out[men & (d <= 10)] = "meninges 0–10 µm"
    out[men & (d > 10) & (d <= 80)] = "meninges 10–80 µm"
    out[par & (d >= 20) & (d <= 80)] = "parenchyma 20–80 µm"
    out[par & (d >= 150) & (d <= 300)] = "parenchyma 150–300 µm"
    return pd.Series(out, index=df.index)


GROUPS = ["meninges 0–10 µm", "meninges 10–80 µm", "parenchyma 20–80 µm", "parenchyma 150–300 µm"]


def section_alignment(df: pd.DataFrame, ecc_min: float = 0.85, min_cells: int = 15) -> pd.DataFrame:
    """Per section × group (replicate slides averaged): order parameter S and share (%) of nuclei within
    ``PARALLEL_MAX`` of the surface, for nuclei with eccentricity ≥ ``ecc_min``."""
    d = df.assign(group=depth_groups(df))
    d = d[d["group"].notna() & (d["eccentricity"] >= ecc_min)]
    d = d.assign(cos2=np.cos(2 * np.radians(d["surface_angle"])), par=(d["surface_angle"] < PARALLEL_MAX) * 100.0)
    g = d.groupby(["scene", "section", "group"]).agg(S=("cos2", "mean"), parallel_pct=("par", "mean"),
                                                     n=("cos2", "size")).reset_index()
    g = g[g["n"] >= min_cells]
    per = g.groupby(["section", "group"])[["S", "parallel_pct", "n"]].mean().reset_index()
    per["animal"] = per["section"].str.rsplit("_", n=1).str[0]
    return per


def depth_alignment(df: pd.DataFrame, ecc_min: float = 0.85,
                    edges=(0, 10, 20, 30, 40, 60, 80, 120, 160, 200, 300, 500)) -> pd.DataFrame:
    """Order parameter of elongated nuclei per depth bin, per slide (any compartment with a depth)."""
    d = df[df["depth_um"].notna() & (df["eccentricity"] >= ecc_min)]
    b = pd.cut(d["depth_um"], list(edges), include_lowest=True)
    g = d.assign(cos2=np.cos(2 * np.radians(d["surface_angle"]))).groupby(["scene", b], observed=True)["cos2"].agg(
        ["mean", "size"]).reset_index()
    g["mid_um"] = g["depth_um"].map(lambda iv: iv.mid).astype(float)
    return g[g["size"] >= 30]


def shuffled_null(df: pd.DataFrame, n_perm: int = 200, seed: int = 0) -> np.ndarray:
    """Order parameter of ``df`` when the surface normals are shuffled between nuclei of the same slide:
    what is expected if axes had no relation to the *local* surface but kept any slide-wide preferred
    direction (cutting direction, scan grid) and the slide's mix of surface directions."""
    rng = np.random.default_rng(seed)
    o = df["orient_rad"].to_numpy(float)
    nx, ny = df["normal_x"].to_numpy(float), df["normal_y"].to_numpy(float)
    groups = [np.flatnonzero(m) for m in (df["scene"] == s for s in df["scene"].unique())]
    out = np.empty(n_perm)
    for b in range(n_perm):
        perm = np.arange(len(df))
        for g in groups:
            perm[g] = rng.permutation(g)
        out[b] = order_parameter(surface_angle(o, nx[perm], ny[perm]))
    return out


def paired_test(per: pd.DataFrame, a: str, b: str, value: str = "S") -> dict:
    """Sections with both groups: median ``a`` − ``b``, sections higher, Wilcoxon p and hierarchical-bootstrap
    95 % CI of the mean difference (animals, then sections)."""
    from scipy.stats import wilcoxon

    from .morphology import hierarchical_bootstrap

    t = per.pivot_table(index=["section", "animal"], columns="group", values=value).dropna(subset=[a, b])
    d = (t[a] - t[b]).rename("delta").reset_index()
    lo, hi = hierarchical_bootstrap(d)
    return {"comparison": f"{a} − {b}", "sections": len(d), "animals": d["animal"].nunique(),
            "median_delta": d["delta"].median(), "mean_delta": d["delta"].mean(), "ci95_low": lo, "ci95_high": hi,
            "sections_higher": int((d["delta"] > 0).sum()),
            "wilcoxon_p": wilcoxon(d["delta"]).pvalue if len(d) >= 6 else np.nan}


def draw_axes(ax, cells: pd.DataFrame, orient: pd.DataFrame, view: tuple[float, float, float, float],
              ecc_min: float = 0.85, length_um: float = 8.0, cmap=None):
    """Long axis of every elongated nucleus inside ``view`` = (x0, x1, y0, y1) µm as a short line coloured
    by ``surface_angle`` (needs a ``surface_angle`` column in ``orient``)."""
    from matplotlib.collections import LineCollection

    x0, x1, y0, y1 = view
    c = cells[["label", "x_um", "y_um"]].merge(orient, on="label")
    c = c[c["x_um"].between(x0, x1) & c["y_um"].between(y0, y1) & (c["ecc_mom"] >= ecc_min)]
    dx, dy = np.cos(c["orient_rad"]) * length_um / 2, np.sin(c["orient_rad"]) * length_um / 2
    seg = np.stack([np.c_[c["x_um"] - dx, c["y_um"] - dy], np.c_[c["x_um"] + dx, c["y_um"] + dy]], axis=1)
    lc = LineCollection(seg, array=c["surface_angle"].to_numpy(), cmap=cmap, clim=(0, 90), linewidths=2.2)
    ax.add_collection(lc)
    return lc
