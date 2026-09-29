"""Per-cell marker classification (Pu.1+ / Pu.1-)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def marker_signal(cells: pd.DataFrame, channel: str, *, background: str = "annulus") -> np.ndarray:
    """Background-corrected marker intensity per cell.

    ``background='annulus'`` subtracts the per-nucleus local annulus mean
    (``<ch>_bg``), which handles uneven illumination / autofluorescence.
    ``'none'`` returns the raw nuclear mean.
    """
    mean = cells[f"{channel}_mean"].to_numpy(dtype=float)
    if background == "none":
        return mean
    bg = cells[f"{channel}_bg"].to_numpy(dtype=float)
    return np.clip(mean - bg, 0, None)


def classify_marker(cells: pd.DataFrame, channel: str, *, method: str = "gmm", value: float | None = None,
                    background: str = "annulus", min_fold: float = 1.5, min_abs: float = 0.0,
                    out_col: str = "pu1_pos") -> tuple[pd.DataFrame, dict]:
    """Add boolean ``out_col`` (and ``<out_col>_score``) to ``cells``.

    method
        ``'gmm'``     – 2-component Gaussian mixture on log1p(signal); positive = high component
        ``'otsu'``    – Otsu threshold on log1p(signal)
        ``'quantile'``– positive if signal above the ``value`` quantile (0–1)
        ``'absolute'``– positive if signal > ``value`` (intensity units)
    min_fold / min_abs
        extra guards: nuclear mean must exceed ``min_fold`` × annulus background
        and background-corrected signal must exceed ``min_abs``.
    Returns the table and a dict describing the threshold used.
    """
    cells = cells.copy()
    sig = marker_signal(cells, channel, background=background)
    cells[f"{out_col}_score"] = sig
    logsig = np.log1p(sig)
    finite = np.isfinite(logsig) & (sig > 0)
    info = {"channel": channel, "method": method, "background": background}

    if method == "absolute":
        thr = float(value)
        pos = sig > thr
    elif method == "quantile":
        thr = float(np.quantile(sig[finite], value if value is not None else 0.9))
        pos = sig > thr
    elif method == "otsu":
        from skimage.filters import threshold_otsu

        t_log = threshold_otsu(logsig[finite]) if finite.sum() > 10 else np.inf
        thr = float(np.expm1(t_log))
        pos = sig > thr
    elif method == "gmm":
        from sklearn.mixture import GaussianMixture

        x = logsig[finite].reshape(-1, 1)
        if len(x) < 50:
            thr = float(np.expm1(np.quantile(logsig[finite], 0.9))) if len(x) else np.inf
        else:
            gm = GaussianMixture(2, random_state=0, n_init=3).fit(x)
            hi = int(np.argmax(gm.means_.ravel()))
            # threshold = where posterior of the high component crosses 0.5, found on a grid
            grid = np.linspace(x.min(), x.max(), 2000).reshape(-1, 1)
            post = gm.predict_proba(grid)[:, hi]
            cross = np.where(post >= 0.5)[0]
            t_log = float(grid[cross[0], 0]) if len(cross) else float(x.max())
            thr = float(np.expm1(t_log))
            info["gmm_means"] = np.expm1(gm.means_.ravel()).round(2).tolist()
            info["gmm_weights"] = gm.weights_.round(3).tolist()
        pos = sig > thr
    else:
        raise ValueError(f"unknown method {method!r}")

    mean = cells[f"{channel}_mean"].to_numpy(dtype=float)
    bg = cells[f"{channel}_bg"].to_numpy(dtype=float)
    guard = (mean >= min_fold * np.maximum(bg, 1e-6)) & (sig >= min_abs)
    pos = pos & guard
    cells[out_col] = pos
    info.update({"threshold": thr, "n_pos": int(pos.sum()), "n_total": len(pos),
                 "frac_pos": float(pos.mean()) if len(pos) else float("nan")})
    return cells, info
