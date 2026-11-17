"""Factor neutralization.

Each day the raw signal is regressed cross-sectionally on sector dummies,
market beta (and optionally log size); the residual is the neutralized alpha.
This strips out sector bets and market exposure so the alpha measures
stock-specific relative value only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sector_dummies(sector: pd.Series) -> np.ndarray:
    return pd.get_dummies(sector).to_numpy(dtype=float)  # n_tickers x n_sectors


def neutralize(signal: pd.DataFrame, sector: pd.Series,
               exposures: dict[str, pd.DataFrame] | None = None,
               universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """Residualize `signal` on sector dummies + continuous `exposures`, per day.

    Missing exposures are filled with the cross-sectional mean (zero after
    standardization) so a stock is not dropped for lack of a beta estimate.
    """
    sig = signal.to_numpy(dtype=float, copy=True)
    if universe is not None:
        sig[~universe.to_numpy()] = np.nan
    d = sector_dummies(sector.reindex(signal.columns))
    exp = [] if exposures is None else [e.reindex_like(signal).to_numpy(dtype=float) for e in exposures.values()]
    out = np.full_like(sig, np.nan)
    for t in range(sig.shape[0]):
        y = sig[t]
        ok = np.isfinite(y)
        if ok.sum() < 30:
            continue
        cols = [d[ok]]
        for e in exp:
            x = e[t, ok]
            fin = np.isfinite(x)
            if fin.sum() < 0.5 * ok.sum():
                continue
            mu, sd = x[fin].mean(), x[fin].std()
            x = np.where(fin, (x - mu) / (sd if sd > 0 else 1.0), 0.0)
            cols.append(x[:, None])
        X = np.hstack(cols)
        X = X[:, X.any(axis=0)]  # drop empty sectors
        beta, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        out[t, ok] = y[ok] - X @ beta
    return pd.DataFrame(out, index=signal.index, columns=signal.columns)


def standardize(signal: pd.DataFrame, clip: float = 3.0) -> pd.DataFrame:
    """Cross-sectional z-score, clipped at +-clip (robust to outliers)."""
    z = signal.sub(signal.mean(axis=1), axis=0).div(signal.std(axis=1), axis=0)
    return z.clip(-clip, clip)
