"""Alpha selection and walk-forward combination.

Selection uses the in-sample period only. Combination weights are re-fit every
month by pooled cross-sectional ridge regression on a trailing window, with
the training labels *purged* so none of them overlap the period being
forecast (labels at day s use returns through s+delay+h).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def select_alphas(scores: pd.DataFrame, ls_pnl: dict[str, pd.Series], is_slice: slice,
                  min_abs_t: float = 2.0, max_corr: float = 0.5) -> tuple[list[str], dict[str, int]]:
    """Greedy selection on in-sample statistics.

    1. Orient each alpha by the sign of its in-sample IC.
    2. Keep alphas whose in-sample Newey-West IC t-stat exceeds `min_abs_t`.
    3. Walk down by |t| and admit an alpha only if its daily long-short PnL
       correlation with every admitted alpha is below `max_corr`.
    """
    sign = np.sign(scores["is_ic"]).astype(int).to_dict()
    cand = scores[scores["is_ic_t"].abs() >= min_abs_t].copy()
    cand = cand.reindex(cand["is_ic_t"].abs().sort_values(ascending=False).index)
    chosen: list[str] = []
    for name in cand.index:
        p = ls_pnl[name].loc[is_slice] * sign[name]
        if all(abs(p.corr(ls_pnl[c].loc[is_slice] * sign[c])) < max_corr for c in chosen):
            chosen.append(name)
    return chosen, sign


def stack(alphas: dict[str, pd.DataFrame], names: list[str], sign: dict[str, int]) -> np.ndarray:
    """(T, N, K) array of oriented, standardized alphas; missing -> 0 (neutral)."""
    z = np.stack([alphas[n].to_numpy(dtype=float) * sign[n] for n in names], axis=-1)
    return z


def walk_forward_ridge(Z: np.ndarray, y: np.ndarray, valid: np.ndarray, dates: pd.DatetimeIndex,
                       lam: float, window: int, horizon: int, delay: int = 1,
                       embargo: int = 5, start: pd.Timestamp | None = None,
                       refit: str = "ME") -> tuple[np.ndarray, pd.DataFrame]:
    """Rolling pooled ridge regression of forward returns on alphas.

    Z: (T, N, K) alphas, y: (T, N) target (demeaned forward return),
    valid: (T, N) mask of usable observations.
    Returns (forecast (T, N), coefficient history).
    """
    T, N, K = Z.shape
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    zv = valid[..., None] & np.ones(K, dtype=bool)
    Zf = np.where(zv, Zf, 0.0)
    yf = np.where(valid & np.isfinite(y), y, 0.0)
    ok = valid & np.isfinite(y)
    # Per-day sufficient statistics: the pooled regression is a sum over days.
    XtX = np.einsum("tnk,tnj->tkj", Zf * ok[..., None], Zf)
    Xty = np.einsum("tnk,tn->tk", Zf * ok[..., None], yf)
    nobs = ok.sum(axis=1)
    cX, cy, cn = (np.cumsum(a, axis=0) for a in (XtX, Xty, nobs))

    month_ends = pd.Series(np.arange(T), index=dates).groupby(dates.to_period("M")).first()
    refit_pos = month_ends.values  # first trading day of each month
    if start is not None:
        refit_pos = refit_pos[dates[refit_pos] >= start]

    fc = np.full((T, N), np.nan)
    coefs = {}
    for i, m in enumerate(refit_pos):
        hi = m - delay - horizon - embargo      # last label fully known by close m (purged)
        lo = hi - window
        if lo < 0:
            continue
        A = cX[hi] - cX[lo]
        b = cy[hi] - cy[lo]
        n = cn[hi] - cn[lo]
        beta = np.linalg.solve(A + lam * n * np.eye(K), b)
        nxt = refit_pos[i + 1] if i + 1 < len(refit_pos) else T
        fc[m:nxt] = np.einsum("tnk,k->tn", Zf[m:nxt], beta)
        fc[m:nxt][~valid[m:nxt]] = np.nan
        coefs[dates[m]] = beta
    return fc, pd.DataFrame.from_dict(coefs, orient="index")


def equal_weight(Z: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Baseline: average of oriented alpha z-scores."""
    fc = np.nanmean(np.where(np.isfinite(Z), Z, np.nan), axis=-1)
    fc[~valid] = np.nan
    return fc


def daily_ic_matrix(Z: np.ndarray, y: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """(T, K) daily Spearman IC of each alpha against target y."""
    T, N, K = Z.shape
    out = np.full((T, K), np.nan)
    ry = pd.DataFrame(np.where(valid, y, np.nan)).rank(axis=1).to_numpy()
    for k in range(K):
        z = np.where(valid, Z[..., k], np.nan)
        rz = pd.DataFrame(z).rank(axis=1).to_numpy()
        m = np.isfinite(rz) & np.isfinite(ry)
        a = np.where(m, rz, np.nan); b = np.where(m, ry, np.nan)
        a = a - np.nanmean(a, 1, keepdims=True); b = b - np.nanmean(b, 1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = np.nansum(a * b, 1) / np.sqrt(np.nansum(a * a, 1) * np.nansum(b * b, 1))
        ic[m.sum(1) < 30] = np.nan
        out[:, k] = ic
    return out


def walk_forward_ic_weights(Z: np.ndarray, y: np.ndarray, valid: np.ndarray, dates: pd.DatetimeIndex,
                            window: int, horizon: int, delay: int = 1, embargo: int = 5,
                            start: pd.Timestamp | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Monthly re-weighting by trailing (purged) mean IC; negative ICs get 0 weight.

    Forecast = sum_k w_k * z_k with w normalised to sum to 1 (a robust,
    low-variance alternative to regression weights)."""
    T, N, K = Z.shape
    ic = daily_ic_matrix(Z, y, valid)
    Zf = np.where(np.isfinite(Z), Z, 0.0)
    firsts = pd.Series(np.arange(T), index=dates).groupby(dates.to_period("M")).first().values
    if start is not None:
        firsts = firsts[dates[firsts] >= start]
    fc = np.full((T, N), np.nan)
    hist = {}
    for i, m in enumerate(firsts):
        hi = m - delay - horizon - embargo
        lo = hi - window
        if lo < 0:
            continue
        w = np.clip(np.nanmean(ic[lo:hi], axis=0), 0, None)
        w = w / w.sum() if w.sum() > 0 else np.full(K, 1.0 / K)
        nxt = firsts[i + 1] if i + 1 < len(firsts) else T
        fc[m:nxt] = Zf[m:nxt] @ w
        fc[m:nxt][~valid[m:nxt]] = np.nan
        hist[dates[m]] = w
    return fc, pd.DataFrame.from_dict(hist, orient="index")


def grinold_alpha(composite: pd.DataFrame, ic: float, vol: pd.DataFrame) -> pd.DataFrame:
    """Scale-free expected return: alpha_i = IC * sigma_i * zscore_i (Grinold).

    Makes the optimizer's risk/cost trade-off independent of how a particular
    combiner happens to scale its output."""
    z = composite.sub(composite.mean(axis=1), axis=0).div(composite.std(axis=1), axis=0).clip(-3, 3)
    return z * ic * vol
