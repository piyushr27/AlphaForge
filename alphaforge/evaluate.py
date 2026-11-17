"""Alpha evaluation: information coefficient, decay, turnover, long-short PnL."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

ANN = 252


def _row_rank(x: np.ndarray) -> np.ndarray:
    return pd.DataFrame(x).rank(axis=1).to_numpy()


def rank_ic(signal: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily Spearman correlation between signal and forward return."""
    s = signal.to_numpy(dtype=float)
    r = fwd.reindex_like(signal).to_numpy(dtype=float)
    mask = np.isfinite(s) & np.isfinite(r)
    s = np.where(mask, s, np.nan)
    r = np.where(mask, r, np.nan)
    rs, rr = _row_rank(s), _row_rank(r)
    rs = rs - np.nanmean(rs, axis=1, keepdims=True)
    rr = rr - np.nanmean(rr, axis=1, keepdims=True)
    num = np.nansum(rs * rr, axis=1)
    den = np.sqrt(np.nansum(rs ** 2, axis=1) * np.nansum(rr ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = num / den
    ic[mask.sum(axis=1) < 30] = np.nan
    return pd.Series(ic, index=signal.index)


def newey_west_t(x: pd.Series, lags: int = 10) -> float:
    """t-statistic of the mean with HAC (Newey-West) standard errors.
    Needed because overlapping-horizon ICs and persistent signals are autocorrelated."""
    x = x.dropna()
    if len(x) < 50:
        return np.nan
    res = sm.OLS(x.values, np.ones(len(x))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def ls_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """Dollar-neutral weights proportional to the demeaned signal, gross = 1."""
    w = signal.sub(signal.mean(axis=1), axis=0)
    return w.div(w.abs().sum(axis=1), axis=0).fillna(0.0)


def turnover(w: pd.DataFrame) -> pd.Series:
    """Daily one-way turnover as a fraction of gross book: sum |w_t - w_{t-1}|."""
    return w.diff().abs().sum(axis=1)


def ls_pnl(signal: pd.DataFrame, fwd1: pd.DataFrame, cost_bps: float = 0.0) -> pd.DataFrame:
    """Daily PnL (return on gross book) of the long-short portfolio.

    `fwd1` must be the delay-1, horizon-1 forward return aligned to the
    decision date, so row t PnL is realised on day t+2."""
    w = ls_weights(signal)
    gross = (w * fwd1.reindex_like(w)).sum(axis=1, min_count=1)
    to = turnover(w)
    net = gross - to * cost_bps * 1e-4
    return pd.DataFrame({"gross": gross, "net": net, "turnover": to})


def sharpe(x: pd.Series) -> float:
    x = x.dropna()
    return float(np.sqrt(ANN) * x.mean() / x.std()) if x.std() > 0 else np.nan


def max_drawdown(x: pd.Series) -> float:
    eq = x.fillna(0).cumsum()
    return float((eq - eq.cummax()).min())


def summarize(signal: pd.DataFrame, fwd1: pd.DataFrame, fwd_h: dict[int, pd.DataFrame],
              cost_bps: float, periods: dict[str, slice]) -> dict:
    """One row of the alpha scorecard (per period)."""
    ic = rank_ic(signal, fwd1)
    pnl = ls_pnl(signal, fwd1, cost_bps)
    out = {}
    for p, sl in periods.items():
        i, q = ic.loc[sl], pnl.loc[sl]
        out[f"{p}_ic"] = i.mean()
        out[f"{p}_icir"] = i.mean() / i.std() * np.sqrt(ANN)
        out[f"{p}_ic_t"] = newey_west_t(i)
        out[f"{p}_sharpe_gross"] = sharpe(q["gross"])
        out[f"{p}_sharpe_net"] = sharpe(q["net"])
        out[f"{p}_turnover"] = q["turnover"].mean()
    is_sl = next(iter(periods.values()))
    for h, f in fwd_h.items():
        out[f"ic_h{h}"] = rank_ic(signal, f).loc[is_sl].mean()
    return out
