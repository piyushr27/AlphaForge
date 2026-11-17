"""Daily mean-variance portfolio optimizer (cvxpy).

Each day solves

    max  a'w  - gamma * sum_i s_i^2 w_i^2  - kappa * ||w - w_prev||_1
    s.t. sum(w) = 0                       (dollar neutral)
         ||w||_1 <= 1                     (gross exposure = 1x book)
         |beta'w| <= beta_tol             (market neutral)
         |S'w|   <= sector_tol            (sector neutral, per GICS sector)
         -ub <= w <= ub                   (position cap, 0 outside universe)
         ||w - w_prev||_1 <= max_turnover (optional hard turnover cap)

a     : expected next-day return (combined forecast)
L     : (n x k) statistical factor loadings scaled by sqrt(eigenvalue), so
        L L' is the top-k PCA part of the 252-day return covariance (refit monthly)
s_i   : idiosyncratic daily vol (60-day residual std)
kappa : one-way transaction cost per unit traded (bps * 1e-4)

The problem is written with cvxpy Parameters (DPP) so it is compiled once and
re-solved daily with new data.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd


@dataclass
class OptConfig:
    gamma: float = 5.0          # risk aversion (on idio variance)
    cost_bps: float = 5.0       # one-way cost used in the turnover penalty
    tc_mult: float = 1.0        # extra penalty multiplier on costs
    max_pos: float = 0.02       # |w_i| cap (fraction of gross book)
    beta_tol: float = 0.01
    sector_tol: float = 0.01
    max_turnover: float | None = None  # hard cap on daily one-way turnover (fraction of book)


class DailyOptimizer:
    def __init__(self, n: int, sectors: np.ndarray, cfg: OptConfig, n_factors: int = 0):
        self.cfg = cfg
        S = pd.get_dummies(pd.Series(sectors)).to_numpy(dtype=float)  # n x n_sec
        self.w = cp.Variable(n)
        self.a = cp.Parameter(n)
        self.risk = cp.Parameter(n, nonneg=True)   # sqrt(gamma) * s_i
        self.w_prev = cp.Parameter(n)
        self.beta = cp.Parameter(n)
        self.ub = cp.Parameter(n, nonneg=True)
        self.k = n_factors
        self.L = cp.Parameter((n, n_factors)) if n_factors else None
        kappa = cfg.cost_bps * 1e-4 * cfg.tc_mult
        obj = (self.a @ self.w
               - cp.sum_squares(cp.multiply(self.risk, self.w))
               - (cfg.gamma * cp.sum_squares(self.L.T @ self.w) if n_factors else 0)
               - kappa * cp.norm1(self.w - self.w_prev))
        cons = [cp.sum(self.w) == 0,
                cp.norm1(self.w) <= 1.0,
                cp.abs(self.beta @ self.w) <= cfg.beta_tol,
                cp.abs(S.T @ self.w) <= cfg.sector_tol,
                cp.abs(self.w) <= self.ub]
        if cfg.max_turnover is not None:
            cons.append(cp.norm1(self.w - self.w_prev) <= cfg.max_turnover)
        self.prob = cp.Problem(cp.Maximize(obj), cons)

    def solve(self, a, s, beta, tradable, w_prev, L=None) -> np.ndarray:
        cfg = self.cfg
        if self.k:
            self.L.value = np.where(tradable[:, None], L, 0.0)
        self.a.value = np.where(tradable, np.nan_to_num(a), 0.0)
        self.risk.value = np.sqrt(cfg.gamma) * np.where(tradable, np.nan_to_num(s, nan=0.03), 0.0)
        self.beta.value = np.where(tradable, np.nan_to_num(beta, nan=1.0), 0.0)
        self.ub.value = np.where(tradable, cfg.max_pos, 0.0)
        self.w_prev.value = w_prev
        try:
            self.prob.solve(solver=cp.CLARABEL, warm_start=True)
        except cp.SolverError:
            return np.where(tradable, w_prev, 0.0)
        if self.w.value is None or self.prob.status not in ("optimal", "optimal_inaccurate"):
            # Infeasible (e.g. a stock left the universe and the turnover cap
            # cannot absorb it): retry without the cap, else hold.
            if self.cfg.max_turnover is not None and not getattr(self, "_retry", False):
                self._retry = True
                try:
                    return self._solve_uncapped(w_prev, tradable)
                finally:
                    self._retry = False
            return np.where(tradable, w_prev, 0.0)
        w = self.w.value.copy()
        w[np.abs(w) < 1e-6] = 0.0
        return w

    def _solve_uncapped(self, w_prev, tradable):
        if not hasattr(self, "_prob_nocap"):
            cons = [c for c in self.prob.constraints[:-1]]
            self._prob_nocap = cp.Problem(self.prob.objective, cons)
        self._prob_nocap.solve(solver=cp.CLARABEL)
        if self.w.value is None:
            return np.where(tradable, w_prev, 0.0)
        w = self.w.value.copy()
        w[np.abs(w) < 1e-6] = 0.0
        return w


def pca_risk_model(returns: pd.DataFrame, universe: pd.DataFrame, k: int = 10,
                   window: int = 252) -> dict[pd.Timestamp, np.ndarray]:
    """Monthly statistical factor model: top-k eigenvectors of the trailing
    `window`-day covariance of daily returns (missing -> 0), scaled by
    sqrt(eigenvalue). Fit on the first trading day of each month using data
    strictly before that day. Returns {refit date: (n x k) loadings}."""
    R = returns.to_numpy(dtype=float)
    U = universe.to_numpy()
    dates = returns.index
    firsts = pd.Series(np.arange(len(dates)), index=dates).groupby(dates.to_period("M")).first().values
    out = {}
    for m in firsts:
        if m < window:
            continue
        X = np.where(U[m - window:m] & np.isfinite(R[m - window:m]), R[m - window:m], 0.0)
        X = X - X.mean(axis=0)
        C = X.T @ X / (window - 1)
        vals, vecs = np.linalg.eigh(C)
        idx = np.argsort(vals)[::-1][:k]
        out[dates[m]] = vecs[:, idx] * np.sqrt(np.clip(vals[idx], 0, None))
    return out


def run_optimizer(forecast: pd.DataFrame, idio_vol: pd.DataFrame, beta: pd.DataFrame,
                  fwd1: pd.DataFrame, tradable: pd.DataFrame, sectors: pd.Series,
                  cfg: OptConfig, dates: pd.DatetimeIndex,
                  factor_model: dict[pd.Timestamp, np.ndarray] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the daily optimizer over `dates`.

    Row t weights are decided with information up to close t and traded at
    close t+1 (delay 1); they earn fwd1[t] (close t+1 -> close t+2). Between
    rebalances holdings drift with realised returns before the next trade.
    Returns (weights, daily pnl frame).
    """
    cols = forecast.columns
    k = next(iter(factor_model.values())).shape[1] if factor_model else 0
    opt = DailyOptimizer(len(cols), sectors.reindex(cols).fillna("Unknown").to_numpy(), cfg, k)
    fm_dates = pd.DatetimeIndex(sorted(factor_model)) if factor_model else None
    F, V, B, R, M = (x.reindex(index=dates, columns=cols).to_numpy(dtype=float)
                     for x in (forecast, idio_vol, beta, fwd1, tradable.astype(float)))
    W = np.zeros((len(dates), len(cols)))
    rows = []
    w_drift = np.zeros(len(cols))
    for t in range(len(dates)):
        trad = (M[t] > 0) & np.isfinite(F[t]) & np.isfinite(R[t])
        L = None
        if k:
            j = fm_dates.searchsorted(dates[t], side="right") - 1  # latest fit <= today
            L = factor_model[fm_dates[j]]
        w = opt.solve(F[t], V[t], B[t], trad, w_drift, L)
        trade = np.abs(w - w_drift).sum()
        r = np.nan_to_num(R[t])
        gross = float(w @ r)
        cost = trade * cfg.cost_bps * 1e-4
        W[t] = w
        rows.append((dates[t], gross, gross - cost, trade, np.abs(w).sum(), (w > 0).sum(), (w < 0).sum()))
        # Drift weights with realised returns (book normalised to 1).
        w_drift = w * (1 + r) / (1 + gross)
    pnl = pd.DataFrame(rows, columns=["date", "gross", "net", "turnover", "gross_exposure",
                                      "n_long", "n_short"]).set_index("date")
    return pd.DataFrame(W, index=dates, columns=cols), pnl
