"""Stage 2: select alphas, combine walk-forward, build optimized portfolio.

All choices (which alphas, ridge lambda, window, label horizon, risk aversion,
turnover penalty) are made on the in-sample period. The out-of-sample period
2019-01 .. 2026-10 is evaluated once with the frozen configuration.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from alphaforge import alphas as A  # noqa: E402
from alphaforge.combine import (equal_weight, grinold_alpha, select_alphas, stack,
                                walk_forward_ic_weights, walk_forward_ridge)  # noqa: E402
from alphaforge.data import fwd_returns, load_panel, load_spy  # noqa: E402
from alphaforge.evaluate import ls_pnl, max_drawdown, rank_ic, sharpe  # noqa: E402
from alphaforge.ops import ts_std  # noqa: E402
from alphaforge.portfolio import OptConfig, pca_risk_model, run_optimizer  # noqa: E402

CACHE, RES = ROOT / "cache", ROOT / "results"
IS = slice("2011-07-01", "2018-12-31")
OOS = slice("2019-01-01", "2026-10-27")
COMBO_START = pd.Timestamp("2013-07-01")   # first forecast month (needs 2y training)
TUNE = slice("2014-01-01", "2018-12-31")   # IS window used for tuning
COST_BPS = 5.0


def target(panel, h: int) -> np.ndarray:
    """Cross-sectionally demeaned, winsorized h-day forward return (delay 1)."""
    f = fwd_returns(panel, h).where(panel.universe)
    f = f.sub(f.mean(axis=1), axis=0)
    lo, hi = f.quantile(0.01, axis=1), f.quantile(0.99, axis=1)
    return f.clip(lo, hi, axis=0).to_numpy()


def perf(pnl: pd.Series, spy: pd.Series | None = None) -> dict:
    pnl = pnl.dropna()
    d = {"ann_return": pnl.mean() * 252, "ann_vol": pnl.std() * np.sqrt(252),
         "sharpe": sharpe(pnl), "max_dd": max_drawdown(pnl),
         "hit_rate": (pnl > 0).mean(), "days": len(pnl)}
    if spy is not None:
        # PnL row t is realised on day t+2 (delay 1, close t+1 -> t+2).
        realised = spy.shift(-2).reindex(pnl.index)
        d["corr_spy"] = pnl.corr(realised)
        d["beta_spy"] = np.cov(pnl, realised.fillna(0))[0, 1] / realised.var()
    return d


def main() -> None:
    t0 = time.time()
    panel = load_panel()
    ctx = A.Context(panel)
    dates, cols = panel.dates, panel.tickers
    sc = pd.read_csv(RES / "alpha_scorecard.csv", index_col=0)
    sc = sc.drop(index=[n for n in sc.index if n not in A.REGISTRY])
    pnls = pd.read_parquet(CACHE / "ls_pnl.parquet")[sc.index]
    alphas = {n: pd.read_parquet(CACHE / "alphas_neutral" / f"{n}.parquet").astype(float) for n in sc.index}

    # ---- 1. selection (IS only) -------------------------------------------
    chosen, sign = select_alphas(sc, pnls, IS, min_abs_t=2.0, max_corr=0.5)
    print(f"library {len(sc)} alphas; {(sc.is_ic_t.abs() >= 2).sum()} pass |t|>=2; selected {len(chosen)}:")
    print("  ", [(c, sign[c]) for c in chosen])
    corr_sel = (pnls[chosen].loc[IS] * pd.Series(sign)[chosen]).corr()
    offdiag = corr_sel.where(~np.eye(len(chosen), dtype=bool)).abs()

    # ---- 2. walk-forward combination: choose combiner + params on IS -------
    Z = stack(alphas, chosen, sign)
    valid = panel.universe.to_numpy() & (np.isfinite(Z).sum(-1) >= len(chosen) // 2)
    fwd1 = fwd_returns(panel, 1)
    targets = {h: target(panel, h) for h in (1, 5, 10)}
    cands = {("equal_weight", 0, 0, 0.0): equal_weight(Z, valid)}
    for h, win in itertools.product((1, 5, 10), (252, 504, 756)):
        cands[("ic_weight", h, win, 0.0)] = walk_forward_ic_weights(Z, targets[h], valid, dates, win, h, start=COMBO_START)[0]
        for lam in (0.0, 1.0, 10.0):
            cands[("ridge", h, win, lam)] = walk_forward_ridge(Z, targets[h], valid, dates, lam, win, h, start=COMBO_START)[0]
    grid = []
    for key, fc in cands.items():
        fcd = pd.DataFrame(fc, index=dates, columns=cols)
        p = ls_pnl(fcd, fwd1, COST_BPS).loc[TUNE]
        grid.append(dict(combiner=key[0], h=key[1], window=key[2], lam=key[3],
                         is_ic=rank_ic(fcd, fwd1).loc[TUNE].mean(), is_gross_sharpe=sharpe(p.gross),
                         is_net_sharpe=sharpe(p.net), is_turnover=p.turnover.mean()))
    grid = pd.DataFrame(grid).sort_values("is_gross_sharpe", ascending=False)
    grid.to_csv(RES / "combo_tuning_is.csv", index=False, float_format="%.5f")
    print("combo tuning (IS, top 8):\n", grid.head(8).round(4).to_string(index=False))
    best = grid.iloc[0]
    key = (best.combiner, int(best.h), int(best.window), float(best.lam))
    composite = pd.DataFrame(cands[key], index=dates, columns=cols)
    if best.combiner == "ic_weight":
        _, wh = walk_forward_ic_weights(Z, targets[key[1]], valid, dates, key[2], key[1], start=COMBO_START)
        wh.columns = chosen
        wh.to_csv(RES / "combo_weights.csv", float_format="%.5f")
    ew = pd.DataFrame(equal_weight(Z, valid), index=dates, columns=cols)
    ridge_best = grid[grid.combiner == "ridge"].iloc[0]
    ridge = pd.DataFrame(cands[("ridge", int(ridge_best.h), int(ridge_best.window), float(ridge_best.lam))],
                         index=dates, columns=cols)
    # Grinold scaling with the composite's IS one-day IC: alpha = IC * sigma * z.
    idio = ts_std(ctx.resid, 60)
    comp_ic = float(rank_ic(composite, fwd1).loc[TUNE].mean())
    forecast = grinold_alpha(composite, comp_ic, idio)

    # ---- 3. optimizer: tune gamma / turnover penalty on IS -----------------
    beta = ctx.beta
    tradable = panel.universe
    opt_grid = []
    # Turnover is controlled by a hard daily cap; the true 5bps one-way cost is
    # charged in PnL. (A linear cost penalty in a single-period objective
    # dominates a ~1bp/day alpha and the book never gets invested -- see
    # results/v2_penalty_grinold.)
    fm = pca_risk_model(ctx.ret, panel.universe, k=10)
    warm = dates[(dates >= COMBO_START) & (dates <= TUNE.stop)]
    for gamma, cap in itertools.product((1.0, 10.0, 50.0), (0.05, 0.10)):
        cfg = OptConfig(gamma=gamma, cost_bps=COST_BPS, tc_mult=0.0, max_turnover=cap)
        _, p = run_optimizer(forecast, idio, beta, fwd1, tradable, panel.sector, cfg, warm, fm)
        p = p.loc[TUNE]
        opt_grid.append(dict(gamma=gamma, max_turnover=cap, is_net_sharpe=sharpe(p.net),
                             is_gross_sharpe=sharpe(p.gross), is_turnover=p.turnover.mean(),
                             is_gross_exposure=p.gross_exposure.mean()))
        print("  opt", {k: round(float(v), 4) for k, v in opt_grid[-1].items()})
    opt_grid = pd.DataFrame(opt_grid).sort_values("is_net_sharpe", ascending=False)
    opt_grid.to_csv(RES / "optimizer_tuning_is.csv", index=False, float_format="%.5f")
    ob = opt_grid.iloc[0]
    cfg = OptConfig(gamma=float(ob.gamma), cost_bps=COST_BPS, tc_mult=0.0, max_turnover=float(ob.max_turnover))

    # ---- 4. frozen config: full run, report IS and OOS ---------------------
    run_dates = dates[(dates >= COMBO_START) & (dates <= OOS.stop)]  # 6-month warm-up before IS tuning window
    W, pnl = run_optimizer(forecast, idio, beta, fwd1, tradable, panel.sector, cfg, run_dates, fm)
    W.astype("float32").to_parquet(RES / "portfolio_weights.parquet")
    pnl.to_csv(RES / "portfolio_pnl.csv", float_format="%.7f")

    spy = load_spy()
    exp_beta = (W * beta.reindex_like(W)).sum(axis=1)
    out = {"selected_alphas": chosen, "alpha_signs": {c: sign[c] for c in chosen},
           "n_library": len(sc), "n_pass_t2": int((sc.is_ic_t.abs() >= 2).sum()),
           "selected_max_abs_pnl_corr_is": float(offdiag.max().max()),
           "selected_mean_abs_pnl_corr_is": float(offdiag.stack().mean()),
           "combo": {"combiner": key[0], "horizon": key[1], "window": key[2], "lambda": key[3],
                     "embargo_days": 5, "refit": "monthly", "composite_is_ic_1d": comp_ic},
           "optimizer": cfg.__dict__, "cost_bps_one_way": COST_BPS}
    for label, sl in (("is", TUNE), ("oos", OOS)):
        p = pnl.loc[sl]
        out[f"opt_{label}_net"] = perf(p.net, spy)
        out[f"opt_{label}_gross"] = perf(p.gross, spy)
        out[f"opt_{label}_turnover"] = float(p.turnover.mean())
        out[f"opt_{label}_gross_exposure"] = float(p.gross_exposure.mean())
        out[f"opt_{label}_avg_names"] = float((p.n_long + p.n_short).mean())
        out[f"opt_{label}_mean_abs_beta_exposure"] = float(exp_beta.loc[sl].abs().mean())
        for nm, f in (("selected_combo", composite), ("ridge", ridge), ("equal_weight", ew)):
            lp = ls_pnl(f, fwd1, COST_BPS).loc[sl]
            out[f"{nm}_ls_{label}"] = {"ic": float(rank_ic(f, fwd1).loc[sl].mean()),
                                       "sharpe_gross": sharpe(lp.gross), "sharpe_net": sharpe(lp.net),
                                       "turnover": float(lp.turnover.mean())}
    yearly = pnl.loc[TUNE.start:].groupby(pnl.loc[TUNE.start:].index.year).agg(
        net_return=("net", lambda x: x.sum()), net_sharpe=("net", sharpe),
        gross_sharpe=("gross", sharpe), turnover=("turnover", "mean"))
    yearly.to_csv(RES / "portfolio_yearly.csv", float_format="%.4f")
    (RES / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    print(json.dumps(out, indent=2, default=lambda x: round(float(x), 4)))
    print(yearly.round(3).to_string())
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
