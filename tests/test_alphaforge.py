"""Correctness tests: look-ahead, neutralization, IC, point-in-time fundamentals,
purged walk-forward and optimizer constraints. Synthetic data only, so the
suite runs without the downloaded dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from alphaforge import alphas as A
from alphaforge import fundamentals as F
from alphaforge.combine import select_alphas, walk_forward_ic_weights, walk_forward_ridge
from alphaforge.data import Panel, fwd_returns
from alphaforge.evaluate import ls_weights, rank_ic, turnover
from alphaforge.neutralize import neutralize
from alphaforge.portfolio import DailyOptimizer, OptConfig


def synthetic_panel(T=400, N=60, seed=0) -> Panel:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    tick = [f"S{i:02d}" for i in range(N)]
    r = rng.normal(0.0003, 0.015, (T, N))
    close = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), dates, tick)
    adj = close * np.linspace(0.9, 1.0, T)[:, None]  # dividend adjustment drifting to 1
    opn = close.shift(1).fillna(close) * (1 + rng.normal(0, 0.003, (T, N)))
    high = np.maximum(close, opn) * 1.01
    low = np.minimum(close, opn) * 0.99
    vol = pd.DataFrame(rng.lognormal(13, 0.3, (T, N)), dates, tick)
    sector = pd.Series([f"sec{i % 5}" for i in range(N)], index=tick)
    uni = pd.DataFrame(True, dates, tick)
    return Panel(opn, high, low, close, adj, vol, sector, uni)


PRICE_ALPHAS = [n for n, s in A.REGISTRY.items()
                if s.family not in ("earnings", "value", "quality", "investment", "growth", "size")]


@pytest.mark.parametrize("name", PRICE_ALPHAS)
def test_alphas_have_no_lookahead(name):
    """An alpha's value at date t must not change when data after t is removed."""
    p = synthetic_panel()
    full = A.compute(A.Context(p), [name])[name]
    cut = 300
    trunc = Panel(*(getattr(p, f).iloc[:cut] for f in ("open", "high", "low", "close", "adj_close", "volume")),
                  p.sector, p.universe.iloc[:cut])
    part = A.compute(A.Context(trunc), [name])[name]
    pd.testing.assert_frame_equal(full.iloc[:cut], part, check_freq=False, atol=1e-10)


def test_fwd_returns_alignment():
    p = synthetic_panel()
    f = fwd_returns(p, horizon=1, delay=1)
    a = p.adj_close
    t = 100
    expected = a.iloc[t + 2] / a.iloc[t + 1] - 1
    np.testing.assert_allclose(f.iloc[t].values, expected.values)
    assert f.iloc[-2:].isna().all().all()


def test_neutralize_is_orthogonal_to_sectors_and_beta():
    p = synthetic_panel()
    rng = np.random.default_rng(1)
    beta = pd.DataFrame(rng.normal(1, 0.3, p.close.shape), p.dates, p.tickers)
    sig = pd.DataFrame(rng.normal(size=p.close.shape), p.dates, p.tickers) + 2 * beta
    sig.iloc[:, :12] += 5  # sector-ish bias on some names
    res = neutralize(sig, p.sector, {"beta": beta})
    d = pd.get_dummies(p.sector).to_numpy(float)
    for t in (10, 200):
        r = res.iloc[t].to_numpy()
        np.testing.assert_allclose(d.T @ r, 0, atol=1e-9)
        b = beta.iloc[t].to_numpy()
        np.testing.assert_allclose(((b - b.mean()) / b.std()) @ r, 0, atol=1e-8)


def test_rank_ic_matches_scipy():
    p = synthetic_panel()
    rng = np.random.default_rng(2)
    s = pd.DataFrame(rng.normal(size=p.close.shape), p.dates, p.tickers)
    f = fwd_returns(p)
    s.iloc[50, :5] = np.nan
    ic = rank_ic(s, f)
    for t in (10, 50):
        m = s.iloc[t].notna() & f.iloc[t].notna()
        assert ic.iloc[t] == pytest.approx(spearmanr(s.iloc[t][m], f.iloc[t][m])[0], abs=1e-12)


def test_ls_weights_dollar_neutral_unit_gross():
    p = synthetic_panel()
    s = pd.DataFrame(np.random.default_rng(3).normal(size=p.close.shape), p.dates, p.tickers)
    w = ls_weights(s)
    np.testing.assert_allclose(w.sum(axis=1), 0, atol=1e-12)
    np.testing.assert_allclose(w.abs().sum(axis=1), 1, atol=1e-12)
    assert (turnover(w).iloc[1:] > 0).all()


def test_quarterly_q4_is_fy_minus_9m_and_uses_first_filing():
    d = pd.to_datetime
    f = pd.DataFrame([
        # FY2020 cumulative facts (start 2020-01-01)
        dict(ticker="X", concept="net_income", start=d("2020-01-01"), end=d("2020-03-31"), filed=d("2020-05-01"), val=10.0),
        dict(ticker="X", concept="net_income", start=d("2020-01-01"), end=d("2020-06-30"), filed=d("2020-08-01"), val=25.0),
        dict(ticker="X", concept="net_income", start=d("2020-01-01"), end=d("2020-09-30"), filed=d("2020-11-01"), val=45.0),
        dict(ticker="X", concept="net_income", start=d("2020-01-01"), end=d("2020-12-31"), filed=d("2021-02-15"), val=70.0),
        # later restatement of Q1 must be ignored (first filing wins)
        dict(ticker="X", concept="net_income", start=d("2020-01-01"), end=d("2020-03-31"), filed=d("2021-05-01"), val=99.0),
    ])
    q = F._quarterly_flows(f).set_index("end")["val"].sort_index()
    assert q.tolist() == [10.0, 15.0, 20.0, 25.0]
    filed = F._quarterly_flows(f).set_index("end")["filed"]
    assert filed[d("2020-12-31")] == d("2021-02-15")


def test_fundamental_daily_panel_is_point_in_time():
    dates = pd.bdate_range("2021-01-01", periods=60)
    q = pd.DataFrame([dict(ticker="X", concept="eps", end=pd.Timestamp("2020-12-31"),
                           filed=dates[20], val=1.5)])
    daily = F.to_daily(q, "eps", dates, pd.Index(["X"]))
    assert daily["X"].iloc[:21].isna().all()       # not usable on the filing day itself
    assert daily["X"].iloc[21] == 1.5               # usable the next trading day


def test_sue_requires_consecutive_quarters():
    end = pd.date_range("2015-03-31", periods=14, freq="QE").values
    filed = end + np.timedelta64(30, "D")
    val = np.arange(14, dtype=float) + np.random.default_rng(0).normal(0, 0.1, 14)
    (ttm, _), (yoy, _), (sue, _) = F._derive_one(end, filed, val)
    assert np.isnan(ttm[:3]).all() and np.isfinite(ttm[3])
    assert np.isnan(yoy[:4]).all() and yoy[4] == pytest.approx(val[4] - val[0])
    end2 = end.copy(); end2[7:] = end2[7:] + np.timedelta64(200, "D")  # gap breaks the run
    (ttm2, _), _, _ = F._derive_one(end2, filed, val)
    assert np.isnan(ttm2[7:10]).all()


def _wf_inputs(T=500, N=40, K=3, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2015-01-01", periods=T)
    Z = rng.normal(size=(T, N, K))
    y = 0.1 * Z[..., 0] + rng.normal(size=(T, N))
    return Z, y, np.ones((T, N), bool), dates


@pytest.mark.parametrize("fn", ["ridge", "ic"])
def test_walk_forward_is_purged(fn):
    """Perturbing labels inside the purge gap or the future must not change a forecast."""
    Z, y, valid, dates = _wf_inputs()
    h, delay, emb = 5, 1, 5
    run = (lambda yy: walk_forward_ridge(Z, yy, valid, dates, 1.0, 200, h, delay, emb)[0]) if fn == "ridge" else \
          (lambda yy: walk_forward_ic_weights(Z, yy, valid, dates, 200, h, delay, emb)[0])
    base = run(y)
    m = int(np.argmax(np.isfinite(base[:, 0])))          # first forecast day
    # The last training label is day m-delay-h-emb (its return window closes at
    # m-emb). Everything after it is inside the purge gap or the future.
    first_forbidden = m - delay - h - emb + 1
    y2 = y.copy()
    y2[first_forbidden:] = 1e6 * np.random.default_rng(9).normal(size=y2[first_forbidden:].shape)
    pert = run(y2)
    nxt = m + 1
    while nxt < len(dates) and dates[nxt].month == dates[m].month:
        nxt += 1
    np.testing.assert_allclose(base[m:nxt], pert[m:nxt])


def test_walk_forward_ridge_recovers_signal():
    Z, y, valid, dates = _wf_inputs(T=800)
    _, coefs = walk_forward_ridge(Z, y, valid, dates, 0.0, 400, 1)
    assert coefs.iloc[-1, 0] == pytest.approx(0.1, abs=0.02)
    assert abs(coefs.iloc[-1, 1]) < 0.02


def test_select_alphas_drops_correlated_duplicates():
    idx = pd.bdate_range("2015-01-01", periods=300)
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=300), idx)
    pnl = {"a": a, "a_dup": a + 0.01 * rng.normal(size=300), "b": pd.Series(rng.normal(size=300), idx)}
    sc = pd.DataFrame({"is_ic": [0.01, 0.01, -0.01], "is_ic_t": [5.0, 4.0, -3.0]}, index=["a", "a_dup", "b"])
    chosen, sign = select_alphas(sc, pnl, slice(None))
    assert chosen == ["a", "b"] and sign["b"] == -1


def test_optimizer_respects_constraints():
    n = 120
    rng = np.random.default_rng(0)
    sectors = np.array([f"s{i % 6}" for i in range(n)])
    cfg = OptConfig(gamma=5, max_pos=0.03, max_turnover=0.3)
    L = rng.normal(0, 0.005, (n, 3))
    opt = DailyOptimizer(n, sectors, cfg, n_factors=3)
    tradable = rng.random(n) > 0.1
    beta = rng.normal(1, 0.3, n)
    w = opt.solve(rng.normal(0, 1e-3, n), np.full(n, 0.015), beta, tradable, np.zeros(n), L)
    assert abs(w.sum()) < 1e-6
    assert np.abs(w).sum() <= 1 + 1e-6
    assert np.abs(w).max() <= 0.03 + 1e-6
    assert np.all(w[~tradable] == 0)
    assert abs(np.where(tradable, beta, 0) @ w) <= cfg.beta_tol + 1e-6
    S = pd.get_dummies(pd.Series(sectors)).to_numpy(float)
    assert np.abs(S.T @ w).max() <= cfg.sector_tol + 1e-6
    w2 = opt.solve(rng.normal(0, 1e-3, n), np.full(n, 0.015), beta, tradable, w, L)
    assert np.abs(w2 - w).sum() <= 0.3 + 1e-6
