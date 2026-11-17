"""The alpha library.

Every alpha is a function ``f(ctx) -> DataFrame`` returning a raw signal at
close t using only data available by close t (higher = expected to outperform).
Signals are registered with a family label so evaluation can report results
by hypothesis rather than by parameter variant.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import Callable

import numpy as np
import pandas as pd

from . import fundamentals as F
from .data import Panel, market_returns
from .ops import (cs_rank, ts_beta, ts_corr, ts_max, ts_mean, ts_rank, ts_skew,
                  ts_std, ts_sum)


@dataclass
class Context:
    """Lazily computed building blocks shared by all alphas."""
    panel: Panel
    quarterly: pd.DataFrame | None = None
    _cache: dict = field(default_factory=dict)

    @cached_property
    def ret(self) -> pd.DataFrame:
        return self.panel.returns

    @cached_property
    def logret(self) -> pd.DataFrame:
        return np.log1p(self.ret)

    @cached_property
    def mkt(self) -> pd.Series:
        return market_returns(self.panel)

    @cached_property
    def beta(self) -> pd.DataFrame:
        return ts_beta(self.ret, self.mkt, 252)

    @cached_property
    def resid(self) -> pd.DataFrame:
        """Market-model residual return (beta-hedged)."""
        return self.ret - self.beta.shift(1).mul(self.mkt, axis=0)

    @cached_property
    def adv20(self) -> pd.DataFrame:
        return ts_mean(self.panel.dollar_volume, 20)

    @cached_property
    def overnight(self) -> pd.DataFrame:
        """Close(t-1) -> open(t) return, dividend/split adjusted."""
        p, f = self.panel, self.panel.adj_factor
        return (p.open * f) / (p.close * f).shift(1) - 1.0

    @cached_property
    def intraday(self) -> pd.DataFrame:
        p = self.panel
        return p.close / p.open - 1.0

    def fund(self, concept: str) -> pd.DataFrame:
        if concept not in self._cache:
            if self.quarterly is None:
                raise ValueError("fundamental alpha requested without quarterly data")
            self._cache[concept] = F.to_daily(self.quarterly, concept, self.panel.dates, self.panel.tickers)
        return self._cache[concept]

    @cached_property
    def mcap(self) -> pd.DataFrame:
        return self.panel.close * self.fund("shares")

    @cached_property
    def filing_day(self) -> pd.DataFrame:
        return F.filing_dates(self.quarterly, self.panel.dates, self.panel.tickers)


@dataclass(frozen=True)
class AlphaSpec:
    name: str
    family: str
    fn: Callable[[Context], pd.DataFrame]
    doc: str


REGISTRY: dict[str, AlphaSpec] = {}


def alpha(name: str, family: str, doc: str):
    def deco(fn):
        REGISTRY[name] = AlphaSpec(name, family, fn, doc)
        return fn
    return deco


# ---------------------------------------------------------------------------
# Price / volume alphas
# ---------------------------------------------------------------------------
for _k in (1, 3, 5, 10, 21):
    alpha(f"rev_{_k}d", "reversal", f"-{_k}-day return: short-term reversal")(
        (lambda k: lambda c: -ts_sum(c.logret, k))(_k))
    alpha(f"resid_rev_{_k}d", "reversal", f"-{_k}-day beta-hedged residual return")(
        (lambda k: lambda c: -ts_sum(c.resid, k))(_k))

for _lb, _skip in ((252, 21), (126, 21), (63, 5)):
    alpha(f"mom_{_lb}_{_skip}", "momentum", f"return from t-{_lb} to t-{_skip}")(
        (lambda lb, sk: lambda c: ts_sum(c.logret, lb - sk).shift(sk))(_lb, _skip))

alpha("resid_mom_252_21", "momentum", "residual momentum (12-1 on beta-hedged returns)")(
    lambda c: ts_sum(c.resid, 231).shift(21) / ts_std(c.resid, 231).shift(21))


@alpha("high_52w", "momentum", "price / 52-week high (George-Hwang anchoring)")
def _high52(c):
    a = c.panel.adj_close
    return a / ts_max(a, 252)


for _n in (20, 60):
    alpha(f"lowvol_{_n}d", "risk", f"-{_n}-day return volatility (low-vol anomaly)")(
        (lambda n: lambda c: -ts_std(c.ret, n))(_n))

alpha("idio_vol_60d", "risk", "-60-day residual volatility")(lambda c: -ts_std(c.resid, 60))
# Note: betting-against-beta (-beta) is omitted: after beta-neutralization it is
# identically zero up to noise.
alpha("max_21d", "risk", "-max daily return over 21 days (lottery demand)")(lambda c: -ts_max(c.ret, 21))
alpha("skew_60d", "risk", "-60-day return skewness")(lambda c: -ts_skew(c.ret, 60))


@alpha("vol_shock_rev", "volume", "reversal scaled by abnormal volume: -ret_5d * adv5/adv60")
def _volshock(c):
    v = c.panel.volume
    return -ts_sum(c.logret, 5) * (ts_mean(v, 5) / ts_mean(v, 60))


@alpha("vol_trend", "volume", "-log(adv20 / adv120): fading volume attention")
def _voltrend(c):
    v = c.panel.dollar_volume
    return -np.log(ts_mean(v, 20) / ts_mean(v, 120))


@alpha("vol_tsrank", "volume", "-time-series rank of today's volume in 60 days")
def _voltsr(c):
    return -ts_rank(c.panel.volume, 60)


@alpha("pv_corr_10d", "volume", "-10-day correlation of price and volume")
def _pvcorr(c):
    return -ts_corr(c.panel.close, c.panel.volume, 10)


@alpha("amihud_20d", "liquidity", "Amihud illiquidity: mean |ret| / dollar volume")
def _amihud(c):
    return np.log(ts_mean(c.ret.abs() / c.panel.dollar_volume, 20))


@alpha("hl_range_10d", "liquidity", "-mean (high-low)/close over 10 days")
def _hl(c):
    p = c.panel
    return -ts_mean((p.high - p.low) / p.close, 10)


@alpha("clv_rev_5d", "intraday", "-mean close-location-value over 5 days")
def _clv(c):
    p = c.panel
    rng = (p.high - p.low).replace(0, np.nan)
    return -ts_mean(((p.close - p.low) - (p.high - p.close)) / rng, 5)


alpha("overnight_rev_5d", "intraday", "-sum of overnight returns, 5 days")(
    lambda c: -ts_sum(np.log1p(c.overnight), 5))
alpha("intraday_rev_5d", "intraday", "-sum of open-to-close returns, 5 days")(
    lambda c: -ts_sum(np.log1p(c.intraday), 5))
alpha("overnight_mom_21d", "intraday", "sum of overnight returns, 21 days (overnight persistence)")(
    lambda c: ts_sum(np.log1p(c.overnight), 21))


@alpha("ma_dist_20d", "reversal", "-(price / 20-day MA - 1)")
def _ma20(c):
    a = c.panel.adj_close
    return -(a / ts_mean(a, 20) - 1)


@alpha("ma_cross_50_200", "momentum", "50-day MA / 200-day MA - 1 (trend)")
def _ma_cross(c):
    a = c.panel.adj_close
    return ts_mean(a, 50) / ts_mean(a, 200) - 1


@alpha("rsi_14_rev", "reversal", "-(RSI(14) - 50)")
def _rsi(c):
    r = c.ret
    up, dn = ts_mean(r.clip(lower=0), 14), ts_mean((-r).clip(lower=0), 14)
    return -(100 - 100 / (1 + up / dn.replace(0, np.nan)) - 50)


# ---------------------------------------------------------------------------
# Fundamental alphas (point-in-time EDGAR data)
# ---------------------------------------------------------------------------
alpha("sue_eps", "earnings", "standardized unexpected EPS (seasonal random walk)")(lambda c: c.fund("eps_sue"))
alpha("sue_revenue", "earnings", "standardized unexpected revenue")(lambda c: c.fund("revenue_sue"))
alpha("sue_net_income", "earnings", "standardized unexpected net income")(lambda c: c.fund("net_income_sue"))


@alpha("eps_yoy_price", "earnings", "YoY EPS change / price")
def _epsyoy(c):
    return c.fund("eps_yoy") / c.panel.close


@alpha("earn_ann_ret", "earnings", "3-day return around the latest EPS filing, held ~1 quarter (PEAD)")
def _ear(c):
    # filing_day marks t0 = first trading day after the SEC filing date. The
    # reaction window is [t0-1, t0+1]; it is complete (and known) at close t0+1.
    r3 = c.resid.rolling(3, min_periods=3).sum()
    known = c.filing_day.shift(1, fill_value=False)
    return r3.where(known).ffill(limit=60)


@alpha("earnings_yield", "value", "TTM EPS / price")
def _ey(c):
    return c.fund("eps_ttm") / c.panel.close


@alpha("book_to_price", "value", "book equity / market cap")
def _bp(c):
    return c.fund("equity") / c.mcap


@alpha("sales_to_price", "value", "TTM revenue / market cap")
def _sp(c):
    return c.fund("revenue_ttm") / c.mcap


@alpha("cfo_to_price", "value", "TTM operating cash flow / market cap")
def _cfp(c):
    return c.fund("op_cash_flow_ttm") / c.mcap


@alpha("roe", "quality", "TTM net income / book equity")
def _roe(c):
    eq = c.fund("equity")
    return c.fund("net_income_ttm") / eq.where(eq > 0)


@alpha("roa", "quality", "TTM net income / total assets")
def _roa(c):
    return c.fund("net_income_ttm") / c.fund("assets")


@alpha("accruals", "quality", "-(TTM net income - TTM CFO) / assets (Sloan accruals)")
def _accr(c):
    return -(c.fund("net_income_ttm") - c.fund("op_cash_flow_ttm")) / c.fund("assets")


@alpha("cfo_to_assets", "quality", "TTM operating cash flow / assets")
def _cfoa(c):
    return c.fund("op_cash_flow_ttm") / c.fund("assets")


alpha("asset_growth", "investment", "-YoY total asset growth")(lambda c: -c.fund("assets_growth"))


@alpha("sales_growth", "growth", "YoY revenue change / TTM revenue")
def _sg(c):
    return c.fund("revenue_yoy") / c.fund("revenue_ttm").abs()


alpha("size", "size", "-log market cap")(lambda c: -np.log(c.mcap))


def compute(ctx: Context, names: list[str] | None = None) -> dict[str, pd.DataFrame]:
    names = names or list(REGISTRY)
    out = {}
    for n in names:
        sig = REGISTRY[n].fn(ctx)
        out[n] = sig.replace([np.inf, -np.inf], np.nan).where(ctx.panel.universe)
    return out
