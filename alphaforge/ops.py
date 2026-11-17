"""Cross-sectional and time-series operators on (date x ticker) panels."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mp(n: int, lo: int = 2) -> int:
    """min_periods: 80% of the window, at least `lo`, never more than n."""
    return min(n, max(lo, int(n * 0.8)))


def ts_sum(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 2)).sum()


def ts_mean(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 2)).mean()


def ts_std(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 2)).std()


def ts_max(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 2)).max()


def ts_skew(x: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 3)).skew()


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, n: int) -> pd.DataFrame:
    return x.rolling(n, min_periods=_mp(n, 3)).corr(y)


def ts_rank(x: pd.DataFrame, n: int) -> pd.DataFrame:
    """Percentile of today's value within its trailing n-day window."""
    return x.rolling(n, min_periods=_mp(n, 2)).rank(pct=True)


def ts_beta(y: pd.DataFrame, x: pd.Series, n: int) -> pd.DataFrame:
    """Rolling OLS beta of each column of y on the series x."""
    mp = _mp(n, 3)
    xb = x.rolling(n, min_periods=mp).mean()
    yb = y.rolling(n, min_periods=mp).mean()
    xy = y.mul(x, axis=0).rolling(n, min_periods=mp).mean()
    xx = (x * x).rolling(n, min_periods=mp).mean()
    return (xy.sub(yb.mul(xb, axis=0))).div(xx - xb * xb, axis=0)


def cs_rank(x: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional rank scaled to [-0.5, 0.5]."""
    return x.rank(axis=1, pct=True).sub(0.5)


def cs_zscore(x: pd.DataFrame) -> pd.DataFrame:
    return x.sub(x.mean(axis=1), axis=0).div(x.std(axis=1), axis=0)


def cs_winsorize(x: pd.DataFrame, k: float = 3.0) -> pd.DataFrame:
    """Clip each row at mean +- k standard deviations."""
    m, s = x.mean(axis=1), x.std(axis=1)
    return x.clip(lower=m - k * s, upper=m + k * s, axis=0)
