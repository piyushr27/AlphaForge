"""Panel construction: long price/fundamental tables -> wide (date x ticker) frames.

Conventions used across AlphaForge
---------------------------------
* Every panel is a DataFrame indexed by trading date with one column per ticker.
* A value at row t uses only information available at the close of day t.
* Alphas are traded with delay 1: a signal computed at close t is entered at
  close t+1 and earns the close(t+1) -> close(t+2) return (see ``fwd_returns``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Panel:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame       # split-adjusted, not dividend-adjusted
    adj_close: pd.DataFrame   # split + dividend adjusted
    volume: pd.DataFrame
    sector: pd.Series         # ticker -> GICS sector
    universe: pd.DataFrame    # bool mask: tradable and in the index at date t

    @property
    def returns(self) -> pd.DataFrame:
        """Total (dividend-inclusive) close-to-close return ending at t."""
        return self.adj_close.pct_change(fill_method=None)

    @property
    def adj_factor(self) -> pd.DataFrame:
        return self.adj_close / self.close

    @property
    def dollar_volume(self) -> pd.DataFrame:
        return self.close * self.volume

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def tickers(self) -> pd.Index:
        return self.close.columns


def load_panel(min_price: float = 5.0, min_history: int = 252) -> Panel:
    px = pd.read_parquet(DATA / "prices.parquet")
    meta = pd.read_parquet(DATA / "meta.parquet").set_index("ticker")
    wide = {c: px.pivot(index="date", columns="ticker", values=c).sort_index()
            for c in ("open", "high", "low", "close", "adj_close", "volume")}
    close = wide["close"]

    # Universe: (1) priced today, (2) at least `min_history` days of history so
    # every lookback is populated, (3) price above $5, (4) already added to the
    # S&P 500 -- avoids look-ahead from using today's constituent list for
    # stocks that only joined later.
    history = close.notna().cumsum()
    added = pd.DataFrame(close.index.values[:, None] >= meta.loc[close.columns, "date_added"].values[None, :],
                         index=close.index, columns=close.columns)
    universe = close.notna() & (history >= min_history) & (close > min_price) & added
    return Panel(**wide, sector=meta.loc[close.columns, "sector"], universe=universe)


def fwd_returns(panel: Panel, horizon: int = 1, delay: int = 1) -> pd.DataFrame:
    """Return earned by a position decided at close t, entered at close t+delay
    and held for `horizon` days. Aligned to row t (the decision date)."""
    adj = panel.adj_close
    return adj.shift(-(delay + horizon)) / adj.shift(-delay) - 1.0


def market_returns(panel: Panel) -> pd.Series:
    """Equal-weighted universe return (the market factor proxy)."""
    r = panel.returns.where(panel.universe.shift(1, fill_value=False))
    return r.mean(axis=1)


def load_spy() -> pd.Series:
    path = DATA / "spy.parquet"
    if not path.exists():
        import yfinance as yf
        s = yf.download("SPY", start="2010-01-01", end="2026-11-01", auto_adjust=False, progress=False)
        s = s["Adj Close"].squeeze().rename("SPY")
        s.to_frame().to_parquet(path)
    return pd.read_parquet(path)["SPY"].pct_change(fill_method=None)


def to_numpy_safe(df: pd.DataFrame) -> np.ndarray:
    return df.to_numpy(dtype=float, na_value=np.nan)
