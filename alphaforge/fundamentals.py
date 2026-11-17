"""Point-in-time fundamentals from SEC EDGAR XBRL facts.

Rules that keep this look-ahead free:
* For each (ticker, concept, fiscal period) the *first filed* value is used.
  Later restatements and comparative re-reports are ignored.
* A value becomes usable on the first trading day strictly after its SEC
  ``filed`` date (filings often land after the close).
* Q4 flows are not reported directly; they are derived as FY - 9M YTD and
  become available on the 10-K filing date.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DATA

FLOW = ("eps", "net_income", "revenue", "op_cash_flow")
STOCK = ("equity", "assets", "shares")


def _quarterly_flows(f: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (ticker, concept, quarter end) with a 3-month value.

    Filers report flows either as 3-month values or as fiscal-year-to-date
    cumulatives (3M, 6M, 9M, FY sharing a start date). Quarters are taken
    directly when reported, otherwise derived as the difference of two
    consecutive YTD cumulatives; the derived value is available on the later
    filing date. Per-share EPS differences are approximate (share counts drift).
    """
    f = f.copy()
    f["dur"] = (f["end"] - f["start"]).dt.days
    first = (f.sort_values("filed")
               .drop_duplicates(["ticker", "concept", "start", "end"], keep="first"))
    direct = first[first["dur"].between(80, 100)][["ticker", "concept", "end", "filed", "val"]]

    ytd = first[first["dur"].between(80, 380)].sort_values(["ticker", "concept", "start", "end"])
    g = ytd.groupby(["ticker", "concept", "start"], sort=False)
    prev_end, prev_val, prev_filed = g["end"].shift(), g["val"].shift(), g["filed"].shift()
    gap = (ytd["end"] - prev_end).dt.days
    ok = gap.between(80, 100)
    derived = pd.DataFrame({"ticker": ytd["ticker"], "concept": ytd["concept"], "end": ytd["end"],
                            "filed": np.maximum(ytd["filed"], prev_filed), "val": ytd["val"] - prev_val})[ok]
    out = pd.concat([direct.assign(pri=0), derived.assign(pri=1)], ignore_index=True)
    # Prefer a directly reported quarter; then the earliest filing.
    out = out.sort_values(["pri", "filed"]).drop_duplicates(["ticker", "concept", "end"], keep="first")
    return out.drop(columns="pri")


def _stocks(f: pd.DataFrame) -> pd.DataFrame:
    s = f[f["concept"].isin(STOCK)]
    return (s.sort_values("filed")
             .drop_duplicates(["ticker", "concept", "end"], keep="first")
             [["ticker", "concept", "end", "filed", "val"]])


def load_quarterly() -> pd.DataFrame:
    f = pd.read_parquet(DATA / "fundamentals.parquet")
    q = pd.concat([_quarterly_flows(f[f["concept"].isin(FLOW)]), _stocks(f)], ignore_index=True)
    return q.sort_values(["ticker", "concept", "end"]).reset_index(drop=True)


def _derive_one(end: np.ndarray, filed: np.ndarray, val: np.ndarray):
    """TTM, YoY change and SUE for one quarterly series sorted by fiscal end.

    A window only counts if its quarters are consecutive (75-105 day gaps).
    Availability = latest filing date among the quarters used.
    """
    n = len(end)
    gaps = np.diff(end).astype("timedelta64[D]").astype(int)
    consec = np.r_[False, (gaps >= 75) & (gaps <= 105)]
    run = np.zeros(n, dtype=int)  # length of consecutive run ending at i
    for i in range(n):
        run[i] = run[i - 1] + 1 if (i > 0 and consec[i]) else 1
    ttm = np.full(n, np.nan); yoy = np.full(n, np.nan); sue = np.full(n, np.nan)
    f_ttm = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    f_yoy = f_ttm.copy()
    for i in range(n):
        if run[i] >= 4:
            ttm[i] = val[i - 3:i + 1].sum(); f_ttm[i] = filed[i - 3:i + 1].max()
        if run[i] >= 5:
            yoy[i] = val[i] - val[i - 4]; f_yoy[i] = max(filed[i], filed[i - 4])
    for i in range(n):
        # SUE: this quarter's YoY change scaled by the std of the previous 8.
        past = yoy[max(0, i - 8):i]
        past = past[~np.isnan(past)]
        if not np.isnan(yoy[i]) and len(past) >= 6 and past.std(ddof=1) > 0:
            sue[i] = yoy[i] / past.std(ddof=1)
    return (ttm, f_ttm), (yoy, f_yoy), (sue, f_yoy)


def add_derived(q: pd.DataFrame) -> pd.DataFrame:
    """Append `<flow>_ttm`, `<flow>_yoy`, `<flow>_sue` rows and `<stock>_yoy`
    (year-over-year % change of balance-sheet items) to the quarterly table."""
    rows = []
    for (tkr, concept), g in q.groupby(["ticker", "concept"], sort=False):
        g = g.sort_values("end")
        end, filed, val = g["end"].values, g["filed"].values, g["val"].values.astype(float)
        if concept in FLOW:
            for suffix, (v, fd) in zip(("ttm", "yoy", "sue"), _derive_one(end, filed, val)):
                ok = ~np.isnan(v)
                rows.append(pd.DataFrame({"ticker": tkr, "concept": f"{concept}_{suffix}",
                                          "end": end[ok], "filed": fd[ok], "val": v[ok]}))
        elif concept in ("assets",):
            # growth vs the value ~1 year earlier (350-380 days before)
            s = pd.Series(val, index=pd.DatetimeIndex(end))
            prev = s.reindex(s.index - pd.Timedelta(days=365), method="nearest",
                             tolerance=pd.Timedelta(days=20)).values
            pf = pd.Series(filed, index=pd.DatetimeIndex(end)).reindex(
                s.index - pd.Timedelta(days=365), method="nearest", tolerance=pd.Timedelta(days=20)).values
            with np.errstate(divide="ignore", invalid="ignore"):
                growth = val / prev - 1.0
            ok = np.isfinite(growth) & (prev > 0)
            rows.append(pd.DataFrame({"ticker": tkr, "concept": f"{concept}_growth",
                                      "end": end[ok], "filed": np.maximum(filed[ok], pf[ok]),
                                      "val": growth[ok]}))
    return pd.concat([q] + rows, ignore_index=True)


def to_daily(q: pd.DataFrame, concept: str, dates: pd.DatetimeIndex, tickers: pd.Index,
             max_stale_days: int = 200) -> pd.DataFrame:
    """Daily panel of the latest *known* value of `concept` on each date.

    Value filed on day d is first usable on the next trading day after d.
    Values older than `max_stale_days` trading days are dropped.
    """
    s = q[q["concept"] == concept]
    s = s[s["ticker"].isin(tickers)]
    # Among filings known by date t, use the one with the latest fiscal end.
    s = s.sort_values(["ticker", "filed", "end"])
    s = s[s.groupby("ticker")["end"].cummax() == s["end"]]
    pos = dates.searchsorted(s["filed"].values, side="right")  # first date > filed
    s = s.assign(pos=pos)
    s = s[s["pos"] < len(dates)]
    s = s.drop_duplicates(["ticker", "pos"], keep="last")
    out = pd.DataFrame(np.nan, index=dates, columns=tickers)
    arr = out.to_numpy()
    col = tickers.get_indexer(s["ticker"])
    arr[s["pos"].values, col] = s["val"].values
    out = pd.DataFrame(arr, index=dates, columns=tickers)
    return out.ffill(limit=max_stale_days)


def filing_dates(q: pd.DataFrame, dates: pd.DatetimeIndex, tickers: pd.Index) -> pd.DataFrame:
    """Boolean panel: True on the first trading day after an EPS filing."""
    s = q[(q["concept"] == "eps") & q["ticker"].isin(tickers)]
    pos = dates.searchsorted(s["filed"].values, side="right")
    keep = pos < len(dates)
    out = np.zeros((len(dates), len(tickers)), dtype=bool)
    out[pos[keep], tickers.get_indexer(s["ticker"].values[keep])] = True
    return pd.DataFrame(out, index=dates, columns=tickers)
