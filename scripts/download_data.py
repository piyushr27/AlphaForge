"""Download S&P 500 constituents + daily OHLCV from public sources.

Universe: current S&P 500 constituents (Wikipedia) with GICS sector and
"Date added". Prices: Yahoo Finance via yfinance, unadjusted OHLCV plus
Adj Close (split + dividend adjusted).

Known limitation (documented in README): stocks that left the index are not
included, so the universe carries survivorship bias. We mitigate look-ahead on
*additions* by only admitting a stock to the universe after its "Date added".

Output: data/prices.parquet (long format), data/meta.parquet
"""
from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf

DATA = Path(__file__).resolve().parents[1] / "data"
START, END = "2010-01-01", "2026-11-01"
UA = {"User-Agent": "Mozilla/5.0 (AlphaForge research; f20230684@pilani.bits-pilani.ac.in)"}


def load_constituents() -> pd.DataFrame:
    r = requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=UA, timeout=30)
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[0]
    meta = pd.DataFrame({
        "ticker": t["Symbol"].str.replace(".", "-", regex=False),  # BRK.B -> BRK-B for Yahoo
        "name": t["Security"],
        "sector": t["GICS Sector"],
        "industry": t["GICS Sub-Industry"],
        "date_added": pd.to_datetime(t["Date added"], errors="coerce"),
        "cik": t["CIK"].astype(int),
    })
    return meta


def download(tickers: list[str], chunk: int = 50) -> pd.DataFrame:
    frames = []
    for i in range(0, len(tickers), chunk):
        batch = tickers[i:i + chunk]
        for attempt in range(4):
            try:
                d = yf.download(batch, start=START, end=END, auto_adjust=False,
                                progress=False, threads=True, group_by="column")
                break
            except Exception as e:  # rate limit / transient
                print(f"retry {attempt} for batch {i}: {e}")
                time.sleep(10 * (attempt + 1))
        else:
            raise RuntimeError(f"batch {i} failed")
        long = d.stack(level="Ticker", future_stack=True).reset_index()
        long.columns = [str(c).lower().replace(" ", "_") for c in long.columns]
        frames.append(long.dropna(subset=["close"]))
        print(f"{i + len(batch)}/{len(tickers)} tickers")
        time.sleep(2)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    meta = load_constituents()
    px = download(meta["ticker"].tolist())
    px = px.rename(columns={"price": "date"}) if "price" in px.columns else px
    px["date"] = pd.to_datetime(px["date"])
    missing = sorted(set(meta["ticker"]) - set(px["ticker"]))
    print("missing tickers:", missing)
    px.to_parquet(DATA / "prices.parquet", index=False)
    meta.to_parquet(DATA / "meta.parquet", index=False)
    print(px.shape, px["date"].min(), px["date"].max(), px["ticker"].nunique())


if __name__ == "__main__":
    main()
