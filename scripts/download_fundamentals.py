"""Download point-in-time quarterly fundamentals from SEC EDGAR XBRL companyfacts.

Every fact carries the date it was *filed* with the SEC; we key availability on
that date (plus one trading day in the alpha layer) so no fundamental is used
before it was public.

Output: data/fundamentals.parquet with columns
    ticker, concept, end (fiscal period end), filed, val, fp, form
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import requests

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = DATA / "edgar_raw"
UA = {"User-Agent": "Piyush Ranjan f20230684@pilani.bits-pilani.ac.in"}

# concept -> list of us-gaap tags tried in order (companies differ)
CONCEPTS = {
    "eps": ["EarningsPerShareDiluted", "EarningsPerShareBasic"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "assets": ["Assets"],
    "revenue": ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"],
    "op_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
}


def fetch(cik: int) -> dict | None:
    path = RAW / f"{cik:010d}.json"
    if path.exists():
        return json.loads(path.read_text())
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
    for attempt in range(4):
        r = requests.get(url, headers=UA, timeout=30)
        if r.status_code == 200:
            path.write_text(r.text)
            return r.json()
        if r.status_code == 404:
            return None
        time.sleep(2 * (attempt + 1))
    return None


def extract(ticker: str, facts: dict) -> list[dict]:
    gaap = facts.get("facts", {}).get("us-gaap", {})
    rows = []
    for concept, tags in CONCEPTS.items():
        for tag in tags:
            if tag not in gaap:
                continue
            units = gaap[tag]["units"]
            unit = "USD/shares" if concept == "eps" else "USD"
            for f in units.get(unit, []):
                if f.get("form") not in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                    continue
                rows.append({"ticker": ticker, "concept": concept, "tag": tag,
                             "start": f.get("start"), "end": f["end"], "filed": f["filed"],
                             "val": f["val"], "fp": f.get("fp"), "form": f["form"]})
            # no break: filers switch tags over time (e.g. ASC 606 revenue in 2018),
            # so all candidate tags are merged; duplicates are resolved downstream.
    # Shares outstanding lives in the dei taxonomy (cover page of each filing).
    dei = facts.get("facts", {}).get("dei", {}).get("EntityCommonStockSharesOutstanding")
    if dei:
        for f in dei["units"].get("shares", []):
            if f.get("form") in ("10-Q", "10-K", "10-Q/A", "10-K/A"):
                rows.append({"ticker": ticker, "concept": "shares", "tag": "dei",
                             "start": None, "end": f["end"], "filed": f["filed"],
                             "val": f["val"], "fp": f.get("fp"), "form": f["form"]})
    return rows


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    meta = pd.read_parquet(DATA / "meta.parquet")
    rows = []
    for i, (tkr, cik) in enumerate(zip(meta["ticker"], meta["cik"])):
        facts = fetch(int(cik))
        if facts:
            rows.extend(extract(tkr, facts))
        time.sleep(0.12)  # SEC fair-access: <10 req/s
        if i % 50 == 0:
            print(f"{i}/{len(meta)}")
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    df.to_parquet(DATA / "fundamentals.parquet", index=False)
    print(df.shape, df.ticker.nunique(), df.groupby("concept").ticker.nunique().to_dict())


if __name__ == "__main__":
    main()
