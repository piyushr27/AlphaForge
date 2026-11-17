"""Stage 1: compute, neutralize and score every alpha in the library.

In-sample (IS)  : 2011-07-01 .. 2018-12-31  -> used for all selection / tuning
Out-of-sample   : 2019-01-01 .. 2026-10-27  -> touched only for final reporting

Outputs
  cache/alphas_neutral/<name>.parquet   neutralized, standardized alphas
  results/alpha_scorecard.csv           IC, ICIR, NW t-stat, Sharpe, turnover
  cache/ls_pnl.parquet                  daily long-short PnL per alpha
"""
from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.simplefilter("ignore")

from alphaforge import alphas as A  # noqa: E402
from alphaforge.data import fwd_returns, load_panel  # noqa: E402
from alphaforge.evaluate import ls_pnl, summarize  # noqa: E402
from alphaforge.neutralize import neutralize, standardize  # noqa: E402

CACHE = ROOT / "cache"
RES = ROOT / "results"
IS = slice("2011-07-01", "2018-12-31")
OOS = slice("2019-01-01", "2026-10-27")
COST_BPS = 5.0

_G = {}


def _init(sector, beta, universe):
    _G.update(sector=sector, beta=beta, universe=universe)


def _neut(item):
    name, raw = item
    n = neutralize(raw, _G["sector"], {"beta": _G["beta"]}, _G["universe"])
    return name, standardize(n).astype("float32")


def main() -> None:
    (CACHE / "alphas_neutral").mkdir(parents=True, exist_ok=True)
    RES.mkdir(exist_ok=True)
    t0 = time.time()
    panel = load_panel()
    ctx = A.Context(panel, pd.read_parquet(ROOT / "data" / "quarterly.parquet"))
    raw = A.compute(ctx)
    print(f"computed {len(raw)} raw alphas in {time.time() - t0:.1f}s")

    beta = ctx.beta
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=10, initializer=_init,
                             initargs=(panel.sector, beta, panel.universe)) as ex:
        neutral = dict(ex.map(_neut, raw.items()))
    print(f"neutralized in {time.time() - t1:.1f}s")
    for n, df in neutral.items():
        df.to_parquet(CACHE / "alphas_neutral" / f"{n}.parquet")

    fwd1 = fwd_returns(panel, 1)
    fwd_h = {h: fwd_returns(panel, h) for h in (1, 2, 5, 10, 20)}
    rows, pnls = [], {}
    for n, s in neutral.items():
        s = s.astype(float)
        r = summarize(s, fwd1, fwd_h, COST_BPS, {"is": IS, "oos": OOS})
        r.update(name=n, family=A.REGISTRY[n].family, doc=A.REGISTRY[n].doc)
        rows.append(r)
        pnls[n] = ls_pnl(s, fwd1)["gross"]
    sc = pd.DataFrame(rows).set_index("name")
    sc.to_csv(RES / "alpha_scorecard.csv", float_format="%.5f")
    pd.DataFrame(pnls).to_parquet(CACHE / "ls_pnl.parquet")
    pd.set_option("display.width", 200)
    print(sc[["family", "is_ic", "is_ic_t", "is_sharpe_gross", "is_turnover",
              "oos_ic", "oos_ic_t", "oos_sharpe_gross"]].round(4).to_string())
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
