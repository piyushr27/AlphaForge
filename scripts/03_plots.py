"""Stage 3: charts for the README (equity curve, IC decay of selected alphas)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"
pnl = pd.read_csv(RES / "portfolio_pnl.csv", index_col=0, parse_dates=True).loc["2014-01-01":]
fig, ax = plt.subplots(figsize=(9, 4))
for c, lab in (("gross", "gross"), ("net", "net of 5 bps one-way")):
    ax.plot(pnl[c].cumsum(), label=lab)
ax.axvline(pd.Timestamp("2019-01-01"), color="k", ls="--", lw=1)
ax.text(pd.Timestamp("2019-02-01"), ax.get_ylim()[1] * 0.9, "out-of-sample →")
ax.set_title("AlphaForge optimized market-neutral portfolio (cumulative return on gross book)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(RES / "equity_curve.png", dpi=120)

sc = pd.read_csv(RES / "alpha_scorecard.csv", index_col=0)
sel = json.loads((RES / "summary.json").read_text())["selected_alphas"]
dec = sc.loc[sel, [c for c in sc.columns if c.startswith("ic_h")]]
dec.columns = [int(c[4:]) for c in dec.columns]
fig, ax = plt.subplots(figsize=(9, 4))
(dec.T * 100).plot(ax=ax, marker="o")
ax.set_xlabel("horizon (days)"); ax.set_ylabel("in-sample rank IC (%)")
ax.set_title("IC by forecast horizon, selected alphas (in-sample)"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(RES / "ic_decay.png", dpi=120)
print("saved")
