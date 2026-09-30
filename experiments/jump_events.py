"""Per-event volatility-jump analysis, hard-bounded to the validation period.

WHY THIS SCRIPT EXISTS
----------------------
`experiments/regime_test.py` reports the SINGLE largest real jump-and-revert window
and correctly labels it n=1. A single event cannot support a claim. This script
enumerates EVERY qualifying event in train+validation and reports each strategy's
tail risk in that window, so the "does the MPC win when volatility jumps?" question
gets 5 observations instead of 1.

    python -m experiments.jump_events
    -> results/jump_events.csv

SEALED-PERIOD GUARD (do not remove)
-----------------------------------
`find_real_jump_window` in regime_test.py finds only the best event. Here we
enumerate, so the guard has to be re-implemented explicitly rather than inherited:
`search_end` is clipped to the validation end and asserted below. The largest
volatility jump in the full sample is x6.94 (Dec-2019 -> Mar-2020, the COVID crash)
and it lies in the SEALED test period. An unbounded search would silently select the
single event most likely to flatter the method, from data we agreed never to touch.
`tests/test_regime_test.py::test_jump_search_never_reaches_past_the_bound` guards the
sibling function; `tests/test_jump_events.py` guards this one.

SELECTION RULE (fixed before any strategy result was inspected)
--------------------------------------------------------------
A rolling 20-day equal-weight volatility that rises by at least `min_ratio` versus
the preceding 60-day mean, then falls at least 30% from its peak within 60 days.
Consecutive detections are DE-CLUSTERED with a cooldown so one long crisis is not
counted as several independent events.

This is a mechanism test on a handful of events. It is not a performance claim and
no p-value is computed; see the paper for the matched-information bootstrap that
does carry inference.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.regime_test import compute_metrics_from_returns
from experiments.run_all import build
from src.backtest import run_backtest
from src.data import download_prices, to_returns
from src.utils import load_config, set_seed

STRATS = [
    "equal_weight_voltarget",
    "equal_weight_voltarget_fwd",
    "mpc_tailbudget_nofc",
    "mpc_tailbudget_fwdvol",
]


def find_jump_events(returns, min_ratio=2.0, search_end=None, cooldown=140):
    """Enumerate every vol-jump-then-revert event, de-clustered, hard-bounded.

    `search_end` is an INDEX bound (positional), not a date. Clipping happens in
    main() against the validation end; the assertion here is a backstop so this
    function cannot be called unbounded by accident.
    """
    p = returns.mean(axis=1)
    v = (p.rolling(20).std() * np.sqrt(252)).to_numpy()
    n = len(v)
    last = n - 140 if search_end is None else int(search_end)
    if last > n:
        raise ValueError("search_end beyond data length")
    hits = []
    for i in range(60, last):
        pre = float(np.nanmean(v[i - 60:i]))
        jump = float(np.nanmax(v[i:i + 20]))
        if not np.isfinite(pre) or not np.isfinite(jump) or pre <= 0:
            continue
        if jump / pre < min_ratio:
            continue
        seg = v[i:i + 20]
        peak_i = i + int(np.nanargmax(seg))
        if peak_i + 60 >= n:
            continue
        after = float(np.nanmin(v[peak_i:peak_i + 60]))
        if not np.isfinite(after) or after <= 0 or after / jump > 0.7:
            continue
        if hits and peak_i - hits[-1][1] < cooldown:      # de-cluster
            continue
        hits.append((float(jump / pre), i, peak_i))
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--min-ratio", type=float, default=2.0)
    ap.add_argument("--cooldown", type=int, default=140)
    ap.add_argument("--out", default="results/jump_events.csv")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    d, sp = cfg["data"], cfg["splits"]
    prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                             min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    returns = to_returns(prices)

    val_end = pd.Timestamp(sp["val_end"])
    search = returns.loc[:val_end]
    print(f"Real market data: {returns.shape[1]} tickers, {returns.index[0].date()} -> "
          f"{returns.index[-1].date()}")
    print(f"SEARCH HARD-BOUNDED to validation end: {val_end.date()} "
          f"(sealed test period and later are NOT searched)")

    events = find_jump_events(search, a.min_ratio, len(search) - 140, a.cooldown)
    if not events:
        raise SystemExit("no qualifying jump events in train+val")
    print(f"\n{len(events)} de-clustered jump events (min_ratio={a.min_ratio}, "
          f"cooldown={a.cooldown}d):")
    for ratio, i, pk in events:
        print(f"  {search.index[pk].date()}  jump x{ratio:.2f}")

    rows = []
    for ratio, i, pk in events:
        lo = search.index[i]
        hi_idx = min(pk + 60, len(search) - 1)
        hi = search.index[hi_idx]
        row = {"event": str(search.index[pk].date()), "jump_ratio": round(ratio, 3),
               "window_start": str(lo.date()), "window_end": str(hi.date())}
        for s in STRATS:
            res = run_backtest(returns, build(s, cfg, returns.shape[1]), lo, hi,
                               cfg["backtest"]["cost_bps"])
            m = compute_metrics_from_returns(res.net_returns, cfg["metrics"]["cvar_alpha"])
            row[f"{s}_cvar"] = m["cvar_daily"]
            row[f"{s}_dd"] = m["max_drawdown"]
            row[f"{s}_sharpe"] = m["sharpe"]
        rows.append(row)

    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)

    vt = df["equal_weight_voltarget_cvar"].to_numpy()
    vf = df["equal_weight_voltarget_fwd_cvar"].to_numpy()
    nf = df["mpc_tailbudget_nofc_cvar"].to_numpy()
    fw = df["mpc_tailbudget_fwdvol_cvar"].to_numpy()
    n = len(df)
    print(f"\n=== WIN COUNTS over {n} events (lower CVaR magnitude = better) ===")
    print(f"  forward-vol scalar beats trailing scalar : {int((vf < vt).sum())}/{n}")
    print(f"  MPC (no fc)      beats forward-vol scalar: {int((nf < vf).sum())}/{n}")
    print(f"  MPC (forward vol) beats forward-vol scalar: {int((fw < vf).sum())}/{n}")
    print(f"\n  mean CVaR, MPC (no fc)   - matched scalar: {nf.mean() - vf.mean():+.5f}")
    print(f"  mean CVaR, MPC (fwd vol) - matched scalar: {fw.mean() - vf.mean():+.5f}")
    print(f"\nSaved to {a.out}")
    print("NOTE: n is small. This is a mechanism test, not a performance claim.")


if __name__ == "__main__":
    main()
