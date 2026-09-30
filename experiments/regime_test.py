"""Regime testbed: does a tail-risk budget beat volatility-targeting when vol JUMPS?

This is the experiment motivated by the decisive ablation. On real 2016-2019 NSE data the
volatility regime was persistent, which is exactly where a trailing-vol estimator wins and
reactive tightening has least to add - so the MPC could not be distinguished from a simple
scalar risk control. This script asks the complementary question: in a regime where
volatility JUMPS and then mean-reverts, does the adaptive tail-risk budget hold up better?

TWO data sources, and they carry DIFFERENT weight:
  * `--synthetic` : the block-regime fixture (src/data.make_regime_prices). A MECHANISM
    test - it shows the controller responds as designed. Never reportable.
  * default      : REAL NSE prices, restricted to a real vol-jump window if one exists.

The honest reading is always the real-data row. The synthetic fixture can only ever show
that the machinery works as designed; it cannot show the method is better at anything.
"""
import argparse

import numpy as np
import pandas as pd

from experiments.run_all import build
from src.backtest import run_backtest
from src.data import make_regime_prices, to_returns
from src.metrics import compute_metrics
from src.utils import load_config, set_seed

STRATS = ["equal_weight", "equal_weight_voltarget", "mpc_naive", "mpc_tailbudget",
          "mpc_tailbudget_nofc", "mpc_selfcal"]


def run_all(returns, cfg, start, end, strats):
    mc = cfg["metrics"]
    out, curves = {}, {}
    for name in strats:
        strat = build(name, cfg, returns.shape[1])
        res = run_backtest(returns, strat, start, end, cfg["backtest"]["cost_bps"])
        out[name] = compute_metrics(res, cvar_alpha=mc["cvar_alpha"], risk_free=mc["risk_free"])
        curves[name] = res.net_returns
    return out, curves


def report(title, metrics, curves, returns, phase_marks=None):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    df = pd.DataFrame(metrics).T
    cols = ["sharpe", "ann_return", "ann_vol", "cvar_daily", "max_drawdown",
            "avg_daily_turnover", "total_cost"]
    print(df[cols].round(4).to_string())

    if phase_marks:
        for label, lo, hi in phase_marks:
            print(f"\n--- {label} ({lo.date()} .. {hi.date()}) ---")
            sub = {}
            for name, c in curves.items():
                w = c.loc[lo:hi]
                if len(w) > 5:
                    sub[name] = compute_metrics_from_returns(w, cfg_alpha=0.05)
            s = pd.DataFrame(sub).T
            print(s[["sharpe", "cvar_daily", "max_drawdown"]].round(4).to_string())


def compute_metrics_from_returns(net, cfg_alpha=0.05):
    r = net.values
    eq = np.cumprod(1 + r)
    vol = float(np.std(r, ddof=1)) * np.sqrt(252)
    sharpe = float(np.mean(r) / np.std(r, ddof=1) * np.sqrt(252)) if vol > 0 else 0.0
    dd = float((eq / np.maximum.accumulate(eq) - 1).min())
    q = np.quantile(r, cfg_alpha)
    tail = r[r <= q]
    return {"sharpe": sharpe, "cvar_daily": -float(tail.mean()), "max_drawdown": dd}


def find_real_jump_window(returns, cfg, min_ratio=2.0, search_end=None):
    """Find a real window where realized vol JUMPS by >= min_ratio then mean-reverts.

    Selection rule (fixed before looking at any strategy result, so this is not
    result-shopping): a rolling 20d vol that rises by at least `min_ratio` versus the
    preceding 60d, followed by a 60d decline of at least 30% from the peak.

    `search_end` HARD-BOUNDS the search. This is not optional: the largest vol jump in
    this sample is Dec-2019 -> Mar-2020, i.e. the COVID crash, which lies in the SEALED
    TEST period. Without an explicit bound the selection rule would silently pick the one
    event most likely to flatter the method, from data we agreed never to touch.
    """
    p = returns.mean(axis=1)
    v = p.rolling(20).std() * np.sqrt(252)
    last = len(v) - 140 if search_end is None else search_end
    best = None
    for i in range(60, last):
        pre = float(v.iloc[i - 60:i].mean())
        jump = float(v.iloc[i:i + 20].max())
        if pre <= 0 or jump / pre < min_ratio:
            continue
        peak_i = i + int(np.argmax(v.iloc[i:i + 20].values))
        if peak_i + 60 >= len(v):
            continue
        after = float(v.iloc[peak_i:peak_i + 60].min())
        if after > 0 and after / jump <= 0.7:      # reverted >=30% from the peak
            cand = (float(jump / pre), i, peak_i)
            if best is None or cand[0] > best[0]:
                best = cand
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--synthetic", action="store_true",
                    help="mechanism test on the block-regime fixture (NOT reportable)")
    ap.add_argument("--strategies", nargs="+", default=STRATS)
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    mc = cfg["metrics"]
    bt = cfg["backtest"]

    if a.synthetic:
        print("!! SYNTHETIC block-regime fixture - MECHANISM TEST ONLY, NOT A RESULT")
        prices = make_regime_prices(cfg, seed=cfg["seed"])
        returns = to_returns(prices)
        start, end = returns.index[bt["lookback"]], returns.index[-1]
        metrics, curves = run_all(returns, cfg, start, end, a.strategies)
        # mark the phases from the same block schedule the generator used
        rc = cfg["regime_test"]
        marks, t = [], bt["lookback"]
        phase, cyc = "calm", 0
        while t < len(returns):
            L = {"calm": rc["calm_len"], "spike": rc["spike_len"], "decay": rc["decay_len"]}[phase]
            lo, hi = returns.index[t], returns.index[min(t + L, len(returns)) - 1]
            marks.append((f"cycle {cyc} {phase}", lo, hi))
            t += L
            phase = {"calm": "spike", "spike": "decay", "decay": "calm"}[phase]
            if phase == "calm":
                cyc += 1
            if cyc >= 3:
                break
        report("BLOCK-REGIME MECHANISM TEST (synthetic, not reportable)", metrics, curves,
               returns, marks)
        return

    # ---- real data path ----
    from src.data import download_prices
    d = cfg["data"]
    prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                             min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    returns = to_returns(prices)
    print(f"Real market data: {returns.shape[1]} tickers, "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")

    # HARD BOUND: search (and any resulting window) is clipped to validation end. The
    # biggest jump in the full sample is COVID (Dec-2019 ->), which is sealed test data.
    sp = cfg["splits"]
    val_end = pd.Timestamp(sp["val_end"])
    search_returns = returns.loc[:val_end]
    found = find_real_jump_window(search_returns, cfg,
                                  search_end=len(search_returns) - 140)
    if found is None:
        print("No real vol-jump + mean-revert window in train+val. "
              "The hypothesis needs more data, not a tuned fixture.")
        return
    ratio, i, peak_i = found
    lo = returns.index[max(0, i - 60)]
    hi = returns.index[min(len(returns) - 1, peak_i + 120)]
    print(f"\nReal vol-jump window: {lo.date()} -> {hi.date()}  "
          f"(vol jump x{ratio:.2f}, peak {returns.index[peak_i].date()})")

    metrics, curves = run_all(returns, cfg, lo, hi, a.strategies)
    report(f"REAL NSE, vol-jump window {lo.date()}..{hi.date()}", metrics, curves, returns)
    print("\nNOTE: a single event window is n=1. This is a case study, not evidence.")


if __name__ == "__main__":
    main()
