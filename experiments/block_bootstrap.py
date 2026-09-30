"""Step 1: stationary block bootstrap over the REAL validation returns.

  python -m experiments.block_bootstrap

Produces, for every strategy:
  - the point estimate and a 95% bootstrap interval for Sharpe, CVaR 5% and max drawdown
  - PAIRED difference intervals against every reference strategy

and writes results/<timestamp>-val-bootstrap/{metrics.csv, marginal.csv, paired.csv}.
The test period is never touched; see docs/stress_definition.md for the label rules used
by the other two steps.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.run_all import ALL, build
from src.backtest import run_backtest
from src.bootstrap import (bootstrap_metrics, paired_difference_ci, percentile_ci)
from src.data import download_prices, period_bounds, to_returns
from src.metrics import set_trading_days
from src.utils import load_config, make_run_dir, save_config, set_seed

METRICS = ["sharpe", "cvar_daily", "max_drawdown"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategies", nargs="+", default=ALL)
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    set_trading_days(cfg["backtest"]["trading_days"])
    d = cfg["data"]
    prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                             min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    returns = to_returns(prices)
    print(f"Real market data: {returns.shape[1]} tickers, "
          f"{returns.index[0].date()} -> {returns.index[-1].date()}")

    start, end = period_bounds(cfg["splits"], "val", returns.index)
    if start > returns.index[-1]:
        raise SystemExit("validation window is empty")
    # Defensive: refuse to operate on anything at/after the test boundary.
    test_start = end + pd.Timedelta(days=1)
    if returns.index.max() < test_start:
        raise SystemExit("data does not even reach the test period; refusing")

    run_dir = make_run_dir(tag="val-bootstrap")
    save_config({**cfg, "cli": vars(a), "bootstrap_window": [str(start), str(end)]}, run_dir)
    print(f"Bootstrap window (validation only): {start.date()} -> {end.date()}")

    # --- point-estimate backtests -------------------------------------------
    series = {}
    for name in a.strategies:
        res = run_backtest(returns, build(name, cfg, returns.shape[1]), start, end,
                           cfg["backtest"]["cost_bps"])
        series[name] = res.net_returns
        print(f"  ran {name:24s} {len(res.net_returns)} days")
    # cache the daily net-return series so interval re-analysis never needs a re-backtest
    pd.DataFrame(series).to_csv(run_dir / "daily_net_returns.csv")

    # --- bootstrap ----------------------------------------------------------
    bs = cfg["bootstrap"]
    mc = cfg["metrics"]
    print(f"\nBootstrap: mean block {bs['mean_block_days']}d, {bs['n_resamples']} resamples, "
          f"seed {cfg['seed']}")
    point, dist = bootstrap_metrics(
        series, bs["mean_block_days"], bs["n_resamples"], cfg["seed"],
        trading_days=cfg["backtest"]["trading_days"], risk_free=mc["risk_free"],
        cvar_alpha=mc["cvar_alpha"])
    conf = bs["confidence"]

    marg_rows, pair_rows = [], []
    for name in a.strategies:
        for m in METRICS:
            lo, hi = percentile_ci(dist[name][m], conf)
            marg_rows.append({"strategy": name, "metric": m, "point": point[name][m],
                              "ci_low": lo, "ci_high": hi, "confidence": conf,
                              "n_resamples": bs["n_resamples"]})
    marginal = pd.DataFrame(marg_rows)

    refs = [r for r in bs["reference_strategies"] if r in a.strategies]
    for name in a.strategies:
        for ref in refs:
            if name == ref:
                continue
            for m in METRICS:
                st = paired_difference_ci(dist[name][m], dist[ref][m], conf)
                pair_rows.append({"strategy": name, "reference": ref, "metric": m,
                                  "point_diff": point[name][m] - point[ref][m], **st})
    paired = pd.DataFrame(pair_rows)

    marginal.to_csv(run_dir / "marginal.csv", index=False)
    if len(paired):
        paired.to_csv(run_dir / "paired.csv", index=False)

    # --- report -------------------------------------------------------------
    print("\n=== MARGINAL 95% BOOTSTRAP INTERVALS (validation, real NSE) ===")
    print(f"{'strategy':24s} {'metric':13s} {'point':>9s} {'lo':>9s} {'hi':>9s}")
    for _, r in marginal.iterrows():
        print(f"{r['strategy']:24s} {r['metric']:13s} {r['point']:9.3f} "
              f"{r['ci_low']:9.3f} {r['ci_high']:9.3f}")

    for ref in refs:
        sub = paired[paired["reference"] == ref] if len(paired) else pd.DataFrame()
        if not len(sub):
            continue
        print(f"\n=== PAIRED DIFFERENCES vs {ref} ===")
        print(f"{'strategy':24s} {'metric':13s} {'diff':>9s} {'lo':>9s} {'hi':>9s} "
              f"{'p':>7s} {'sig':>4s}")
        for _, r in sub.iterrows():
            print(f"{r['strategy']:24s} {r['metric']:13s} {r['mean_diff']:9.3f} "
                  f"{r['ci_low']:9.3f} {r['ci_high']:9.3f} {r['p_value']:7.3f} "
                  f"{'YES' if r['excludes_zero'] else '-':>4s}")

    # --- figure: interval plot for Sharpe ----------------------------------
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(a.strategies) + 2))
    sm = marginal[marginal["metric"] == "sharpe"].set_index("strategy")
    ys = range(len(sm))
    for y, name in zip(ys, sm.index):
        r = sm.loc[name]
        ax.plot([r["ci_low"], r["ci_high"]], [y, y], color="#2b6cb0", lw=2.5, solid_capstyle="round")
        ax.plot([r["point"]], [y], "o", color="#1a365d", ms=5)
    ax.axvline(0.0, color="#a0aec0", lw=1, ls="--")
    ax.set_yticks(list(ys)); ax.set_yticklabels(sm.index, fontsize=8.5)
    ax.set_xlabel("Sharpe (95% stationary block bootstrap interval)")
    ax.set_title(f"Validation {start.date()} to {end.date()} - {bs['n_resamples']} resamples, "
                 f"mean block {bs['mean_block_days']}d", fontsize=9)
    ax.grid(axis="x", alpha=0.25); fig.tight_layout()
    fig.savefig(run_dir / "sharpe_intervals.png", dpi=150); plt.close(fig)

    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()