"""EXPOSURE-MATCHED comparison: is the MPC's edge real, or is it just de-risking?

WHY THIS EXPERIMENT
-------------------
Across the validation period the MPC beats the matched scalar volatility-targeting rule in
exactly ONE calendar year (2016) and wins 2 of 5 volatility-jump events. Every one of those
cases looks like the same thing: a year the market fell, in which the MPC's mean exposure was
far below the baseline's. In 2016 the MPC returned -5.4% at 5.9% vol against the baseline's
-0.5% at 10.1% vol - i.e. it "won" the tail metric while holding mostly cash.

That is the confound that has to be removed before anything else can be claimed. A strategy
that simply holds less risk will always look better on a downside-tail statistic and worse on
return and Sharpe. Those are the same decision, not evidence of skill.

METHOD
------
For each strategy we (1) record its realized mean gross exposure, (2) rescale its DAILY
NET returns to a common target exposure, and (3) recompute every risk/return metric at that
matched exposure. Comparing at matched exposure is the standard way to ask "given the same
amount of risk taken, which allocation is better?"

The rescaling is an ANALYTICAL normalization, not a tradeable backtest: it assumes a cash
account at 0% return and scales the strategy's return series. Turnover and costs do NOT
re-scale consistently, so cost-based conclusions are NOT drawn from this script - use
`experiments/run_all.py` for those. Its purpose is purely to compare allocation quality at
equal risk, which is exactly the question the headline numbers confound.

CAVEAT, stated up front: this is still validation-only, and matching on the full validation
period uses the same data twice. It is a diagnostic for the confound, not a new result.
"""
import argparse

import numpy as np
import pandas as pd

from experiments.run_all import build
from src.backtest import run_backtest
from src.data import download_prices, period_bounds, to_returns
from src.metrics import compute_metrics
from src.utils import load_config, set_seed

PAIRS = [("mpc_tailbudget_nofc", "equal_weight_voltarget"),
         ("mpc_tailbudget", "equal_weight_voltarget"),
         ("mpc_tailbudget_fwdvol", "equal_weight_voltarget_fwd"),
         ("mpc_tailbudget", "equal_weight_voltarget_fwd")]


def mean_exposure(res) -> float:
    """Mean gross exposure actually held (sum of |weights|) over the backtest."""
    return float(res.weights.abs().sum(axis=1).mean())


def scale_to_exposure(net: pd.Series, from_exp: float, to_exp: float,
                      rf_daily: float = 0.0) -> pd.Series:
    """Rescale a net-return series so it carries `to_exp` gross exposure instead of `from_exp`.

    Scaling a strategy by k means holding k units of it funded by (k - 1) units of cash at
    rf_daily, so the portfolio return is rf + k*(net - rf).
    """
    if from_exp <= 0:
        return net
    k = to_exp / from_exp
    return rf_daily + k * (net - rf_daily)


def synth_result(net: pd.Series):
    """Wrap a rescaled return series in the minimal BacktestResult shape compute_metrics needs."""
    from src.backtest import BacktestResult
    z = pd.DataFrame(0.0, index=net.index, columns=["w"])
    return BacktestResult(net_returns=net, gross_returns=net,
                          weights=z, turnover=pd.Series(0.0, index=net.index),
                          costs=pd.Series(0.0, index=net.index))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--period", default="val", choices=["val", "train"])
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    d, bt, mc = cfg["data"], cfg["backtest"], cfg["metrics"]

    if cfg.get("synthetic"):
        print("!! SYNTHETIC data - diagnostic only, never reportable")
    prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                             min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    returns = to_returns(prices)
    start, end = period_bounds(cfg["splits"], a.period, returns.index)
    print(f"Real NSE, period={a.period}: {start.date()} -> {end.date()}, "
          f"{returns.shape[1]} tickers")

    strategies = sorted({s for pair in PAIRS for s in pair})
    raw, exps = {}, {}
    for s in strategies:
        st = build(s, cfg, returns.shape[1])
        res = run_backtest(returns, st, start, end, bt["cost_bps"])
        raw[s] = res
        exps[s] = mean_exposure(res)

    print("\n" + "=" * 74)
    print("REALIZED MEAN GROSS EXPOSURE (the confound, measured)")
    print("=" * 74)
    for s in strategies:
        print(f"  {s:<28} {exps[s]:.4f}")

    # Match everything to the LOWEST realized exposure in the set, so nobody is penalized
    # for having taken more risk than the comparison.
    target = min(exps.values())
    print(f"\nMatching all strategies to mean gross exposure = {target:.4f}")

    matched = {}
    for s in strategies:
        net = raw[s].net_returns.loc[start:end]
        matched[s] = scale_to_exposure(net, exps[s], target)

    cols = ["sharpe", "sortino", "cvar_daily", "max_drawdown", "ann_return", "ann_vol"]

    print("\n" + "=" * 74)
    print("AT MATCHED EXPOSURE")
    print("=" * 74)
    rows = {}
    for s in strategies:
        m = compute_metrics(synth_result(matched[s]), cvar_alpha=mc["cvar_alpha"],
                            risk_free=mc["risk_free"])
        rows[s] = m
    df = pd.DataFrame(rows).T
    print(df[[c for c in cols if c in df.columns]].round(4).to_string())

    print("\n" + "=" * 74)
    print("MPC vs BASELINE, matched exposure")
    print("  cvar_daily is stored as a NEGATIVE loss magnitude, so the baseline wins when")
    print("  the MPC's value is MORE negative. 'diff' is mpc - base; negative = base better.")
    print("=" * 74)
    for mpc, base in PAIRS:
        c_m = rows[mpc]["cvar_daily"]
        c_b = rows[base]["cvar_daily"]
        verdict = "MPC better" if c_m > c_b else "BASE better"
        print(f"  {mpc:<24} vs {base:<28} CVaR {c_m:+.4f} vs {c_b:+.4f} "
              f"-> diff {c_m - c_b:+.4f}   {verdict}")
        print(f"  {'':<24}    {'':<28} maxDD {rows[mpc]['max_drawdown']:+.4f} vs "
              f"{rows[base]['max_drawdown']:+.4f}   Sharpe {rows[mpc]['sharpe']:+.3f} vs "
              f"{rows[base]['sharpe']:+.3f}")

    # How much of the UNMATCHED advantage survives? Report the exposure gap explicitly.
    print("\n" + "=" * 74)
    print("HOW MUCH OF THE EDGE WAS DE-RISKING?")
    print("=" * 74)
    for mpc, base in PAIRS:
        raw_c = compute_metrics(raw[mpc], cvar_alpha=mc["cvar_alpha"], risk_free=mc["risk_free"])
        raw_b = compute_metrics(raw[base], cvar_alpha=mc["cvar_alpha"], risk_free=mc["risk_free"])
        u = raw_c["cvar_daily"] - raw_b["cvar_daily"]
        m = rows[mpc]["cvar_daily"] - rows[base]["cvar_daily"]
        print(f"  {mpc:<24} vs {base}")
        print(f"      exposure gap {exps[mpc]:.3f} vs {exps[base]:.3f}  "
              f"(ratio {exps[mpc] / exps[base]:.2f}x)")
        print(f"      CVaR diff unmatched {u:+.4f}  ->  matched {m:+.4f}  "
              f"({100 * (1 - m / u) if u != 0 else float('nan'):.0f}% of the edge was exposure)")

    print("\nNOTE: costs/turnover are not comparable after rescaling; see the module docstring.")


if __name__ == "__main__":
    main()