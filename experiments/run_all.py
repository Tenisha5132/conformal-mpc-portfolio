"""Run strategies over one period and save metrics + plots.

  python -m experiments.run_all --synthetic --period val --strategies equal_weight mpc_naive
  python -m experiments.run_all --period val
  python -m experiments.run_all --period test --final      # only when everything is frozen
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.baselines import (BuyAndHold, EqualWeight, EqualWeightVolTarget, Markowitz,
                           MPCDrawdownRiskAversion)
from src.data import download_prices, make_synthetic_prices, period_bounds, to_returns
from src.metrics import compute_metrics, set_trading_days
from src.strategies import MPCSelfCalStrategy, MPCTailBudgetStrategy, MPCStrategy
from src.utils import load_config, make_run_dir, save_config, set_seed

ALL = ["equal_weight", "buy_hold", "markowitz", "equal_weight_voltarget", "mpc_naive",
       "mpc_dd_riskaversion", "mpc_fc", "mpc_fc_robust", "mpc_fc_tight", "mpc_selfcal",
       "mpc_tailbudget", "mpc_full"]


def build(name, cfg, n):
    bt, m = cfg["backtest"], cfg["mpc"]
    if name == "equal_weight":
        return EqualWeight()
    if name == "buy_hold":
        return BuyAndHold()
    if name == "markowitz":
        return Markowitz(bt["lookback"], m["risk_aversion"], m["max_weight"], m["cov_shrink"],
                         m["max_exposure"])
    if name == "equal_weight_voltarget":
        bl = cfg["baselines"]
        return EqualWeightVolTarget(bl["vol_target"], bl["vol_lookback"], m["max_exposure"],
                                    bl["vol_max_scale"], bt["trading_days"])
    if name == "mpc_dd_riskaversion":
        return MPCDrawdownRiskAversion(cfg, n, seed=cfg["seed"])
    flags = {
        "mpc_naive":     dict(),
        "mpc_fc":        dict(use_forecaster=True),
        "mpc_fc_robust": dict(use_forecaster=True, use_conformal=True),
        "mpc_fc_tight":  dict(use_forecaster=True, use_conformal=True, use_tighten=True),
        "mpc_full":      dict(use_forecaster=True, use_conformal=True, use_tighten=True, use_regime=True),
    }.get(name)
    if name == "mpc_selfcal":
        return MPCSelfCalStrategy(cfg, n, name=name, seed=cfg["seed"])
    if name == "mpc_tailbudget":
        return MPCTailBudgetStrategy(cfg, n, name=name, seed=cfg["seed"])
    return MPCStrategy(cfg, n, name=name, seed=cfg["seed"], **flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--period", choices=["val", "test"], default="val")
    ap.add_argument("--strategies", nargs="+", default=ALL)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--final", action="store_true", help="required to touch the test period")
    a = ap.parse_args()

    if a.period == "test" and not a.final:
        raise SystemExit("Refusing to run on the TEST period without --final. Tune on val only.")

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    set_trading_days(cfg["backtest"]["trading_days"])
    d = cfg["data"]
    if a.synthetic:
        prices = make_synthetic_prices(cfg, seed=cfg["seed"])
        print("!! SYNTHETIC data - offline dry run only, NOT a publishable result")
    else:
        prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                                 min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
        print(f"Real market data: {len(prices.columns)} tickers, "
              f"{prices.index[0].date()} -> {prices.index[-1].date()}")
    returns = to_returns(prices)
    start, end = period_bounds(cfg["splits"], a.period, returns.index)

    run_dir = make_run_dir(tag=f"{a.period}{'-synthetic' if a.synthetic else ''}")
    save_config({**cfg, "cli": vars(a)}, run_dir)

    rows, curves = {}, {}
    mc = cfg["metrics"]
    diagnostics = {}
    for name in a.strategies:
        strat = build(name, cfg, returns.shape[1])          # fresh object per run
        res = run_backtest(returns, strat, start, end, cfg["backtest"]["cost_bps"])
        rows[name] = compute_metrics(res, cvar_alpha=mc["cvar_alpha"], risk_free=mc["risk_free"])
        curves[name] = res.equity()
        res.weights.to_csv(run_dir / f"weights_{name}.csv")
        # self-calibration diagnostics: realized breach frequency and the beta_t path
        if hasattr(strat, "beta_state"):
            import pandas as pd
            pd.DataFrame(strat.log).assign(date=res.net_returns.index).to_csv(
                run_dir / f"selfcal_{name}.csv", index=False)
            diagnostics[name] = {
                "realized_breach_frequency": strat.breach_frequency(),
                "target_delta": cfg["selfcal"]["delta"],
                "loss_threshold": cfg["selfcal"]["loss_threshold"],
                "beta_final": float(strat.beta_state.beta),
                "beta_mean": float(np.mean(strat.beta_trajectory())) if len(strat.log) else float("nan"),
                "beta_max": float(np.max(strat.beta_trajectory())) if len(strat.log) else float("nan"),
                "ensemble_members": "+".join(strat.ensemble.names()),
            }
            d = diagnostics[name]
            print(f"  [selfcal] breach_freq={d['realized_breach_frequency']:.4f} "
                  f"(target delta={d['target_delta']})  beta final={d['beta_final']:.4f} "
                  f"mean={d['beta_mean']:.4f} max={d['beta_max']:.4f}")
        # tail-risk budget diagnostics (mpc_tailbudget): is the CVaR constraint live?
        if hasattr(strat, "last_cvar_budget") and len(strat.log):
            import pandas as pd
            bud = np.array([r.get("cvar_budget") for r in strat.log], dtype=float)
            used = np.array([r.get("cvar_used") for r in strat.log], dtype=float)
            ok = np.isfinite(bud) & np.isfinite(used) & (bud > 0) & (used > 0)
            live = used[ok] / bud[ok]
            if ok.any():
                diagnostics.setdefault(name, {})
                diagnostics[name].update({
                    "cvar_budget_mean": float(bud[ok].mean()),
                    "budget_binding_freq": float(np.mean(live > 0.98)),
                    "budget_util_mean": float(live.mean()),
                })
                d = diagnostics[name]
                print(f"  [budget]  mean_budget={d['cvar_budget_mean']:.5f}  "
                      f"binding_freq={d['budget_binding_freq']:.3f}  util={d['budget_util_mean']:.3f}")
        print(f"{name:15s} sharpe={rows[name]['sharpe']:.2f}  maxDD={rows[name]['max_drawdown']:.1%}"
              f"  ret={rows[name]['ann_return']:.1%}  turnover={rows[name]['avg_daily_turnover']:.3f}")
    if diagnostics:
        import pandas as pd
        pd.DataFrame(diagnostics).T.to_csv(run_dir / "selfcal_diagnostics.csv")

    table = pd.DataFrame(rows).T
    table.to_csv(run_dir / "metrics.csv")
    fig, ax = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for name, eq in curves.items():
        ax[0].plot(eq.index, eq.values, label=name)
        ax[1].plot(eq.index, eq / eq.cummax() - 1, label=name)
    ax[0].set_ylabel("wealth"); ax[1].set_ylabel("drawdown"); ax[0].legend()
    fig.tight_layout(); fig.savefig(run_dir / "equity_drawdown.png", dpi=140)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
