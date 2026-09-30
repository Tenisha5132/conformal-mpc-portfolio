"""Steps 2 and 3: regime-conditional performance and per-episode exposure plots.

  python -m experiments.regime_conditional

Step 2: metrics for every strategy on STRESS days vs CALM days, using the independent
        labelling rule in docs/stress_definition.md (NOT src/regime.py).
Step 3: realized-exposure trajectories for EVERY stress episode in train+val, all on
        shared axes, so the de-risking behaviour of the core layer is visible.

One backtest sweep over train+val serves both steps. The test period is never touched.
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.run_all import ALL, build
from src.backtest import run_backtest
from src.data import download_prices, period_bounds, to_returns
from src.metrics import set_trading_days
from src.stress import (episode_market_drawdown, find_episodes, fit_thresholds,
                        label_stress)
from src.utils import load_config, make_run_dir, save_config, set_seed

# strategies drawn on the episode plots: the ablation core plus the new competitors
PLOT_STRATEGIES = ["equal_weight", "equal_weight_voltarget", "mpc_naive",
                   "mpc_dd_riskaversion", "mpc_fc_tight", "mpc_full"]


def conditional_metrics(r: np.ndarray, trading_days: int, risk_free: float,
                        cvar_alpha: float) -> dict:
    """Metrics for a possibly NON-CONTIGUOUS subset of days.

    Max drawdown is deliberately NOT reported here: on a subset of scattered days the
    "peak to trough" sequence is not a path, so the number would be meaningless.
    """
    r = np.asarray(r, dtype=float)
    if r.size < 3:
        out = {k: float("nan") for k in
               ["sharpe", "mean_daily", "ann_vol", "cvar_daily", "worst_day"]}
        out["n_days"] = int(r.size)
        return out
    excess = r - risk_free
    vol = excess.std(ddof=1)
    q = np.quantile(r, cvar_alpha)
    tail = r[r <= q]
    return {
        "sharpe": float(excess.mean() * trading_days / vol / np.sqrt(trading_days))
        if vol > 0 else float("nan"),
        "mean_daily": float(r.mean()),
        "ann_vol": float(vol * np.sqrt(trading_days)),
        "cvar_daily": float(tail.mean()) if tail.size else float("nan"),
        "worst_day": float(r.min()),
        "n_days": int(r.size),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--strategies", nargs="+", default=ALL)
    ap.add_argument("--pad", type=int, default=20, help="days of context around each episode")
    a = ap.parse_args()

    cfg = load_config(a.config)
    set_seed(cfg["seed"])
    set_trading_days(cfg["backtest"]["trading_days"])
    mc, td = cfg["metrics"], cfg["backtest"]["trading_days"]
    d = cfg["data"]
    prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                             min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    returns = to_returns(prices)
    tr_s, tr_e = period_bounds(cfg["splits"], "train", returns.index)
    va_s, va_e = period_bounds(cfg["splits"], "val", returns.index)

    # Hard stop: the usable window ends at the validation boundary. Anything at or after
    # va_e + 1 day belongs to the untouched test period and must never enter the study.
    usable = returns.loc[:va_e]
    if usable.index.max() > va_e:
        raise SystemExit("usable window leaked past the validation boundary")
    print(f"Study window (train+val only): {usable.index[0].date()} -> {va_e.date()} "
          f"({len(usable)} days)")
    print(f"  train {tr_s.date()} -> {tr_e.date()} | val {va_s.date()} -> {va_e.date()}")
    print(f"  test  {va_e.date()} + 1d onwards: SEALED, never read")

    sl = cfg["stress_label"]
    th = fit_thresholds(usable, tr_e, sl)
    stress = label_stress(usable, th, sl)
    episodes = find_episodes(stress, sl)
    episodes["mkt_dd"] = episode_market_drawdown(usable, episodes)
    print(f"\nIndependent stress labels (thresholds fitted on train only, frozen):")
    print(f"  dispersion p{sl['dispersion_quantile']:.2f} -> {th['disp_threshold']:.5f} | "
          f"market return q{sl['market_return_quantile']:.2f} -> {th['market_return_threshold']:+.5f}")
    print(f"  stress days {int(stress.sum())}/{len(stress)} = {100 * stress.mean():.1f}% | "
          f"{len(episodes)} episodes")

    run_dir = make_run_dir(tag="val-stress")
    th_yaml = {k: (v.isoformat() if isinstance(v, (pd.Timestamp, np.datetime64)) else v)
               for k, v in th.items()}
    save_config({**cfg, "cli": vars(a), "stress_thresholds": th_yaml,
                 "window": [str(usable.index[0]), str(va_e)]}, run_dir)

    # --- single backtest sweep over train+val -------------------------------
    exposure, nets = {}, {}
    for name in a.strategies:
        res = run_backtest(returns, build(name, cfg, returns.shape[1]), tr_s, va_e,
                           cfg["backtest"]["cost_bps"])
        exposure[name] = res.weights.sum(axis=1)
        nets[name] = res.net_returns
        print(f"  ran {name:24s} {len(res.net_returns)} days "
              f"avg exposure {exposure[name].mean():.2f}")
    pd.DataFrame(exposure).to_csv(run_dir / "exposure.csv")
    pd.DataFrame(nets).to_csv(run_dir / "daily_net_returns.csv")

    # --- STEP 2: stress vs calm --------------------------------------------
    val_mask = (usable.index >= va_s) & (usable.index <= va_e)
    stress_np = stress.reindex(usable.index).fillna(False).to_numpy(dtype=bool)
    rows = []
    for period_name, mask in [("train", (usable.index >= tr_s) & (usable.index <= tr_e)),
                              ("val", val_mask)]:
        m_np = np.asarray(mask, dtype=bool)
        st = stress_np[m_np]
        for name in a.strategies:
            r = nets[name].reindex(usable.index).to_numpy()[m_np]
            for cond, sel in [("stress", st), ("calm", ~st)]:
                m = conditional_metrics(r[sel], td, mc["risk_free"], mc["cvar_alpha"])
                rows.append({"period": period_name, "strategy": name, "condition": cond, **m})
    cond_tbl = pd.DataFrame(rows)
    cond_tbl.to_csv(run_dir / "regime_conditional.csv", index=False)

    print("\n=== STEP 2: VALIDATION, stress vs calm (independent labels) ===")
    v = cond_tbl[cond_tbl["period"] == "val"]
    piv = v.pivot_table(index="strategy", columns="condition",
                        values=["sharpe", "mean_daily", "cvar_daily", "worst_day", "n_days"])
    print(f"{'strategy':24s} {'sharpeS':>8s} {'sharpeC':>8s} {'cvarS':>8s} {'cvarC':>8s} "
          f"{'worstS':>8s} {'worstC':>8s} {'nS':>5s} {'nC':>5s}")
    for name in a.strategies:
        s = v[(v.strategy == name) & (v.condition == "stress")].iloc[0]
        c = v[(v.strategy == name) & (v.condition == "calm")].iloc[0]
        print(f"{name:24s} {s['sharpe']:8.2f} {c['sharpe']:8.2f} {s['cvar_daily']:8.4f} "
              f"{c['cvar_daily']:8.4f} {s['worst_day']:8.4f} {c['worst_day']:8.4f} "
              f"{int(s['n_days']):5d} {int(c['n_days']):5d}")
    print("\n  (sharpeS = Sharpe on stress days only; maxDD intentionally omitted on a")
    print("   non-contiguous day subset - see conditional_metrics() docstring)")

    # --- STEP 3: per-episode exposure, all episodes, shared axes ------------
    plot_names = [n for n in PLOT_STRATEGIES if n in exposure]
    n_ep = len(episodes)
    ncol = int(np.ceil(np.sqrt(n_ep)))
    nrow = int(np.ceil(n_ep / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.5, nrow * 1.9 + 1.6),
                             sharex=True, sharey=True, squeeze=False)
    for k, (start, row) in enumerate(episodes.iterrows()):
        ax = axes[k // ncol][k % ncol]
        lo = max(usable.index[0], start - pd.Timedelta(days=int(a.pad * 1.6)))
        hi = min(va_e, row["end"] + pd.Timedelta(days=int(a.pad * 1.6)))
        idx = usable.index[(usable.index >= lo) & (usable.index <= hi)]
        rel = (idx - start).days
        ax.axvspan(0.0, float((row["end"] - start).days), color="#fed7d7", alpha=0.55, lw=0)
        for name in plot_names:
            ax.plot(rel, exposure[name].reindex(idx).to_numpy(), lw=1.1, label=name)
        ax.axhline(0.5, color="#718096", lw=0.6, ls=":")
        if k == 0:
            ax.legend(fontsize=5.0, loc="lower left", ncol=2, framealpha=0.85)
        if k % ncol == 0:
            ax.set_ylabel("exposure", fontsize=6)
        if k // ncol == nrow - 1:
            ax.set_xlabel("days from episode start", fontsize=6)
        ax.tick_params(labelsize=5.5)
        ax.set_title(f"{start.date()} ({row['n_span_days']}d)", fontsize=5.8)
    for k in range(n_ep, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"Realized exposure through all {n_ep} independent stress episodes "
                 f"(train+val, red = episode span)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    fig.savefig(run_dir / "episode_exposure_all.png", dpi=130)
    plt.close(fig)

    # average trajectory aligned at episode start - the summary panel
    agg = {}
    curves = {n: [] for n in plot_names}
    for start, row in episodes.iterrows():
        lo = max(usable.index[0], start - pd.Timedelta(days=int(a.pad * 1.6)))
        hi = min(va_e, row["end"] + pd.Timedelta(days=int(a.pad * 1.6)))
        idx = usable.index[(usable.index >= lo) & (usable.index <= hi)]
        rel = (idx - start).days
        for n in plot_names:
            curves[n].append(pd.Series(exposure[n].reindex(idx).to_numpy(), index=rel))
    grid = np.arange(-int(a.pad * 1.6), int(a.pad * 1.6) + 60)
    for n in plot_names:
        mat = np.array([c.reindex(grid).to_numpy() for c in curves[n]], dtype=float)
        agg[n] = pd.Series(np.nanmean(mat, axis=0), index=grid)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for n in plot_names:
        ax.plot(agg[n].index, agg[n].to_numpy(), lw=1.6, label=n)
    ax.axvspan(0, float(episodes["n_span_days"].median()), color="#fed7d7", alpha=0.5,
               label="median episode span")
    ax.axhline(0.5, color="#718096", lw=0.8, ls=":")
    ax.set_xlabel("days from episode start")
    ax.set_ylabel("mean realized exposure")
    ax.set_title(f"Average exposure trajectory aligned at episode start "
                 f"(n={n_ep} real episodes)", fontsize=10)
    ax.legend(fontsize=7.5); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(run_dir / "episode_exposure_mean.png", dpi=150)
    plt.close(fig)

    print("\n=== STEP 3: mean exposure at episode start / +10d / +20d ===")
    print(f"{'strategy':24s} {'start':>8s} {'+10d':>8s} {'+20d':>8s}")
    for n in plot_names:
        s = agg[n]
        g = lambda k: float(s.reindex([k]).to_numpy()[0]) if k in s.index else float("nan")
        print(f"{n:24s} {g(0):8.3f} {g(10):8.3f} {g(20):8.3f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()