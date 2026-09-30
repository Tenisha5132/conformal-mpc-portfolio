"""Render the closed-loop block diagram of the controller (paper figure).

The controller is a discrete-time feedback loop: the market is the plant, the MPC is
the controller, and the conformal/tightening layer is the measurement-and-scheduling
path that sets the controller's risk limits. This script draws it from the live config
so the figure cannot drift from the code it documents.

    python -m experiments.make_block_diagram            # -> docs/figures/block_diagram.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.utils import load_config

# palette: neutral greys for the standard loop, one accent for the core contribution
INK = "#1b1f24"
MUTED = "#6b7280"
PLANT = "#e8eaed"
STANDARD = "#f4f5f7"
CORE = "#fde2c8"
CORE_EDGE = "#c2410c"
ACCENT = "#b45309"


def _box(ax, cy, title, subtitle, face, edge=MUTED, width=0.58, height=0.082,
         x=0.52, title_dy=0.017, sub_dy=-0.016, badge=None, badge_dy=0.0):
    """Rounded block centred at (x, cy) with a bold title and a detail line."""
    p = FancyBboxPatch(
        (x - width / 2, cy - height / 2), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.012",
        linewidth=1.3, facecolor=face, edgecolor=edge, zorder=2)
    ax.add_patch(p)
    ax.text(x, cy + title_dy, title, ha="center", va="center",
            fontsize=10.5, fontweight="bold", color=INK, zorder=3)
    if badge:
        ax.text(x, cy + badge_dy, badge, ha="center", va="center",
                fontsize=7.6, fontweight="bold", color=ACCENT, zorder=3)
    if subtitle:
        ax.text(x, cy + sub_dy, subtitle, ha="center", va="center",
                fontsize=7.6, color=MUTED, zorder=3)
    return p


def _arrow(ax, p_from, p_to, label=None, color=INK, rad=0.0, lw=1.25, ls="-"):
    """Arrow between two points, optionally with a label beside the shaft."""
    a = FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=13,
                        linewidth=lw, color=color, zorder=4,
                        connectionstyle=f"arc3,rad={rad}",
                        linestyle=ls, shrinkA=0, shrinkB=0)
    ax.add_patch(a)
    if label:
        mx, my = (p_from[0] + p_to[0]) / 2, (p_from[1] + p_to[1]) / 2
        ax.text(mx + 0.045, my, label, ha="left", va="center", fontsize=7.6,
                color=color, zorder=5,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.94))
    return a


def build(cfg, out_path: Path):
    m, b = cfg["mpc"], cfg["backtest"]
    fig, ax = plt.subplots(figsize=(8.4, 12.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    H, bw, bh, X = m["horizon"], 0.58, 0.082, 0.52
    y_market, y_meas = 0.915, 0.800
    y_fc, y_conf, y_sched, y_mpc, y_act = 0.685, 0.570, 0.450, 0.290, 0.130
    h_mpc = bh + 0.034

    # ---- plant ------------------------------------------------------------
    ax.text(X, 0.985, "MARKET  (plant)", ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=INK)
    ax.text(X, 0.963, f"10 NSE large caps   |   daily   |   {b['trading_days']} trading days/yr",
            ha="center", va="center", fontsize=7.8, color=MUTED)
    _box(ax, y_market, "price process  x(t)", "adjusted close, real market data",
         PLANT, INK, width=bw, height=0.070, x=X, title_dy=0.015, sub_dy=-0.015)

    # ---- measurement ------------------------------------------------------
    _box(ax, y_meas, "MEASUREMENT", "hist = returns[:d]   (strictly t < d, no look-ahead)",
         STANDARD, x=X, width=bw, height=bh)

    # ---- forecaster -------------------------------------------------------
    _box(ax, y_fc, "FORECASTER", "ridge on lagged returns, retrained walk-forward",
         STANDARD, x=X, width=bw, height=bh)

    # ---- conformal --------------------------------------------------------
    _box(ax, y_conf, "ADAPTIVE CONFORMAL PREDICTION",
         f"per-asset ACI (Gibbs & Candes), target {1 - cfg['conformal']['alpha']:.0%} intervals",
         STANDARD, x=X, width=bw, height=bh)

    # ---- core contribution ------------------------------------------------
    h_sched = bh + 0.052
    _box(ax, y_sched, "RISK-LIMIT SCHEDULER",
         r"tighten_limits():   scale $= 1/(1+\beta\,\max(\mathrm{ratio}-1,\,0))$"
         "\n" + f"regime:   stress \u2192 risk aversion \u00d7{cfg['regime']['stress_risk_mult']:.0f},"
         f"   exposure \u2264 {cfg['regime']['stress_exposure']:.0%}",
         CORE, CORE_EDGE, x=X, width=bw, height=h_sched,
         title_dy=0.040, sub_dy=-0.011,
         badge="\u2605  CORE CONTRIBUTION  \u2014  uncertainty-driven constraint tightening",
         badge_dy=0.021)

    # ---- MPC --------------------------------------------------------------
    _box(ax, y_mpc, "MPC OPTIMIZER  (CVXPY, receding horizon)",
         r"max $\sum_k$ [$(\mu_k-\kappa hw_k)^{\!\top} w_k$  $-\ \frac{\gamma}{2}w_k^{\!\top}\Sigma w_k$"
         r"  $-\ c\|w_k-w_{k-1}\|_1$ ]"
         "\n" + r"$0\leq w_k\leq$" + f"{m['max_weight']:.0%}"
         + r",   $\sum_k w_k\leq e_{\max}(t)$,   optionally  $\sqrt{w_k^{\!\top}\Sigma w_k}\leq$ cap"
         + "\n" + f"H = {H} steps, apply first step only",
         STANDARD, x=X, width=bw, height=h_mpc, title_dy=0.030, sub_dy=-0.014)

    # ---- actuator ---------------------------------------------------------
    _box(ax, y_act, "ACTUATOR",
         f"rebalance to w*, pay {b['cost_bps']:.0f} bps one-way on turnover",
         STANDARD, x=X, width=bw, height=bh)

    # ---- main forward path ------------------------------------------------
    _arrow(ax, (X, y_market - 0.035), (X, y_meas + bh / 2))
    _arrow(ax, (X, y_meas - bh / 2), (X, y_fc + bh / 2))
    _arrow(ax, (X, y_fc - bh / 2), (X, y_conf + bh / 2), label=r"$\mu_d$")
    _arrow(ax, (X, y_conf - bh / 2), (X, y_sched + h_sched / 2), label=r"$hw_t$")
    _arrow(ax, (X, y_sched - h_sched / 2), (X, y_mpc + h_mpc / 2),
           label=r"$e_{\max},\ \gamma$", color=CORE_EDGE)
    _arrow(ax, (X, y_mpc - h_mpc / 2), (X, y_act + bh / 2), label=r"$w^*_1$")

    # ---- feedback path (actuator -> market) -------------------------------
    fb_x = 0.90
    ax.plot([X + bw / 2, fb_x], [y_act, y_act], color=INK, lw=1.25, zorder=1)
    ax.plot([fb_x, fb_x], [y_act, y_market], color=INK, lw=1.25, zorder=1)
    _arrow(ax, (fb_x, y_market), (X + bw / 2, y_market))
    ax.text(fb_x + 0.014, (y_act + y_market) / 2,
            "closed loop\nweights drift\nwith prices", ha="left", va="center",
            fontsize=7.6, color=MUTED, rotation=90)

    # ---- side inputs into the scheduler -----------------------------------
    bus = 0.105
    ax.plot([bus, bus], [0.505, y_sched], color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    _arrow(ax, (bus, y_sched), (X - bw / 2, y_sched), color=MUTED, ls=(0, (4, 3)), lw=1.0)
    ax.text(bus - 0.015, 0.512, "interval-width\nhistory", ha="right", va="bottom",
            fontsize=7.4, color=MUTED)
    ax.text(bus - 0.015, 0.492, f"{cfg['regime']['vol_window']}d market vol,\nexpanding "
            f"p{cfg['regime']['quantile']:.0%} threshold", ha="right", va="top",
            fontsize=7.4, color=MUTED)


    ax.text(0.5, 0.032,
            "Each layer is ablated independently:  "
            "mpc_naive $\\rightarrow$ mpc_fc $\\rightarrow$ mpc_fc_robust "
            "$\\rightarrow$ mpc_fc_tight $\\rightarrow$ mpc_full",
            ha="center", va="center", fontsize=8.0, color=MUTED, style="italic")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", default="docs/figures/block_diagram.png")
    a = ap.parse_args()
    cfg = load_config(a.config)
    p = build(cfg, Path(a.out))
    print(f"Wrote {p}")


if __name__ == "__main__":
    main()
