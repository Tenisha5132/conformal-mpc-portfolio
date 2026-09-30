"""Performance metrics (risk-free rate assumed 0; state this in the paper)."""
import numpy as np

from src.backtest import BacktestResult

# Trading days per year. Set from config (backtest.trading_days) by set_trading_days()
# so annualization matches the market's actual calendar rather than a hard-coded value.
TD = 252


def set_trading_days(n):
    """Configure the annualization factor from config instead of hard-coding it."""
    global TD
    TD = int(n)
    return TD


def compute_metrics(res: BacktestResult, cvar_alpha=0.05, risk_free=0.0) -> dict:
    r = res.net_returns.values
    n = len(r)
    eq = np.concatenate([[1.0], np.cumprod(1 + r)])
    peak = np.maximum.accumulate(eq)
    max_dd = float(((eq - peak) / peak).min())
    ann_ret = float(eq[-1] ** (TD / n) - 1)
    ann_vol = float(r.std(ddof=1) * np.sqrt(TD))
    var = np.quantile(r, cvar_alpha)
    tail = r[r <= var]
    excess = r - risk_free
    exc_vol = float(excess.std(ddof=1) * np.sqrt(TD))
    exc_down = float(np.sqrt(np.mean(np.minimum(excess, 0.0) ** 2)) * np.sqrt(TD))
    return {
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": float(excess.mean() * TD / exc_vol) if exc_vol > 0 else np.nan,
        "sortino": float(excess.mean() * TD / exc_down) if exc_down > 0 else np.nan,
        "max_drawdown": max_dd,
        "cvar_daily": float(tail.mean()) if len(tail) else np.nan,
        "avg_daily_turnover": float(res.turnover.mean()),
        "total_cost": float(res.costs.sum()),
        "final_wealth": float(eq[-1]),
        "n_days": n,
    }
