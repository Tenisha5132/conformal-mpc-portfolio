"""Walk-forward backtest engine.

Timeline (this is the anti-look-ahead contract):
  for each trading day d in the period:
      hist   = returns[:d]            <- only days STRICTLY BEFORE d
      target = strategy.decide(hist, current_weights)
      trade to target, pay cost = cost_rate * sum|target - current|
      earn returns[d] with the target weights held during day d
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    net_returns: pd.Series
    gross_returns: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    costs: pd.Series

    def equity(self) -> pd.Series:
        return (1 + self.net_returns).cumprod()


def _clean_target(target: np.ndarray, n: int) -> np.ndarray:
    target = np.asarray(target, dtype=float).ravel()
    if target.shape[0] != n or not np.all(np.isfinite(target)):
        raise ValueError("strategy returned invalid weights")
    target = np.clip(target, 0.0, None)              # long-only
    s = target.sum()
    if s > 1.0:                                       # never lever
        target = target / s
    return target


def run_backtest(returns: pd.DataFrame, strategy, start, end, cost_bps: float = 10.0) -> BacktestResult:
    n_assets = returns.shape[1]
    cost_rate = cost_bps / 1e4
    start_i = max(int(returns.index.searchsorted(pd.Timestamp(start), side="left")), 1)
    end_i = int(returns.index.searchsorted(pd.Timestamp(end), side="right"))  # exclusive

    w_drift = np.zeros(n_assets)                      # start fully in cash
    dates, net, gross, turns, costs, W = [], [], [], [], [], []
    # Optional outcome callback: strategies that track internal state (e.g. the drawdown
    # baseline) can implement observe(realized_asset_returns, net_return, target, w_drift_prev)
    # so they update AFTER the day is fully realized. This adds no look-ahead because it is
    # only ever called with day d's return, after the day-d decision has already been made.
    observer = getattr(strategy, "observe", None)

    for d in range(start_i, end_i):
        hist = returns.iloc[:d]                       # NO data from day d or later
        target = _clean_target(strategy.decide(hist, w_drift.copy()), n_assets)

        turn = float(np.abs(target - w_drift).sum())
        cost = cost_rate * turn
        r = returns.iloc[d].values
        g = float(target @ r)
        nr = (1.0 - cost) * (1.0 + g) - 1.0

        if observer is not None:
            observer(r, nr, target.copy(), w_drift.copy())

        w_drift = target * (1.0 + r) / (1.0 + g)      # weights drift with prices

        dates.append(returns.index[d])
        net.append(nr); gross.append(g); turns.append(turn); costs.append(cost); W.append(target)

    idx = pd.DatetimeIndex(dates)
    return BacktestResult(
        net_returns=pd.Series(net, index=idx, name="net"),
        gross_returns=pd.Series(gross, index=idx, name="gross"),
        weights=pd.DataFrame(W, index=idx, columns=returns.columns),
        turnover=pd.Series(turns, index=idx, name="turnover"),
        costs=pd.Series(costs, index=idx, name="cost"),
    )
