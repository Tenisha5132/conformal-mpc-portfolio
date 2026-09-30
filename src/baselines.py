"""Baselines for the paper experiments."""
import cvxpy as cp
import numpy as np
import pandas as pd

from src.estimators import chol_psd, shrunk_cov


class EqualWeight:
    name = "equal_weight"

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        n = hist.shape[1]
        return np.full(n, 1.0 / n)


class BuyAndHold:
    name = "buy_hold"

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        if w_prev.sum() < 1e-12:                      # first day: buy equal weights, then never trade
            return np.full(hist.shape[1], 1.0 / hist.shape[1])
        return w_prev


class Markowitz:
    """Rolling mean-variance, long-only, using historical mean and shrunk covariance."""
    name = "markowitz"

    def __init__(self, lookback=252, risk_aversion=5.0, max_weight=0.30, shrink=0.2,
                 max_exposure=1.0):
        self.lookback, self.gamma, self.wmax, self.shrink = lookback, risk_aversion, max_weight, shrink
        self.max_exposure = max_exposure

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        n = hist.shape[1]
        if len(hist) < self.lookback:
            return np.full(n, 1.0 / n)
        R = hist.values[-self.lookback:]
        mu = R.mean(axis=0)
        L = chol_psd(shrunk_cov(R, self.shrink))
        w = cp.Variable(n)
        obj = mu @ w - 0.5 * self.gamma * cp.sum_squares(L.T @ w)
        prob = cp.Problem(cp.Maximize(obj),
                          [w >= 0, w <= self.wmax, cp.sum(w) <= self.max_exposure])
        try:
            prob.solve()
        except Exception:
            return w_prev
        if w.value is None:
            return w_prev
        return np.clip(w.value, 0, None)


class EqualWeightVolTarget:
    """Volatility-targeted equal weight (portfolio-level realized volatility targeting).

    For each decision, scale w_{1/N} by s_t so that portfolio volatility matches
    the target, where volatility is estimated on a rolling window of recent returns.
    The scaling is clipped (vol_max_scale) and exposure is capped by max_exposure.
    """
    name = "equal_weight_voltarget"

    def __init__(self, vol_target=0.10, vol_lookback=60, max_exposure=1.0, vol_max_scale=2.5,
                 trading_days=252):
        self.vol_target = float(vol_target)
        self.vol_lb = int(max(10, vol_lookback))
        self.max_exposure = float(max_exposure)
        self.vol_max_scale = float(vol_max_scale)
        self.trading_days = int(trading_days)

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        n = hist.shape[1]
        w_eq = np.full(n, 1.0 / n)
        if len(hist) < self.vol_lb:
            return w_eq
        R = hist.values[-self.vol_lb:]
        # Portfolio returns of equal-weight over that window
        p = R @ w_eq
        sigma_d = float(np.std(p, ddof=1)) if len(p) > 1 else 0.0
        # `vol_target` is ANNUALIZED, so the estimate must be annualized too. Comparing a
        # daily sigma against an annualized target inflates the scale by sqrt(TD) ~ 16x,
        # which pins the scale at its cap and makes this baseline inert.
        sigma = sigma_d * np.sqrt(self.trading_days)
        if sigma <= 1e-9 or not np.isfinite(sigma):
            s = 1.0
        else:
            s = self.vol_target / sigma
        s = np.clip(s, 1.0 / self.vol_max_scale, self.vol_max_scale)
        w = w_eq * s
        if w.sum() > self.max_exposure:
            w = w * (self.max_exposure / w.sum()) if w.sum() > 0 else w_eq
        return w


class EqualWeightVolTargetFwd:
    """Volatility-targeted equal weight driven by a FORWARD-LOOKING risk signal.

    This is the matched-information control for the paper's forward-volatility experiment.
    `equal_weight_voltarget` estimates volatility on a trailing 60-day window, so when a jump
    happens it de-risks late. This variant substitutes the GARCH one-step conditional
    volatility from `src/fwdvol.py`, which reacts to the most recent shock.

    The point of this class is FAIRNESS. If only the MPC received the forward-looking signal,
    any "win" would only show that a good signal beats no signal. Giving the scalar rule the
    SAME signal isolates what the MPC adds on top of the signal itself.
    """

    name = "equal_weight_voltarget_fwd"

    def __init__(self, vol_target=0.10, max_exposure=1.0, vol_max_scale=2.5,
                 trading_days=252, **fwd_kwargs):
        self.vol_target = float(vol_target)
        self.max_exposure = float(max_exposure)
        self.vol_max_scale = float(vol_max_scale)
        self.trading_days = int(trading_days)
        from src.fwdvol import FwdVolForecaster
        self.fwd = FwdVolForecaster(trading_days=trading_days, **fwd_kwargs)

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        n = hist.shape[1]
        w_eq = np.full(n, 1.0 / n)
        sigma = self.fwd.decide_sigma(hist)          # annualized, same units as the target
        s = 1.0 if (sigma <= 1e-9 or not np.isfinite(sigma)) else self.vol_target / sigma
        s = float(np.clip(s, 1.0 / self.vol_max_scale, self.vol_max_scale))
        w = w_eq * s
        if w.sum() > self.max_exposure:
            w = w * (self.max_exposure / w.sum()) if w.sum() > 0 else w_eq
        return w


class MPCDrawdownRiskAversion:
    """Naive-mean MPC whose risk aversion is raised while the strategy is in drawdown.

    This is the closest competitor to the paper's core contribution: both adapt the risk
    limits, but this one reacts to REALIZED losses while `mpc_fc_tight` reacts to FORECAST
    uncertainty. It is the baseline that isolates *what* the signal buys you.

    Internal equity is maintained from realized outcomes fed back by the backtest engine
    through `observe()`, which is only ever called after a day has been fully realized.
    The drawdown state used on day d therefore reflects days <= d-1: no look-ahead.
    """
    name = "mpc_dd_riskaversion"

    def __init__(self, cfg, n_assets, seed=0):
        from src.estimators import shrunk_cov
        from src.mpc import MPCController

        m, bt = cfg["mpc"], cfg["backtest"]
        bb = cfg.get("baselines", {})
        self.lookback = bt["lookback"]
        self.cost_rate = bt["cost_bps"] / 1e4
        self.ctrl = MPCController(m["horizon"], m["risk_aversion"], bt["cost_bps"],
                                  m["max_weight"], m["forecast_persistence"])
        self.max_exposure = m["max_exposure"]
        self.vol_cap = m["vol_cap"]
        self.cov_shrink = m["cov_shrink"]
        self.n = int(n_assets)
        self.dd_trigger = float(bb.get("dd_trigger", 0.05))
        self.dd_mult = float(bb.get("dd_risk_mult", 3.0))
        self._equity = 1.0
        self._peak = 1.0
        self.dd_armed = False
        self.dd_history = 0

    def drawdown(self) -> float:
        """Current drawdown of this strategy's own net equity from its running peak."""
        if self._peak <= 0:
            return 0.0
        return float((self._peak - self._equity) / self._peak)

    def observe(self, realized_r, net_return, target, w_prev_drift):
        """Called by the backtest AFTER a day is realized. Advances the equity curve."""
        self._equity *= (1.0 + float(net_return))
        self._peak = max(self._peak, self._equity)
        self.dd_armed = self.drawdown() >= self.dd_trigger
        self.dd_history += int(self.dd_armed)

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        if len(hist) < self.lookback:
            return np.full(self.n, 1.0 / self.n)
        R = hist.values[-self.lookback:]
        cov = shrunk_cov(R, self.cov_shrink)
        mu = R.mean(axis=0)
        gamma = self.ctrl.gamma * self.dd_mult if self.dd_armed else None
        return self.ctrl.solve(w_prev, mu, cov, halfwidth=None, kappa=0.0,
                               max_exposure=self.max_exposure, vol_cap=self.vol_cap,
                               risk_aversion=gamma)
