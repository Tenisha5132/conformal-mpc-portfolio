"""MPC strategies. One class, feature flags = the ablation ladder.

  mpc_naive            historical-mean forecast, plain MPC
  mpc_fc               + learned forecaster
  mpc_fc_robust        + conformal (adaptive) intervals, worst-case mu = mu - kappa*halfwidth
  mpc_fc_tight         + uncertainty-tightened risk limits            <- core novelty
  mpc_selfcal          + learned beta_t, ensemble-disagreement ratio
  mpc_tailbudget       + HARD CVaR(5%) tail-risk budget (R-U)          <- paper title
  mpc_full             + regime-aware (calm/stress) parameters        <- add-on #1
"""
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.conformal import AdaptiveConformal
from src.estimators import shrunk_cov
from src.forecaster import build_forecaster
from src.mpc import MPCController, cvar_budget_scale, tighten_limits
from src.regime import VolRegimeDetector
from src.selfcal import BetaState, DisagreementEnsemble, uncertainty_ratio


class MPCStrategy:
    def __init__(self, cfg, n_assets, use_forecaster=False, use_conformal=False,
                 use_tighten=False, use_regime=False, name="mpc", seed=0):
        self.name = name
        m, bt = cfg["mpc"], cfg["backtest"]
        self.cfg = cfg
        self.lookback = bt["lookback"]
        self.ctrl = MPCController(m["horizon"], m["risk_aversion"], bt["cost_bps"], m["max_weight"],
                                  m["forecast_persistence"])
        self.max_exposure, self.vol_cap, self.cov_shrink = m["max_exposure"], m["vol_cap"], m["cov_shrink"]
        self.mu_shrink = cfg["forecaster"]["mu_shrink"]
        self.n = n_assets
        self.forecaster = build_forecaster(cfg, n_assets, seed) if use_forecaster else None
        c = cfg["conformal"]
        self.conformal = (AdaptiveConformal(n_assets, c["alpha"], c["gamma"], c["window"], c["min_cal"])
                          if (use_conformal and use_forecaster) else None)
        self.kappa = c["kappa"] if self.conformal else 0.0
        self.use_tighten = use_tighten and self.conformal is not None
        self.t = cfg["tighten"]
        self._hw_hist = deque(maxlen=self.t["ratio_history"])
        self._last_pred = None
        r = cfg["regime"]
        self.regime = (VolRegimeDetector(r["vol_window"], r["quantile"], r["min_history"],
                                         r["min_vol_history"]) if use_regime else None)
        self.r = r
        self.last_regime = 0

    def decide(self, hist: pd.DataFrame, w_prev: np.ndarray) -> np.ndarray:
        if len(hist) < self.lookback:
            return np.full(self.n, 1.0 / self.n)
        R = hist.values[-self.lookback:]
        cov = shrunk_cov(R, self.cov_shrink)
        mu = R.mean(axis=0)

        hw = None
        if self.forecaster is not None:
            pred = self.forecaster.predict(hist)
            if self.conformal is not None and self._last_pred is not None:
                # the return realized on the last day of hist vs what we predicted for it
                self.conformal.update(hist.values[-1], self._last_pred)
            self._last_pred = pred
            mu = self.mu_shrink * pred + (1 - self.mu_shrink) * mu
            if self.conformal is not None:
                # Pre-calibration fallback half-width: the normal quantile matching the
                # target coverage, derived from alpha rather than hard-coded.
                z = norm.ppf(1.0 - self.conformal.alpha / 2.0)
                hw = self.conformal.halfwidth(fallback=self.forecaster.resid_std * z)

        exposure, vol_cap, gamma = self.max_exposure, self.vol_cap, None
        if hw is not None:
            hw_mean = float(np.mean(hw))
            ratio = (hw_mean / np.median(self._hw_hist)
                     if len(self._hw_hist) >= self.t["min_ratio_history"] else 1.0)
            self._hw_hist.append(hw_mean)
            if self.use_tighten:
                exposure, vol_cap = tighten_limits(ratio, exposure, vol_cap,
                                                   self.t["beta"], self.t["min_exposure"])

        if self.regime is not None:
            self.last_regime = self.regime.update(hist)
            if self.last_regime == 1:
                gamma = self.ctrl.gamma * self.r["stress_risk_mult"]
                exposure = min(exposure, self.r["stress_exposure"])

        return self.ctrl.solve(w_prev, mu, cov, halfwidth=hw, kappa=self.kappa,
                               max_exposure=exposure, vol_cap=vol_cap, risk_aversion=gamma)


class MPCSelfCalStrategy(MPCStrategy):
    """Self-calibrating tightening + ensemble-disagreement uncertainty (mpc_selfcal).

    Two changes relative to `mpc_fc_tight`:

    * the tightening coefficient `beta` passed to `tighten_limits` is `beta_t`, learned
      online from realized breaches (src/selfcal.BetaState) instead of the fixed
      `tighten.beta` constant;
    * the tightening signal is the geometric mean of the conformal-width ratio and the
      ensemble-disagreement ratio, each normalized by its own trailing median.

    Everything else (MPC, forecaster, conformal calibration, costs) is inherited unchanged,
    so the ladder comparison stays controlled.

    NO LOOK-AHEAD: beta_t is advanced only by `observe()`, which the backtest calls after
    a day is fully realized, and the value used on day t is the one settled through day
    t-1. The disagreement history and width history are appended for the CURRENT day's
    signal only after the ratio has been computed for it.
    """

    def __init__(self, cfg, n_assets, name="mpc_selfcal", seed=0):
        super().__init__(cfg, n_assets, use_forecaster=True, use_conformal=True,
                         use_tighten=True, use_regime=False, name=name, seed=seed)
        sc = cfg["selfcal"]
        self.beta_state = BetaState(sc["beta0"], sc["eta"], sc["delta"], sc["loss_threshold"])
        self.ensemble = DisagreementEnsemble(cfg, n_assets, seed)
        self.dis_history = deque(maxlen=int(cfg["ensemble"]["history"]))
        # per-run logs (populated day by day, read after the backtest finishes)
        self.log = []
        self.last_cvar_budget = None
        self.last_cvar_used = None
        # forecaster ON by default; mpc_tailbudget_nofc sets this False (see _blend_mu)
        self.use_forecast = True

    def _blend_mu(self, hist_mu, pred):
        """Blend the forecast into the drift estimate.

        Normally the standard convex blend. When `use_forecast=False` (mpc_tailbudget_nofc)
        this returns the historical mean, which isolates the FORECAST while keeping the
        tail-risk budget and the whole self-calibration loop intact.
        """
        if not self.use_forecast:
            return hist_mu
        return self.mu_shrink * pred + (1 - self.mu_shrink) * hist_mu

    def _compute_budget(self, R, ratio, beta_t):
        """Tail-risk budget for the current day, or None to disable the CVaR constraint.

        The base `mpc_selfcal` returns None, so its behaviour is byte-for-byte unchanged.
        `mpc_tailbudget` overrides this to return a vol-scaled budget shrunk by beta_t.
        """
        return None

    def decide(self, hist, w_prev):
        if len(hist) < self.lookback:
            return np.full(self.n, 1.0 / self.n)
        R = hist.values[-self.lookback:]
        cov = shrunk_cov(R, self.cov_shrink)
        hist_mu = R.mean(axis=0)

        # --- conformal calibration on the day whose return is now observable ---
        ens_pred, dis = self.ensemble.predict(hist)
        pred = self.forecaster.predict(hist)
        if self._last_pred is not None:
            self.conformal.update(hist.values[-1], self._last_pred)
        self._last_pred = pred

        mu = self._blend_mu(hist_mu, pred)
        z = norm.ppf(1.0 - self.conformal.alpha / 2.0)
        hw = self.conformal.halfwidth(fallback=self.forecaster.resid_std * z)

        hw_mean = float(np.mean(hw))
        self._hw_hist.append(hw_mean)
        self.dis_history.append(float(dis))
        self.ensemble.observe_disagreement(dis)

        # --- geometric-mean uncertainty ratio, with LEARNED beta ---
        ratio = uncertainty_ratio(hw_mean, dis, self._hw_hist, self.dis_history,
                                  min_history=self.t["min_ratio_history"])
        beta_t = self.beta_state.beta
        exposure, vol_cap = tighten_limits(ratio, self.max_exposure, self.vol_cap,
                                           beta_t, self.t["min_exposure"])

        budget = self._compute_budget(R, ratio, beta_t)
        self.log.append({
            "beta": float(beta_t),
            "uncertainty_ratio": float(ratio),
            "hw_mean": hw_mean,
            "disagreement": float(dis),
            "exposure_cap": float(exposure),
            "breach_freq": self.beta_state.breach_frequency(),
            "cvar_budget": budget,
            "cvar_used": None,          # filled in by the subclass after the solve
            "use_forecast": int(self.use_forecast),
        })
        w = self.ctrl.solve(w_prev, mu, cov, halfwidth=hw, kappa=self.kappa,
                            max_exposure=exposure, vol_cap=vol_cap, cvar_budget=budget)
        self.log[-1]["cvar_used"] = (None if budget is None else self.ctrl.last_cvar)
        return w

    def observe(self, realized_r, net_return, target, w_prev_drift):
        """Settle the just-realized day: advance beta_t for use from tomorrow on."""
        self.beta_state.observe(net_return)

    def beta_trajectory(self) -> np.ndarray:
        return np.asarray(self.beta_state.history, dtype=float)

    def breach_frequency(self, window: int | None = None) -> float:
        return self.beta_state.breach_frequency(window)


class MPCTailBudgetStrategy(MPCSelfCalStrategy):
    """SELF-CALIBRATING UNCERTAINTY-AWARE MPC FOR TAIL-RISK BUDGETING (mpc_tailbudget).

    This is the flagship strategy behind the paper's title. It changes exactly ONE thing
    relative to `mpc_selfcal`: instead of tightening the mean-exposure cap (a blunt, scalar
    instrument), it imposes a HARD cap on the daily CVaR of the loss - the tail-risk BUDGET -
    via the Rockafellar-Uryasev epigraph in `MPCController.solve`. Everything else (conformal
    intervals, the learned beta_t, the uncertainty ratio, costs) is inherited unchanged.

    budget_t = vol_mult * sigma_t * scale_t,
    scale_t = 1 / (1 + beta_t * max(ratio_t - 1, 0))   (floored at min_scale)

    * sigma_t is the std of equal-weight daily returns over the lookback: a no-look-ahead,
      vol-denominated budget that is stable across regimes (realized CVaR/vol is ~2.0-2.17).
    * beta_t is the learned self-calibration coefficient, and `ratio_t` the conformal/
      ensemble uncertainty ratio, so wide intervals or breaches both shrink the budget.

    The budget sits just below the observed CVaR/vol, so it is a LIVE constraint at full
    investment rather than a permanently slack one (the failure mode flagged in
    docs/experiment_log.md for large risk_aversion).

    NO LOOK-AHEAD: sigma_t is computed from `R = hist.values[-lookback:]` (the trailing window
    only); beta_t is settled through day t-1; the scenario shocks are fixed by seed.
    """

    def __init__(self, cfg, n_assets, name="mpc_tailbudget", seed=0, use_forecast=True):
        super().__init__(cfg, n_assets, name=name, seed=seed)
        cb = cfg["cvar_budget"]
        self.vol_mult = float(cb["vol_mult"])
        self.cb_alpha = float(cb["alpha"])
        self.cb_scenarios = int(cb["n_scenarios"])
        self.cb_min_scale = float(cb["min_scale"])
        # scenario seed is fixed and strategy-specific -> reproducible, still no look-ahead
        self.cb_seed = seed + 9973
        # `use_forecast=False` is the ISOLATING CONTROL (mpc_tailbudget_nofc): drop the
        # return forecast from the objective while keeping the tail-risk budget and the
        # whole uncertainty/self-calibration loop. See _blend_mu.
        self.use_forecast = use_forecast

    def _compute_budget(self, R, ratio, beta_t):
        sigma_t = float(np.std(R.mean(axis=1)))            # trailing equal-weight vol
        if sigma_t <= 0:
            return None
        base = self.vol_mult * sigma_t
        scale = cvar_budget_scale(ratio, beta_t, self.cb_min_scale)
        budget = base * scale
        self.last_cvar_budget = float(budget)
        return float(budget)

