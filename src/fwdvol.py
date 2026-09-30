"""Forward-looking volatility forecast: jump-augmented GARCH(1,1).

WHY THIS EXISTS
---------------
The paper's negative result (docs/experiment_log.md, 2026-10-01) is that the MPC's tail
budget is denominated in TRAILING vol, so on a volatility jump it is sized from stale risk
and loses to volatility-targeting. Beating a trailing-vol rule on a jump needs a
FORWARD-looking risk signal. This module is that signal.

We use a GARCH(1,1) with Student-t innovations plus a jump component. The reason GARCH and
not a rolling standard deviation: conditional variance

    h_{t+1} = omega + alpha * eps_t^2 + beta * h_t

reacts to the shock eps_t IMMEDIATELY, and the forecast h_{t+1} is a statement about the
FUTURE distribution rather than a summary of the past sample. A rolling 60-day vol needs
~60 days to absorb a jump; GARCH absorbs it the same day. That is precisely the mechanism
that made the trailing-vol budget lose.

The Student-t innovations give fat tails so a single crisis day does not get normalized
away, and the jump term adds a second, heavier component on top (Bates-style), which lets
the model distinguish "high-vol-but-continuous" from "there was an actual jump".

NOTHING here looks ahead. `forecast_variance` takes only the history it is given, and all
parameters are re-estimated on a trailing window of that history, exactly like every other
estimator in this project. See tests/test_fwdvol.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

__all__ = ["GarchTParams", "FwdVolForecaster", "fit_garch_t", "garch_t_negloglik",
           "garch_t_forecast_variance"]


# ---------------------------------------------------------------------------- fitting

def _pack(p):
    """Map unconstrained optimizer coords -> GARCH parameters.

    Only ALPHA and BETA are exponentiated. They are constrained to (0,1) and their sum must
    stay < 1, which a log-parameterization handles cleanly. OMEGA and NU are already in
    natural units in `p` and are passed through unchanged: exponentiating them as well would
    silently turn a variance of 0.05 into something else and a nu of 8 into e^8 ~ 3000.
    """
    omega, alpha, beta, nu = p
    return float(omega), float(np.exp(alpha)), float(np.exp(beta)), float(nu)


def garch_t_negloglik(p, eps, jump=None, jump_scale=0.0):
    """Negative log-likelihood of a GARCH(1,1)-t with an optional jump-scaled shock.

    `p` = (omega, log alpha, log beta, nu): omega and nu in NATURAL units, alpha/beta
    log-parameterized (see `_pack`). Constrains alpha+beta < 1 by returning +inf when
    violated, so the optimizer cannot wander into an explosive-variance region.

    The jump enters as a MULTIPLIER on the shock's variance contribution,

        alpha * eps^2 * (1 + jump_scale * 1{jump})

    not as an additive constant. An additive `jump**2` term would inject ~1.0 into a daily
    variance of order 1e-3 and swamp the GARCH dynamics entirely, driving the forecast to
    the cap. Scaling the SHOCK keeps the jump on the same units as the rest of the model.
    """
    omega, alpha, beta, nu = _pack(p)
    if not (0 <= alpha < 1 and 0 <= beta < 1 and alpha + beta < 0.9995):
        return np.inf
    if nu <= 2.05 or nu > 100 or omega <= 0:
        return np.inf
    n = len(eps)
    h = np.empty(n)
    h[0] = max(np.var(eps[:min(n, 50)]), 1e-10)
    ll = 0.0
    js = 1.0 + jump_scale * float(jump[0] if jump is not None else 0.0)
    for t in range(1, n):
        h[t] = omega + alpha * (eps[t - 1] ** 2) * js + beta * h[t - 1]
        if h[t] <= 1e-14 or not np.isfinite(h[t]):
            return np.inf
        e = eps[t] / np.sqrt(h[t])
        # accumulate the LOG-LIKELIHOOD and negate it at the end, so the optimizer
        # MINIMIZES negative log-likelihood. Getting this sign backwards makes it maximize
        # the loss and it returns garbage parameters.
        ll += gammaln((nu + 1) / 2) - gammaln(nu / 2) \
            - 0.5 * np.log(np.pi * (nu - 2)) - 0.5 * np.log(h[t]) \
            - ((nu + 1) / 2) * np.log1p(e * e / (nu - 2))
        if not np.isfinite(ll):
            return np.inf
        js = 1.0 + jump_scale * (jump[t] if jump is not None else 0.0)
    # TOTAL log-likelihood, not per-observation: the GARCH likelihood is very flat along the
    # alpha+beta ridge, and dividing by n shrinks the objective until the finite-difference
    # gradient is numerical noise and the optimizer never moves off its start point.
    return -ll


def fit_garch_t(r: np.ndarray, jump: np.ndarray | None = None, jump_scale=0.0):
    """MLE fit of GARCH(1,1)-t. Returns (omega, alpha, beta, nu).

    The GARCH likelihood is notoriously flat along the alpha+beta ridge: a gradient-based
    optimizer started at one point usually crawls and returns (almost) its start value. We
    therefore do a deterministic GRID over persistence (alpha+beta) and volatility
    clustering (alpha), and warm-start the local optimizer from the best grid point. This is
    slower but reliable, and it is a real fit rather than a fallback.
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 80:
        v = max(np.var(r), 1e-10) if len(r) > 1 else 1e-10
        return (v * 0.05, 0.08, 0.90, 8.0)
    scale = np.std(r)
    e = r / scale                       # scale-free: the optimizer is far better conditioned
    v0 = max(np.var(e), 1e-10)

    def obj(p):
        return garch_t_negloglik(p, e, jump, jump_scale)

    # --- deterministic grid over (persistence, alpha), with nu fixed at a sane default
    best_x, best_f = None, np.inf
    for pers in (0.90, 0.95, 0.98, 0.995):
        for frac in (0.02, 0.05, 0.10, 0.20, 0.35):
            alpha = min(frac, pers * 0.9)
            beta = pers - alpha
            if beta <= 0:
                continue
            # unconditional variance of a GARCH is omega/(1-pers); set it to 5% of sample var
            x = np.array([0.05 * v0 * (1 - pers), np.log(alpha), np.log(beta), 8.0])
            f = obj(x)
            if np.isfinite(f) and f < best_f:
                best_f, best_x = f, x

    if best_x is None:                  # grid found nothing feasible -> naive fallback
        return (v0 * 0.05 * scale ** 2, 0.08, 0.90, 8.0)

    # omega and nu are NATURAL units; alpha/beta are log units (see _pack)
    bounds = [(1e-8 * v0, 10.0 * v0), (np.log(1e-4), np.log(0.6)),
              (np.log(1e-4), np.log(0.999)), (2.1, 60.0)]
    for start in (best_x, np.array([0.02 * v0, np.log(0.05), np.log(0.93), 15.0])):
        try:
            res = minimize(obj, start, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 800, "ftol": 1e-12, "gtol": 1e-8})
        except Exception:
            continue
        if not (np.isfinite(res.fun) and res.fun < best_f):
            continue
        cand = _pack(res.x)
        # GUARD: L-BFGS-B on this flat, badly scaled surface sometimes walks a perfectly good
        # grid point down onto the LOWER BOUNDS of alpha/beta, where the objective is
        # (numerically) better but the model is degenerate - it never builds variance from a
        # shock, so the "forward" forecast is flat and useless. Reject any candidate whose
        # parameters sit on a bound, and fall back to the grid point.
        at_bound = (cand[1] <= 1e-4 * 1.001 or cand[2] <= 1e-4 * 1.001
                    or cand[3] <= 2.1 * 1.001 or cand[0] <= 1e-8 * v0 * 1.001)
        if at_bound:
            continue
        best_f, best_x = res.fun, res.x
    omega, alpha, beta, nu = _pack(best_x)
    return (omega * scale ** 2, alpha, beta, nu)


# ---------------------------------------------------------------------------- forecast

def garch_t_forecast_variance(params, eps, jump=None, jump_scale=0.0):
    """One-step-ahead conditional variance h_{t+1} given the fitted params and history.

    This is the forward-looking object: a variance for TOMORROW, not a summary of the past.
    """
    omega, alpha, beta, nu = params
    e = np.asarray(eps, dtype=float)
    h = max(np.var(e[:min(len(e), 50)]), 1e-10)
    # Iterate over EVERY observation, including the last one. h_{t+1} depends on eps_t, so
    # stopping at len(e)-1 would ignore the most recent return - exactly the shock a
    # one-step-ahead forecast is supposed to react to - and the forecast would be
    # insensitive to the newest data.
    for t in range(len(e)):
        js = 1.0 + jump_scale * (jump[t] if jump is not None else 0.0)
        h = omega + alpha * (e[t] ** 2) * js + beta * h
    return max(h, 1e-14)


# ---------------------------------------------------------------------------- forecaster

class FwdVolForecaster:
    """Trailing-window GARCH-t forecaster producing a 1-step conditional VOLATILITY.

    `decide_sigma` returns an ANNUALIZED forward volatility for the next period, the same
    units as the trailing-vol baseline, so the two are directly comparable. Re-estimated
    every `refit_every` days on a trailing window (cheap and stable) rather than every day.
    """

    def __init__(self, lookback=1000, refit_every=20, trading_days=252,
                 jump_quantile=0.975, jump_scale=3.0, min_obs=250,
                 vol_floor=0.02, vol_cap=1.5):
        self.lookback = int(lookback)
        self.refit_every = int(max(1, refit_every))
        self.trading_days = int(trading_days)
        self.jump_quantile = float(jump_quantile)
        self.jump_scale = float(jump_scale)
        self.min_obs = int(min_obs)
        self.vol_floor = float(vol_floor)
        self.vol_cap = float(vol_cap)
        self._cache = None          # (n_obs_when_fitted, params)
        self.last_params = None
        self.last_annual_vol = None
        self.n_fits = 0

    # -- the signal ---------------------------------------------------------
    def forward_vol(self, r: np.ndarray) -> float:
        """Annualized 1-step conditional vol from the history `r` (daily returns)."""
        r = np.asarray(r, dtype=float)
        r = r[np.isfinite(r)]
        if len(r) < self.min_obs:
            return self._clip(np.sqrt(max(float(np.mean(r ** 2)), 1e-10) * self.trading_days))
        w = r[-self.lookback:]
        # Degenerate history (all zeros, or a constant): there is no volatility to model and
        # std(w) is 0, which would make the scale-free fit divide by zero and return nan.
        if float(np.std(w)) <= 0.0:
            return self._clip(float(np.sqrt(max(float(np.mean(w ** 2)), 1e-10)
                                          * self.trading_days)))
        # jump indicator from realized |returns| in the history only (no look-ahead)
        thr = float(np.quantile(np.abs(w), self.jump_quantile))
        jump = (np.abs(w) >= thr).astype(float) if thr > 0 else np.zeros(len(w))
        if self._cache is None or (len(w) - self._cache[0]) >= self.refit_every:
            self._cache = (len(w), fit_garch_t(w, jump, self.jump_scale))
            self.n_fits += 1
        params = self._cache[1]
        self.last_params = params
        s = float(np.sqrt(garch_t_forecast_variance(params, w, jump, self.jump_scale)))
        if not np.isfinite(s):
            s = float(np.std(w))
        return self._clip(s * np.sqrt(self.trading_days))

    def _clip(self, ann: float) -> float:
        """Clip to the configured band and never return nan/inf."""
        if not np.isfinite(ann):
            ann = self.vol_floor
        v = float(np.clip(ann, self.vol_floor, self.vol_cap))
        self.last_annual_vol = v
        return v

    # -- convenience --------------------------------------------------------
    def decide_sigma(self, hist: pd.DataFrame) -> float:
        """Forward annualized vol of the equal-weight portfolio implied by `hist`."""
        n = hist.shape[1]
        return self.forward_vol(hist.values @ np.full(n, 1.0 / n))


import pandas as pd  # noqa: E402  (kept last: only needed for a type hint)
