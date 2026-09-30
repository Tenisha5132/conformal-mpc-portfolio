"""Multi-period MPC portfolio controller (CVXPY).

maximize   sum_k [ (mu_k - kappa*hw_k)' w_k  -  gamma/2 * w_k' Sigma w_k  -  c * ||w_k - w_{k-1}||_1 ]
subject to 0 <= w_k <= wmax,  sum(w_k) <= max_exposure,  [ sqrt(w_k' Sigma w_k) <= vol_cap ]

Only the first step w_1 is applied; the problem is re-solved every day (receding horizon).
mu_k - kappa*hw_k is the worst case of a box uncertainty set (valid because w >= 0).
"""
import warnings

import cvxpy as cp
import numpy as np

from src.estimators import chol_psd


def _to_HN(x, H, N, persistence=1.0):
    """Broadcast a 1-D per-asset quantity across the horizon.

    The forecaster only produces a ONE-day-ahead forecast. Holding it flat for all
    H steps would reward a 1-day signal H times and make the controller H times
    bolder than intended, so the mean forecast decays by `persistence` per step.
    Set persistence=1.0 for quantities that genuinely persist (e.g. half-widths).
    """
    if x is None:
        return np.zeros((H, N))
    x = np.asarray(x, dtype=float)
    if x.ndim == 2:
        return x
    return x[None, :] * (persistence ** np.arange(H))[:, None]


def tighten_limits(hw_ratio, base_exposure, base_vol_cap, beta, min_exposure):
    """CORE NOVELTY (v1): shrink risk limits when forecast uncertainty spikes.

    hw_ratio = current interval width / its own recent typical width (1.0 = normal).
    scale = 1 / (1 + beta * max(hw_ratio - 1, 0))  -> 1 when normal, smaller when uncertain.
    """
    scale = 1.0 / (1.0 + beta * max(hw_ratio - 1.0, 0.0))
    exposure = max(min_exposure, base_exposure * scale)
    vol_cap = None if base_vol_cap is None else base_vol_cap * scale
    return exposure, vol_cap


class MPCController:
    def __init__(self, horizon=5, risk_aversion=5.0, cost_bps=10.0, max_weight=0.30,
                 forecast_persistence=0.6):
        self.H = horizon
        self.gamma = risk_aversion
        self.c = cost_bps / 1e4
        self.wmax = max_weight
        self.persistence = forecast_persistence

    def solve(self, w0, mu, cov, halfwidth=None, kappa=0.0,
              max_exposure=1.0, vol_cap=None, risk_aversion=None) -> np.ndarray:
        N, H = len(w0), self.H
        gamma = self.gamma if risk_aversion is None else risk_aversion
        mu = _to_HN(mu, H, N, self.persistence)
        hw = _to_HN(halfwidth, H, N, 1.0)
        L = chol_psd(cov)

        W = cp.Variable((H, N))
        obj, cons, prev = 0, [], np.asarray(w0, dtype=float)
        for k in range(H):
            w = W[k]
            obj += (mu[k] - kappa * hw[k]) @ w
            obj -= 0.5 * gamma * cp.sum_squares(L.T @ w)
            obj -= self.c * cp.norm1(w - prev)
            cons += [w >= 0, w <= self.wmax, cp.sum(w) <= max_exposure]
            if vol_cap is not None:
                cons.append(cp.norm(L.T @ w, 2) <= vol_cap)
            prev = w
        prob = cp.Problem(cp.Maximize(obj), cons)
        try:
            prob.solve()
        except Exception as e:                        # solver failure -> hold current weights
            warnings.warn(f"MPC solve failed: {e}")
            return np.asarray(w0, dtype=float)
        if W.value is None:
            warnings.warn("MPC infeasible/unsolved; holding weights")
            return np.asarray(w0, dtype=float)
        return np.clip(W.value[0], 0, None)
