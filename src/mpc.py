"""Multi-period MPC portfolio controller (CVXPY).

maximize   sum_k [ (mu_k - kappa*hw_k)' w_k  -  gamma/2 * w_k' Sigma w_k  -  c * ||w_k - w_{k-1}||_1 ]
subject to 0 <= w_k <= wmax,  sum(w_k) <= max_exposure,  [ sqrt(w_k' Sigma w_k) <= vol_cap ]
           [ CVaR_alpha(-(1/H) sum_k w_k' r_k^(s)) <= budget ]        <- tail-risk budget

Only the first step w_1 is applied; the problem is re-solved every day (receding horizon).
mu_k - kappa*hw_k is the worst case of a box uncertainty set (valid because w >= 0).

TAIL-RISK BUDGET (Rockafellar & Uryasev 2000). `budget` is a hard cap on the CVaR of the
H-step portfolio loss, over a FIXED scenario set. The R-U epigraph

    min_{eta}  eta + 1/(alpha(1-alpha)) * sum_s max(0, L_s - eta)

is imposed as a constraint with eta free, which is exactly CVaR_alpha(L) <= budget. Since
each scenario return is DRAWN as mu_k + L chol @ z (fixed z), the path loss L_s is LINEAR
in w, so this adds only linear constraints and auxiliary variables: the problem remains a
convex QP, not an SOCP.

No look-ahead: the scenario shocks z are generated once from a fixed seed and are pure
exogenous randomness - they carry no information about future returns and are never refit.
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


def scenario_returns(mu_HN, L, n_scenarios, seed):
    """Fixed Gaussian scenario set: r^(s)_k = mu_k + L z^(s),  z^(s) ~ N(0, I) iid.

    Returned shape (S, H, N). The shocks are drawn ONCE from `seed` and reused every day,
    so they are exogenous randomness, never information about future returns. Because z is
    fixed, the path loss is linear in w and the R-U reformulation stays a QP.
    """
    H, N = mu_HN.shape
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((int(n_scenarios), H, N))
    return mu_HN[None, :, :] + np.einsum("ij,shj->shi", L, z)


def cvar_of_paths(path_losses, alpha):
    """Empirical CVaR_alpha of a 1-D loss sample (loss = -return, larger is worse).

    Because losses are the NEGATIVE of returns, the tail is the UPPER end of the sample:
    the threshold is the (1-alpha) quantile, and the risk measure is the mean loss above it.
    (Using quantile(alpha) would silently measure the best-case tail instead.)
    """
    q = float(np.quantile(path_losses, 1.0 - alpha))
    tail = path_losses[path_losses >= q]
    return float(tail.mean()) if tail.size else q


def cvar_budget_scale(ratio, beta, min_scale):
    """Tighten the tail-risk budget with the same learned beta_t and uncertainty ratio.

    Same shape as `tighten_limits` so the two mechanisms stay comparable: budget = base *
    1/(1 + beta*(ratio-1)). A ratio of 1 (normal uncertainty) leaves the budget untouched;
    breaches and wide intervals both shrink it.
    """
    return max(min_scale, 1.0 / (1.0 + beta * max(ratio - 1.0, 0.0)))


class MPCController:
    def __init__(self, horizon=5, risk_aversion=5.0, cost_bps=10.0, max_weight=0.30,
                 forecast_persistence=0.6):
        self.H = horizon
        self.gamma = risk_aversion
        self.c = cost_bps / 1e4
        self.wmax = max_weight
        self.persistence = forecast_persistence
        self.last_cvar = None
        self.last_status = None

    def solve(self, w0, mu, cov, halfwidth=None, kappa=0.0,
              max_exposure=1.0, vol_cap=None, risk_aversion=None,
              cvar_budget=None, cvar_alpha=0.05, n_scenarios=200, scenario_seed=0) -> np.ndarray:
        """Solve one day of the receding-horizon problem and return w_1.

        `cvar_budget` (a positive daily-loss fraction) enables the tail-risk budget via
        the R-U epigraph; when None the problem is exactly the previous mean-variance QP.
        """
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

        cvar_slack = None
        if cvar_budget is not None and cvar_budget > 0:
            # Scenario set is FIXED given the seed -> loss is linear in W[0].
            # We constrain the FIRST-STEP (daily) loss because (a) only w_1 is executed and
            # (b) it matches the cvar_daily metric the study reports, so "budget" and
            # "measured tail risk" are the same quantity.
            R = scenario_returns(mu, L, n_scenarios, scenario_seed)          # (S,H,N)
            S = R.shape[0]
            R1 = R[:, 0, :]                                                 # (S,N)
            loss_s = -(R1 @ W[0])                                           # (S,) positive=loss
            # Rockafellar-Uryasev epigraph as a CONSTRAINT with eta free:
            #   CVaR_alpha(L) = min_eta { eta + 1/(alpha(1-alpha)) * E[(L-eta)_+] }
            # The expectation is the empirical MEAN over the S scenarios, so the coefficient
            # is kappa_ru/S. Omitting the 1/S inflates the term S-fold and makes the
            # constraint badly conditioned (solver returns optimal_inaccurate and VIOLATES
            # the budget instead of binding it).
            eta = cp.Variable()
            u = cp.Variable(S, nonneg=True)
            kappa_ru = 1.0 / (cvar_alpha * (1.0 - cvar_alpha) * S)
            cons += [u >= loss_s - eta,
                     eta + kappa_ru * cp.sum(u) <= cvar_budget]
            cvar_slack = (eta, u)

        prob = cp.Problem(cp.Maximize(obj), cons)
        status = "unknown"
        if cvar_budget is not None:
            # Tight tolerances: the whole point of a budget is that it is RESPECTED, and the
            # default solver happily returns solutions ~20% over budget ("optimal_inaccurate").
            # OSQP stalls on this epigraph (user_limit) - CLARABEL handles it reliably.
            try:
                prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-10, tol_gap_rel=1e-10,
                           tol_feas=1e-10, max_iter=500)
                status = prob.status
            except Exception as e:
                warnings.warn(f"CLARABEL failed ({e}); falling back to default solver")
                prob.solve()
                status = prob.status
        else:
            try:
                prob.solve()
                status = prob.status
            except Exception as e:                    # solver failure
                warnings.warn(f"MPC solve failed: {e}")
                status = "solver_exception"
        if W.value is None or status not in ("optimal", "optimal_inaccurate"):
            self.last_status = status
            if cvar_budget is not None:
                # An INFEASIBLE tail budget means no portfolio meets it. Falling back to
                # w0 (fully invested) would be the exact opposite of risk-off -> cash.
                warnings.warn(f"MPC status={status} (cvar_budget={cvar_budget}); holding cash")
                return np.zeros(N)
            warnings.warn(f"MPC status={status}; holding current weights")   # legacy path
            return np.asarray(w0, dtype=float)

        if cvar_slack is not None:
            # Persist realised scenario CVaR (the constrained, daily quantity) so callers
            # can log budget utilisation = last_cvar / cvar_budget.
            w_val = np.clip(W.value[0], 0, None)
            self.last_cvar = cvar_of_paths(-(R1 @ w_val), cvar_alpha)
        self.last_status = status
        return np.clip(W.value[0], 0, None)
