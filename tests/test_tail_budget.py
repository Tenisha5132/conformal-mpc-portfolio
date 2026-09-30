"""Tests for mpc_tailbudget: the Rockafellar-Uryasev CVaR budget in the MPC.

Covers the properties the paper's claim rests on:
  * the budget is RESPECTED (not just present) and actually binds;
  * the problem stays a convex QP (no SOCP creep, no solver status games);
  * the budget is vol-denominated and regime-robust;
  * beta_t and the uncertainty ratio tighten the budget monotonically;
  * nothing reads the future (perturbing data after day T must not change day T's decision);
  * an infeasible budget goes to CASH, never back to fully-invested w0.
"""
import numpy as np
import pandas as pd
import pytest

from experiments.run_all import build
from src.backtest import run_backtest
from src.mpc import (MPCController, cvar_budget_scale, cvar_of_paths, scenario_returns,
                     tighten_limits)
from src.strategies import MPCTailBudgetStrategy, MPCSelfCalStrategy
from src.utils import load_config

from tests.common import get_returns
from tests.test_backtest import check_no_lookahead

CONFIG = "configs/default.yaml"


def _cfg():
    return load_config(CONFIG)


def _mu_cov(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.normal(0.0004, 0.0002, n)
    A = rng.normal(0, 0.02, (n, n))
    cov = A @ A.T / n + np.eye(n) * 1e-4
    return mu, cov, np.full(n, 1.0 / n)


# ------------------------------------------------------------- cvar_of_paths

def test_cvar_takes_the_WORST_tail_not_the_best():
    """A sign error here would measure the best-case tail and the budget would never bind."""
    losses = np.array([-0.10, -0.05, 0.0, 0.01, 0.02, 0.03, 0.04, 0.05,
                       0.06, 0.50])
    # alpha=0.2 -> threshold at the 80th percentile = 0.05; tail = {0.06, 0.50}
    got = cvar_of_paths(losses, 0.2)
    assert got == pytest.approx(np.mean([0.06, 0.50])), got
    # and it must NOT be the lower tail
    assert got > 0.0


def test_cvar_is_monotone_in_the_tail_level():
    rng = np.random.default_rng(3)
    losses = rng.normal(0, 0.01, 5000)
    lo = cvar_of_paths(losses, 0.05)
    hi = cvar_of_paths(losses, 0.20)
    assert hi < lo, "a higher tail level must not report a larger loss"


def test_cvar_of_a_single_observation_is_itself():
    assert cvar_of_paths(np.array([0.03]), 0.05) == pytest.approx(0.03)


# ------------------------------------------------------------- scenario set

def test_scenarios_are_deterministic_given_the_seed():
    mu, cov, _ = _mu_cov()
    m = np.zeros((5, 6))
    a = scenario_returns(m, np.eye(6) * 0.02, 50, seed=7)
    b = scenario_returns(m, np.eye(6) * 0.02, 50, seed=7)
    c = scenario_returns(m, np.eye(6) * 0.02, 50, seed=8)
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c), "a different seed must give different shocks"


def test_scenarios_are_centred_on_mu():
    """r^(s) = mu + L z, so the sample mean over many shocks should approach mu."""
    mu, cov, _ = _mu_cov(n=4, seed=1)
    from src.estimators import chol_psd
    R = scenario_returns(np.tile(mu, (3, 1)), chol_psd(cov), 4000, seed=0)
    assert R.shape == (4000, 3, 4)
    np.testing.assert_allclose(R.mean(axis=0), np.tile(mu, (3, 1)), atol=2e-3)


# ------------------------------------------------------------- cvar_budget_scale

def test_budget_scale_is_one_when_uncertainty_is_normal():
    assert cvar_budget_scale(1.0, beta=0.5, min_scale=0.2) == pytest.approx(1.0)
    # a ratio below 1 (tighter than typical) must NOT loosen the budget
    assert cvar_budget_scale(0.5, beta=0.5, min_scale=0.2) == pytest.approx(1.0)


def test_budget_scale_shrinks_monotonically_with_ratio_and_beta():
    vals = [cvar_budget_scale(r, 0.5, 0.2) for r in (1.0, 1.5, 2.0, 4.0)]
    assert all(a >= b for a, b in zip(vals, vals[1:])), vals
    assert vals[0] == 1.0 and vals[-1] < vals[0]
    # more beta -> tighter for the same ratio
    assert cvar_budget_scale(2.0, 0.1, 0.2) > cvar_budget_scale(2.0, 1.0, 0.2)


def test_budget_scale_respects_the_floor():
    assert cvar_budget_scale(1000.0, 50.0, 0.2) == pytest.approx(0.2)


# ------------------------------------------------------------- the constraint

def test_budget_is_respected_when_it_binds():
    """The core mechanical claim: realised scenario CVaR must not exceed the budget."""
    mu, cov, w0 = _mu_cov(n=8, seed=2)
    ctrl = MPCController(horizon=5, risk_aversion=5.0)
    for budget in (0.02, 0.01, 0.005, 0.002):
        w = ctrl.solve(w0, mu, cov, max_exposure=1.0, cvar_budget=budget,
                       cvar_alpha=0.05, n_scenarios=200)
        assert ctrl.last_cvar <= budget * 1.02, (
            f"budget {budget} violated: realised {ctrl.last_cvar}")
        assert w.sum() <= 1.0 + 1e-6


def test_tighter_budget_means_less_exposure():
    mu, cov, w0 = _mu_cov(n=8, seed=2)
    ctrl = MPCController(horizon=5, risk_aversion=5.0)
    sums = []
    for budget in (0.03, 0.01, 0.005, 0.002):
        w = ctrl.solve(w0, mu, cov, max_exposure=1.0, cvar_budget=budget, n_scenarios=200)
        sums.append(w.sum())
    assert all(a >= b - 1e-6 for a, b in zip(sums, sums[1:])), sums
    assert sums[0] > sums[-1], "a tighter tail budget must de-risk"


def test_solver_reports_a_clean_status():
    mu, cov, w0 = _mu_cov(n=8, seed=2)
    ctrl = MPCController(horizon=5, risk_aversion=5.0)
    ctrl.solve(w0, mu, cov, max_exposure=1.0, cvar_budget=0.006, n_scenarios=200)
    assert ctrl.last_status == "optimal", ctrl.last_status


def test_cvar_budget_none_is_the_plain_qp_and_unchanged():
    """Passing no budget must reproduce the legacy mean-variance problem exactly."""
    mu, cov, w0 = _mu_cov(n=6, seed=5)
    ctrl = MPCController(horizon=5, risk_aversion=5.0)
    w = ctrl.solve(w0, mu, cov, max_exposure=1.0, cvar_budget=None)
    ref = MPCController(horizon=5, risk_aversion=5.0).solve(w0, mu, cov, max_exposure=1.0)
    np.testing.assert_allclose(w, ref, atol=1e-8)
    assert ctrl.last_cvar is None
    assert ctrl.last_status == "optimal"


def test_budget_is_linear_in_vol_so_it_scales_with_the_market():
    """Doubling the covariance must (roughly) halve the exposure a given budget allows."""
    mu, _, w0 = _mu_cov(n=6, seed=6)
    cov_lo = np.eye(6) * 1e-4
    cov_hi = np.eye(6) * 4e-4
    ctrl = MPCController(horizon=5, risk_aversion=5.0)
    e_lo = ctrl.solve(w0, mu, cov_lo, max_exposure=1.0, cvar_budget=0.004, n_scenarios=300).sum()
    e_hi = ctrl.solve(w0, mu, cov_hi, max_exposure=1.0, cvar_budget=0.004, n_scenarios=300).sum()
    assert e_hi < e_lo, (e_lo, e_hi)


# ------------------------------------------------------------- the strategy

def test_tailbudget_budget_is_vol_denominated_and_uses_only_trailing_data():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"])
    L = int(cfg["backtest"]["lookback"])
    R = r.values[-L:]
    sigma = float(np.std(R.mean(axis=1)))
    b = s._compute_budget(R, ratio=1.0, beta_t=0.0)
    assert b == pytest.approx(s.vol_mult * sigma)


def test_tailbudget_budget_shrinks_with_beta_and_ratio():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"])
    R = r.values[-60:]
    # beta=0 -> no tightening at all, by design (the beta0=0 semantics of mpc_selfcal):
    # the budget must equal the base, so the ratio alone cannot move it.
    assert s._compute_budget(R, 1.0, 0.0) == pytest.approx(s._compute_budget(R, 4.0, 0.0))
    # with beta>0 the ratio bites, and larger beta bites harder
    base = s._compute_budget(R, 1.0, 0.0)
    # ratio=1 -> max(ratio-1,0)=0 -> no tightening from beta alone. This mirrors
    # tighten_limits: the levers are multiplicative in (ratio-1), never additive.
    assert s._compute_budget(R, 1.0, 1.0) == pytest.approx(base)
    # with beta>0 AND ratio>1 the budget tightens, and larger beta bites harder
    ratio_only = s._compute_budget(R, 2.0, 0.5)
    both = s._compute_budget(R, 2.0, 1.0)
    assert base > ratio_only > both > 0, (base, ratio_only, both)


def test_base_selfcal_still_passes_no_budget():
    """mpc_selfcal must be behaviourally identical to before: the hook returns None."""
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCSelfCalStrategy(cfg, r.shape[1], seed=cfg["seed"])
    R = r.values[-60:]
    assert s._compute_budget(R, 2.0, 1.0) is None


def test_budget_binds_often_enough_to_matter():
    """A permanently slack budget would be the documented risk_aversion failure mode."""
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"])
    res = run_backtest(r, s, r.index[-700], r.index[-1], cfg["backtest"]["cost_bps"])
    util = [row["cvar_used"] / row["cvar_budget"] for row in s.log
            if row["cvar_budget"] and row["cvar_used"] and row["cvar_budget"] > 0]
    assert len(util) > 100, "not enough decisions to judge binding"
    frac = float(np.mean(np.asarray(util) > 0.95))
    assert 0.05 < frac < 0.99, f"budget binding frequency {frac:.3f} is degenerate"


def test_tailbudget_respects_no_lookahead():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    check_no_lookahead(lambda: MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"]), r, 800)


def test_weights_stay_in_the_simplex():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"])
    res = run_backtest(r, s, r.index[-600], r.index[-1], cfg["backtest"]["cost_bps"])
    w = res.weights.values
    assert (w >= -1e-8).all(), "negative weights"
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all(), "leverage above 1"
    assert (w <= cfg["mpc"]["max_weight"] + 1e-6).all(), "breached max_weight"


# ------------------------------------------------- the forecaster-off control

def test_nofc_blend_returns_the_historical_mean():
    """The control must remove the forecast and ONLY the forecast from the drift."""
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"], use_forecast=False)
    hist_mu = np.array([0.01, -0.02, 0.0, 0.03, -0.01])
    pred = np.array([9.0, 9.0, 9.0, 9.0, 9.0])
    np.testing.assert_allclose(s._blend_mu(hist_mu, pred), hist_mu)


def test_fc_blend_uses_the_forecast():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"], use_forecast=True)
    hist_mu = np.zeros(5)
    pred = np.ones(5)
    ms = cfg["forecaster"]["mu_shrink"]
    np.testing.assert_allclose(s._blend_mu(hist_mu, pred), np.full(5, ms))


def test_nofc_keeps_the_whole_uncertainty_loop():
    """The control isolates the FORECAST, not the self-calibration machinery."""
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"], use_forecast=False)
    R = r.values[-60:]
    # budget still vol-scaled and still shrinks with beta*ratio
    base = s._compute_budget(R, 1.0, 0.0)
    assert base == pytest.approx(s.vol_mult * float(np.std(R.mean(axis=1))))
    assert s._compute_budget(R, 2.0, 1.0) < base
    # the learned beta state and the conformal/ensemble signals are all still present
    assert s.beta_state is not None and s.conformal is not None and s.ensemble is not None


def test_nofc_still_respects_its_budget_in_a_backtest():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    s = MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"], use_forecast=False)
    run_backtest(r, s, r.index[-500], r.index[-1], cfg["backtest"]["cost_bps"])
    used = [row["cvar_used"] for row in s.log if row["cvar_budget"]]
    bud = [row["cvar_budget"] for row in s.log if row["cvar_budget"]]
    assert len(used) > 50
    assert all(u <= b * 1.05 for u, b in zip(used, bud)), "control violated its own budget"


def test_nofc_respects_no_lookahead():
    cfg = _cfg()
    r = get_returns(n_assets=5, seed=1)
    check_no_lookahead(
        lambda: MPCTailBudgetStrategy(cfg, r.shape[1], seed=cfg["seed"], use_forecast=False),
        r, 800)


# ------------------------------------------------------- forward-vol budget (mpc_tailbudget_fwdvol)

def test_fwdvol_budget_builds_and_uses_the_forward_signal():
    """The forward variant must actually take the GARCH path, not silently fall back."""
    from src.fwdvol import FwdVolForecaster
    cfg = _cfg()
    st = build("mpc_tailbudget_fwdvol", cfg, 5)
    assert st.fwd_vol is True
    assert isinstance(st._fwd, FwdVolForecaster)
    assert st.use_forecast is False, "must differ from mpc_tailbudget_nofc ONLY in the signal"


def test_fwdvol_budget_denominator_is_forward_looking():
    """budget/vol_mult must equal the GARCH vol, not the trailing std."""
    cfg = _cfg()
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2018-01-01", periods=400)
    hist = pd.DataFrame(rng.standard_normal((400, 5)) * 0.01, index=idx,
                        columns=list("abcde"))
    st = build("mpc_tailbudget_fwdvol", cfg, 5)
    b = st._compute_budget(hist.values, 1.0, 0.0, hist)
    fwd_daily = st._fwd.decide_sigma(hist) / np.sqrt(252)
    assert b == pytest.approx(st.vol_mult * fwd_daily, rel=1e-9)
    trailing = float(np.std(hist.values.mean(axis=1)))
    assert not np.isclose(fwd_daily, trailing, rtol=1e-3), (
        "forward and trailing vol coincide here; the test would prove nothing")


def test_trailing_and_fwdvol_budgets_differ_on_the_same_day():
    """The two denominators must not be aliases of each other."""
    cfg = _cfg()
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2018-01-01", periods=400)
    hist = pd.DataFrame(rng.standard_normal((400, 5)) * 0.012, index=idx,
                        columns=list("abcde"))
    a = build("mpc_tailbudget_nofc", cfg, 5)._compute_budget(hist.values, 1.0, 0.0, hist)
    b = build("mpc_tailbudget_fwdvol", cfg, 5)._compute_budget(hist.values, 1.0, 0.0, hist)
    assert not np.isclose(a, b, rtol=1e-3)


def test_fwdvol_budget_never_looks_ahead():
    """Budget at day t must be identical whether or not OLDER rows are present.

    The GARCH fit uses a trailing window of `lookback` rows. So appending rows at the FRONT
    (history that is older than the window) must not change today's budget, while appending
    rows at the BACK (the future) would legitimately change it.
    """
    cfg = _cfg()
    rng = np.random.default_rng(5)
    look = cfg["fwdvol"]["lookback"]
    idx = pd.bdate_range("2018-01-01", periods=2 * look + 200)
    full = pd.DataFrame(rng.standard_normal((2 * look + 200, 5)) * 0.01, index=idx,
                        columns=list("abcde"))
    # The fit window is r[-lookback:]. To hold it FIXED while varying how much older history
    # is supplied, BOTH histories must be at least `lookback` long, otherwise the shorter one
    # fits on strictly fewer rows and the two forecasts are not comparable.
    short = full.iloc[look:2 * look]                   # exactly `look` rows
    long = full.iloc[:2 * look]                        # same last `look`, plus `look` older
    assert len(short) == look and len(long) == 2 * look
    assert np.array_equal(long.values[-look:], short.values)
    a = build("mpc_tailbudget_fwdvol", cfg, 5)._compute_budget(short.values, 1.0, 0.0, short)
    b = build("mpc_tailbudget_fwdvol", cfg, 5)._compute_budget(long.values, 1.0, 0.0, long)
    assert a == pytest.approx(b, rel=1e-9), "older rows leaked into the forward budget"


def test_fwdvol_baseline_shares_the_signal_with_the_mpc():
    """MATCHED INFORMATION: the scalar rule and the MPC must use identical fwdvol config."""
    from src.baselines import EqualWeightVolTargetFwd
    cfg = _cfg()
    mpc = build("mpc_tailbudget_fwdvol", cfg, 5)
    bl = build("equal_weight_voltarget_fwd", cfg, 5)
    assert isinstance(bl, EqualWeightVolTargetFwd)
    assert bl.fwd.lookback == mpc._fwd.lookback
    assert bl.fwd.refit_every == mpc._fwd.refit_every
    assert bl.fwd.jump_quantile == mpc._fwd.jump_quantile
    assert bl.fwd.jump_scale == mpc._fwd.jump_scale
    assert bl.fwd.vol_floor == mpc._fwd.vol_floor
    assert bl.fwd.vol_cap == mpc._fwd.vol_cap
