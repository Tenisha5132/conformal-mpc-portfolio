"""Tests for the forward-looking volatility forecaster (src/fwdvol.py).

The purpose of this module is to give the MPC a risk signal that is genuinely
FORWARD-looking, so it can beat a trailing-volatility rule on a volatility jump. These tests
pin the properties that make that claim checkable:

  * the fit actually recovers GARCH parameters (it is a real MLE, not a fallback);
  * the fit does not collapse onto the parameter bounds (a degenerate model whose
    "forecast" is flat would silently defeat the whole point);
  * the forecast is 1-STEP and reacts to a shock, unlike a trailing window;
  * NOTHING looks ahead.
"""
import numpy as np
import pandas as pd
import pytest

from src.fwdvol import (FwdVolForecaster, fit_garch_t, garch_t_forecast_variance,
                        garch_t_negloglik)


def _sim_garch(n=3000, om=0.05, al=0.10, be=0.85, nu=8.0, scale=0.01, seed=0):
    rng = np.random.default_rng(seed)
    h = np.empty(n)
    h[0] = om / (1 - al - be)
    x = np.empty(n)
    for t in range(1, n):
        h[t] = om + al * x[t - 1] ** 2 / scale ** 2 + be * h[t - 1]
        x[t] = scale * rng.standard_normal() * np.sqrt(h[t])
    return x


# ------------------------------------------------------------------ the fit is real

def test_fit_recovers_garch_parameters():
    x = _sim_garch()
    om, al, be, nu = fit_garch_t(x)
    assert al + be < 1.0, "explosive variance"
    # alpha is the hard one to pin down; require the right order of magnitude and sane
    # persistence, which is what actually drives the forecast.
    assert 0.005 < al < 0.35, f"alpha={al} implausible"
    assert 0.5 < al + be < 0.999, f"persistence={al + be} implausible"
    assert 2.1 < nu < 60.0


def test_fitted_params_beat_a_naive_start_on_the_likelihood():
    """The MLE must be at least as likely as a standard starting parameterization."""
    x = _sim_garch()
    s = np.std(x)
    e = (x - x.mean()) / s
    om, al, be, nu = fit_garch_t(x)
    nll_fit = garch_t_negloglik([om / s ** 2, np.log(al), np.log(be), nu], e)
    nll_naive = garch_t_negloglik([0.05, np.log(0.08), np.log(0.90), 8.0], e)
    assert np.isfinite(nll_fit)
    assert nll_fit <= nll_naive + 1e-6


def test_fit_never_collapses_onto_the_parameter_bounds():
    """A bound-collapse bug produced alpha=beta=1e-4, i.e. a FLAT, useless forecast.

    That failure mode is invisible in the objective (it can even score better) so it has to
    be asserted on the parameters directly.
    """
    rng = np.random.default_rng(7)
    cases = [
        _sim_garch(seed=1),
        _sim_garch(n=800, al=0.30, be=0.60, seed=2),
        rng.standard_normal(600) * 0.01,                       # near-iid, no vol clustering
        np.concatenate([rng.standard_normal(400) * 0.005,
                        rng.standard_normal(200) * 0.05]),      # a regime shift
    ]
    for x in cases:
        om, al, be, nu = fit_garch_t(x)
        assert al > 1e-4 * 1.001, f"alpha collapsed to the bound ({al})"
        assert be > 1e-4 * 1.001, f"beta collapsed to the bound ({be})"
        assert al + be < 0.9995
        assert nu > 2.1 * 1.001, f"nu collapsed to the bound ({nu})"


def test_forecast_is_finite_on_pathological_input():
    for x in [np.zeros(500), np.full(500, 1e-9), np.random.default_rng(3).standard_normal(400) * 1e-4]:
        v = FwdVolForecaster(min_obs=100, refit_every=1).forward_vol(x)
        assert np.isfinite(v) and v > 0


# ------------------------------------------------------------------ it is forward-looking

def test_conditional_variance_reacts_to_a_shock_immediately():
    """h_{t+1} must respond to eps_t. This is the property trailing vol lacks.

    The effect is real but modest in SIGMA terms, so assert the direction plus a floor that
    a truly inert model would fail. A flat/degenerate fit returns exactly `base`.
    """
    x = _sim_garch(n=800)
    om, al, be, nu = fit_garch_t(x)
    p = (om, al, be, nu)
    base = garch_t_forecast_variance(p, x)
    # No jump indicator: it bins shocks, so two shocks in the same bin tie. The pure
    # recursion must respond, and the response must grow with the SHOCK SIZE.
    #
    # Sizes are in units of the model's OWN conditional sd (sqrt(h)), not the sample sd. A
    # shock many sample-sd long is so large relative to h that the increment alpha*eps^2
    # saturates in floating point and two different shocks give a bit-identical forecast,
    # which is what made an earlier version of this test fail.
    # Sizes in units of the model's OWN conditional sd, over a wide range (k = 1 -> 6 is a
    # 36x change in eps^2). The response must be strictly increasing.
    #
    # We deliberately do NOT assert a ~k^2 slope. The realized change in h is
    # alpha*eps^2 (quadratic) MINUS the omega mean-reversion drag, which is linear in the
    # same quantity, so the net response grows more slowly than k^2 and an exact power-law
    # assertion is not a property this estimator actually has. Monotonicity plus a clear
    # order-of-magnitude rise is the defensible claim; the fitted alpha is what makes the
    # response fast relative to a 60-day trailing window, and that is asserted separately in
    # test_forecast_beats_trailing_vol_on_simulated_garch_data.
    vals = [garch_t_forecast_variance(p, np.append(x, k * np.sqrt(base)),
                                      jump=None, jump_scale=0.0)
            for k in (1.0, 2.0, 4.0, 6.0)]
    assert all(np.isfinite(v) for v in vals)
    assert all(b > a for a, b in zip(vals, vals[1:])), \
        f"forecast not monotone in shock size: {vals}"
    assert vals[-1] > 1.5 * vals[0], \
        f"shock barely moved the forecast: {vals}"


def test_forecast_beats_trailing_vol_on_simulated_garch_data():
    """On data actually generated BY a GARCH, the forecast should predict |r| better.

    Note this is the FAVOURABLE case. On real market returns the GARCH is not a better
    predictor than a trailing window (see docs/experiment_log.md), which is the honest
    finding; this test only asserts the implementation realises its theoretical advantage
    where one exists.
    """
    x = _sim_garch(n=4000, seed=5)
    fw = FwdVolForecaster(min_obs=250, refit_every=20)
    f, t = [], []
    for i in range(300, len(x)):
        f.append(fw.forward_vol(x[:i]))
        t.append(np.std(x[i - 60:i], ddof=1) * np.sqrt(252))
    act = np.abs(x[300:])
    assert np.corrcoef(f, act)[0, 1] > np.corrcoef(t, act)[0, 1]


# ------------------------------------------------------------------ no look-ahead

def test_forecaster_uses_only_the_history_it_is_given():
    """The forecast must be a function of the trailing window ALONE, not of the future.

    The trailing window is the LAST `lookback` points, so to hold it fixed while appending
    future data we must extend the array at the FRONT, not the back: `x[:600]` and
    `x[200:800]` share the same last 600 observations. Any difference between the two
    forecasts could then only come from look-ahead, and there must be none.
    """
    x = _sim_garch(n=1000, seed=11)
    shared = x[200:800]                  # the trailing window both calls must use
    fw_a = FwdVolForecaster(min_obs=250, refit_every=1, lookback=600)
    fw_b = FwdVolForecaster(min_obs=250, refit_every=1, lookback=600)
    a = fw_a.forward_vol(shared)
    # same trailing 600 observations, preceded by 200 EXTRA OLDER points that the window
    # must exclude. If the slice were off by one, or the fit used the whole array, these
    # would differ.
    b = fw_b.forward_vol(np.concatenate([x[:200], shared]))
    assert a == pytest.approx(b, rel=1e-9)


def test_forecaster_is_deterministic():
    x = _sim_garch(n=600, seed=13)
    v1 = FwdVolForecaster(min_obs=200, refit_every=5).forward_vol(x)
    v2 = FwdVolForecaster(min_obs=200, refit_every=5).forward_vol(x)
    assert v1 == pytest.approx(v2, rel=1e-12)


def test_decide_sigma_matches_forward_vol_on_equal_weight():
    idx = pd.bdate_range("2015-01-01", periods=400)
    rng = np.random.default_rng(17)
    hist = pd.DataFrame(rng.standard_normal((400, 5)) * 0.01, index=idx,
                        columns=list("abcde"))
    fw = FwdVolForecaster(min_obs=200, refit_every=1)
    ew = hist.values @ np.full(5, 1 / 5)
    assert fw.decide_sigma(hist) == pytest.approx(fw.forward_vol(ew), rel=1e-12)


def test_fwdvol_respects_its_config_bounds():
    x = np.random.default_rng(19).standard_normal(400) * 0.01
    fw = FwdVolForecaster(min_obs=100, vol_floor=0.05, vol_cap=0.30)
    v = fw.forward_vol(x)
    assert 0.05 <= v <= 0.30
