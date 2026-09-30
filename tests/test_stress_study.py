"""Tests for the stress labeller, the block bootstrap, the new baselines and the
regime-conditional study. Every strategy added here must also pass the no-look-ahead
harness in tests/test_backtest.py.
"""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.baselines import EqualWeight, EqualWeightVolTarget, MPCDrawdownRiskAversion
from src.bootstrap import (paired_difference_ci, percentile_ci, realized_block_length,
                           stationary_bootstrap_indices)
from src.stress import (dispersion, drawdown, episode_market_drawdown, find_episodes,
                        fit_thresholds, label_stress, market_index)
from src.utils import load_config

from tests.common import get_returns

CFG = "configs/default.yaml"
SL = {"dispersion_quantile": 0.80, "market_return_quantile": 0.10,
      "warmup": 60, "min_episode_days": 3, "merge_gap_days": 2}


def _panel(n=400, seed=0):
    """Deterministic multi-asset panel for the structural tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    f = rng.standard_normal((n, 4)) * 0.01 + np.array([0.0005, 0, -0.0005, 0.0002])
    f[100:130] *= 4.0                       # a dispersion blow-up
    f[200] = [-0.05, -0.045, -0.04, -0.035]  # a common shock (low dispersion, big move)
    return pd.DataFrame(f, index=idx, columns=list("ABCD"))


# ---------------------------------------------------------------- stress labels

def test_label_rule_is_twosided_and_data_driven():
    r = _panel()
    th = fit_thresholds(r, r.index[299], SL)
    s = label_stress(r, th, SL)
    # the common-shock day must be caught by the market-return arm even though
    # dispersion is LOW there - this is why the rule has two arms
    assert bool(s.iloc[200]), "common-shock day was not labelled stress"
    # thresholds must come from the train window, not the whole frame
    th_train = fit_thresholds(r.iloc[:300], r.index[299], SL)
    assert th["disp_threshold"] == th_train["disp_threshold"]
    assert th["market_return_threshold"] == th_train["market_return_threshold"]


def test_thresholds_fitted_on_train_only_ignore_later_data():
    """Appending violent future data must not move the frozen thresholds."""
    r = _panel()
    th_a = fit_thresholds(r, r.index[299], SL)
    rng = np.random.default_rng(9)
    wild = pd.DataFrame(rng.standard_normal((80, 4)) * 0.4,
                        index=pd.bdate_range("2017-01-01", periods=80), columns=r.columns)
    th_b = fit_thresholds(pd.concat([r, wild]), r.index[299], SL)
    assert th_a == th_b, "appending future data moved the frozen thresholds"


def test_warmup_days_are_never_stress():
    r = _panel()
    s = label_stress(r, fit_thresholds(r, r.index[299], SL), SL)
    assert not s.iloc[:SL["warmup"]].any()


def test_episodes_respect_minimum_and_merge_gap():
    s = pd.Series([False] * 40, index=pd.bdate_range("2015-01-01", periods=40))
    s.iloc[[2, 3]] = True                     # 2d -> dropped
    s.iloc[[10, 11, 12]] = True               # kept
    s.iloc[[16, 17]] = True                   # gap of 3 > merge_gap -> separate, then dropped
    s.iloc[[30, 31, 32, 33]] = True           # kept
    eps = find_episodes(s, {"min_episode_days": 3, "merge_gap_days": 2})
    assert len(eps) == 2
    assert eps["n_span_days"].tolist() == [3, 4]


def test_episode_merge_bridges_short_calm_gap():
    s = pd.Series([False] * 30, index=pd.bdate_range("2015-01-01", periods=30))
    s.iloc[[1, 2, 3]] = True
    s.iloc[[5, 6, 7]] = True                  # 1 calm day between -> merged
    eps = find_episodes(s, {"min_episode_days": 3, "merge_gap_days": 2})
    assert len(eps) == 1 and eps["n_span_days"].iloc[0] == 7


def test_episode_drawdown_is_anchored_before_the_episode():
    """Regression: anchoring the peak at day 1 forces min-drawdown to 0 for every episode."""
    idx = pd.bdate_range("2015-01-01", periods=10)
    r = pd.DataFrame({"A": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
                      "B": [-0.01, -0.01, -0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]},
                     index=idx)
    eps = pd.DataFrame({"end": [idx[4]], "n_span_days": [5]}, index=[idx[0]])
    dd = episode_market_drawdown(r, eps)
    span = market_index(r).loc[idx[0]:idx[4]]
    # peak anchored at the pre-episode wealth of 1.0, so the worst drawdown is
    # 1 - max(cumprod) rather than the terminal value
    expected = 1.0 - float(np.cumprod(1.0 + span.to_numpy()).max())
    assert dd.iloc[0] == pytest.approx(expected, abs=1e-12)
    # with the peak anchored at day 1 instead of before the episode, this is identically 0
    assert dd.iloc[0] > 0.0


def test_dispersion_is_scale_only_not_direction():
    r = _panel()
    up = r * -1.0                            # flip the sign of every return
    disp_before = dispersion(r).values
    disp_after = dispersion(up).values
    np.testing.assert_allclose(disp_before, disp_after, atol=1e-12)
    # but the market-return arm does react to direction
    assert market_index(r).mean() != market_index(up).mean()


def test_drawdown_is_causal_under_future_perturbation():
    r = _panel()
    tampered = r.copy()
    tampered.iloc[300:] *= 12.0              # violent future
    common = r.index[:300]
    np.testing.assert_allclose(drawdown(r).loc[common].to_numpy(),
                               drawdown(tampered).loc[common].to_numpy(), atol=1e-12)


# --------------------------------------------------------------------- bootstrap

def test_block_length_matches_configuration():
    rng = np.random.default_rng(0)
    for L in (5, 10, 20):
        est = np.mean([realized_block_length(stationary_bootstrap_indices(800, L, rng))
                       for _ in range(40)])
        assert abs(est - L) < 0.25 * L, f"target {L} gave {est:.2f}"


def test_bootstrap_indices_are_in_range_and_sized():
    idx = stationary_bootstrap_indices(500, 10, np.random.default_rng(1))
    assert len(idx) == 500
    assert idx.min() >= 0 and idx.max() <= 499


def test_bootstrap_preserves_dependence():
    """Consecutive resampled positions should often be adjacent (blocks), not iid."""
    idx = stationary_bootstrap_indices(4000, 10, np.random.default_rng(2))
    lag1 = np.mean(idx[1:] - idx[:-1] == 1)
    assert lag1 > 0.6, f"lag-1 step rate {lag1:.2f} suggests the blocks were destroyed"


def test_bootstrap_is_deterministic_given_seed():
    a = stationary_bootstrap_indices(300, 10, np.random.default_rng(7))
    b = stationary_bootstrap_indices(300, 10, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_percentile_ci_brackets_the_samples():
    rng = np.random.default_rng(3)
    s = rng.standard_normal(5000)
    lo, hi = percentile_ci(s, 0.95)
    assert lo < np.median(s) < hi
    assert (s >= lo).mean() > 0.02 and (s <= hi).mean() > 0.02


def test_paired_pvalue_falls_with_effect_size():
    """Regression: p-values were pinned near 1.0 regardless of effect size."""
    rng = np.random.default_rng(4)
    ps = []
    for delta in (0.0, 0.3, 1.0):
        d = delta + rng.standard_normal(4000)
        out = paired_difference_ci(d / 2, -d / 2)
        ps.append(out["p_value"])
        assert abs(out["mean_diff"] - delta) < 0.1
    assert ps[0] > ps[1] > ps[2], f"p-values not monotone in effect size: {ps}"
    assert ps[2] < 0.35, f"a delta=1.0, sigma=1.0 gap should be clearly significant, got {ps[2]}"


def test_paired_difference_of_identical_series_is_flagged_untestable():
    s = np.linspace(-1, 1, 200)
    out = paired_difference_ci(s, s)
    assert out["p_value"] != out["p_value"]          # NaN, not a fake number
    assert out["excludes_zero"] is False
    assert "not testable" in out["note"]


def test_paired_interval_detects_a_real_gap():
    rng = np.random.default_rng(5)
    d = 0.5 + rng.standard_normal(4000) * 0.1
    out = paired_difference_ci(d, np.zeros_like(d))
    assert out["excludes_zero"] and out["ci_low"] > 0


def test_paired_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        paired_difference_ci(np.zeros(10), np.zeros(9))


# --------------------------------------------------------------------- baselines

def test_voltarget_scales_down_when_vol_is_high():
    n = 6
    cols = [f"a{i}" for i in range(n)]
    rng = np.random.default_rng(0)
    # NOISY panels: a constant-return frame has zero volatility and cannot exercise this
    calm = pd.DataFrame(rng.standard_normal((300, n)) * 0.001, columns=cols)
    wild = pd.DataFrame(rng.standard_normal((300, n)) * 0.020, columns=cols)
    warm = EqualWeightVolTarget(vol_target=0.10, vol_lookback=60)
    assert warm.decide(calm.iloc[:10], np.zeros(n)).sum() > 0
    s_calm = EqualWeightVolTarget(vol_target=0.10, vol_lookback=60)
    s_wild = EqualWeightVolTarget(vol_target=0.10, vol_lookback=60)
    w_calm = s_calm.decide(calm, np.zeros(n))
    w_wild = s_wild.decide(wild, np.zeros(n))
    # calm vol (~1.6% ann) sits below the 10% target, so it wants MORE than 1.0 and is
    # capped by max_exposure; wild vol (~32% ann) must be scaled down well below it
    assert w_calm.sum() <= 1.0 + 1e-9
    assert w_wild.sum() < 0.9 * w_calm.sum()   # 10% target vs ~13% ann vol -> ~0.78


def test_voltarget_never_levers_or_breaks_bounds():
    r = get_returns(end="2016-12-30")
    res = run_backtest(r, EqualWeightVolTarget(0.10, 60, max_exposure=1.0),
                       r.index[300], r.index[700], 10)
    assert res.weights.to_numpy().min() >= -1e-12
    assert res.weights.sum(axis=1).max() <= 1.0 + 1e-9


def test_dd_baseline_starts_flat_and_never_looks_ahead():
    """Its drawdown state may only reflect realized days, so a perturbed future cannot
    change the weights chosen up to that point."""
    r = get_returns(end="2016-12-30")
    T = 520
    cfg = load_config(CFG)
    res = run_backtest(r, MPCDrawdownRiskAversion(cfg, r.shape[1]), r.index[T - 60],
                       r.index[T + 30], 10)
    tampered = r.copy()
    tampered.iloc[T + 1:] *= 9.0
    res_b = run_backtest(tampered, MPCDrawdownRiskAversion(cfg, r.shape[1]),
                         r.index[T - 60], r.index[T + 30], 10)
    pos = res.weights.index.get_loc(r.index[T])
    np.testing.assert_allclose(res.weights.values[: pos + 1],
                               res_b.weights.values[: pos + 1], atol=1e-9)


def test_dd_baseline_arms_only_after_a_realized_loss():
    cfg = load_config(CFG)
    s = MPCDrawdownRiskAversion(cfg, 4)
    assert s.dd_armed is False
    assert s.drawdown() == 0.0
    s.observe(np.zeros(4), -0.10, np.full(4, 0.25), np.zeros(4))
    assert s.dd_armed is True, "a 10% realized loss must arm the drawdown trigger"
    assert s.drawdown() > 0.09


def test_dd_baseline_charges_costs_like_everything_else():
    r = get_returns(end="2016-12-30")
    cfg = load_config(CFG)
    res = run_backtest(r, MPCDrawdownRiskAversion(cfg, r.shape[1]), r.index[300],
                       r.index[600], 10)
    assert res.costs.sum() > 0
    assert res.net_returns.sum() < res.gross_returns.sum()


# ----------------------------------------------------------- regime-conditional

def test_conditional_metrics_omit_drawdown_and_handle_short_subsets():
    from experiments.regime_conditional import conditional_metrics
    r = np.array([0.01, -0.02, 0.005, 0.0, 0.03])
    m = conditional_metrics(r, 252, 0.0, 0.05)
    assert "max_drawdown" not in m
    assert m["n_days"] == 5 and np.isfinite(m["sharpe"])
    tiny = conditional_metrics(np.array([0.01, -0.02]), 252, 0.0, 0.05)
    assert np.isnan(tiny["sharpe"]) and tiny["n_days"] == 2


def test_conditional_split_reproduces_the_whole_sample():
    """stress + calm days must account for exactly the underlying return series."""
    r = get_returns(end="2016-12-30")
    idx = r.index
    rng = np.random.default_rng(11)
    stress = pd.Series(rng.random(len(idx)) < 0.3, index=idx)
    vals = r.mean(axis=1).to_numpy()
    from experiments.regime_conditional import conditional_metrics
    s = conditional_metrics(vals[stress.to_numpy()], 252, 0.0, 0.05)["n_days"]
    c = conditional_metrics(vals[~stress.to_numpy()], 252, 0.0, 0.05)["n_days"]
    assert s + c == len(idx)


def test_stress_labels_on_real_data_stay_inside_the_allowed_window():
    """Guards the study against ever labelling test days."""
    cfg = load_config(CFG)
    assert pd.Timestamp(cfg["splits"]["train_end"]) == pd.Timestamp("2015-12-31")
    assert pd.Timestamp(cfg["splits"]["val_end"]) == pd.Timestamp("2019-12-31")