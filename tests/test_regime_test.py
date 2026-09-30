"""Tests for the regime testbed (experiments/regime_test.py) and the vol-jump search.

The load-bearing test in this file is `test_jump_search_never_reaches_past_the_bound`.
The largest volatility jump in the full sample is Dec-2019 -> Mar-2020 (the COVID crash),
which lies in the SEALED TEST period. An unbounded search would silently select exactly the
event most likely to flatter the method, from data the project agreed never to touch. That
guard must not regress.
"""
import numpy as np
import pandas as pd
import pytest

from experiments.regime_test import compute_metrics_from_returns, find_real_jump_window
from src.data import make_regime_prices, to_returns
from src.utils import load_config


def _cfg():
    return load_config("configs/default.yaml")


# ------------------------------------------------------------- the generator

def test_regime_generator_is_deterministic():
    cfg = _cfg()
    a = make_regime_prices(cfg, seed=cfg["seed"])
    b = make_regime_prices(cfg, seed=cfg["seed"])
    pd.testing.assert_frame_equal(a, b)
    assert not make_regime_prices(cfg, seed=cfg["seed"] + 1).equals(a)


def test_regime_generator_actually_jumps_and_reverts():
    """The fixture must contain a real vol jump followed by a decay, or it tests nothing."""
    cfg = _cfg()
    r = to_returns(make_regime_prices(cfg, seed=cfg["seed"]))
    v = (r.mean(axis=1).rolling(20).std() * np.sqrt(252)).dropna()
    rc = cfg["regime_test"]
    assert v.max() / v.min() > rc["vol_spike"] / rc["vol_calm"] * 0.5, "no vol jump"
    # after the first peak, vol must fall materially (the mean reversion under test)
    pk = int(np.argmax(v.values[:len(v) // 2]))
    assert v.iloc[pk + 60] < 0.6 * v.iloc[pk], "vol did not revert after the jump"


def test_regime_generator_respects_its_config():
    cfg = _cfg()
    r = to_returns(make_regime_prices(cfg, seed=cfg["seed"]))
    assert r.shape[1] == cfg["regime_test"]["n_assets"]
    assert not r.isna().any().any()
    assert (r.values > -1).all(), "a return below -100% means the fixture is broken"


# ------------------------------------------------------------- the search

def test_jump_search_never_reaches_past_the_bound():
    """THE GUARD. A bound earlier than the true jump must not return that jump."""
    idx = pd.bdate_range("2012-01-02", periods=800)
    rng = np.random.default_rng(0)
    r = pd.DataFrame(rng.normal(0, 0.01, (800, 3)), index=idx,
                     columns=["a", "b", "c"])
    # force one unmistakable jump at day 600
    r.iloc[600:610] = -0.09
    p = r.mean(axis=1)
    v = p.rolling(20).std() * np.sqrt(252)

    full = find_real_jump_window(r, cfg=None, min_ratio=2.0)
    assert full is not None and full[2] >= 600, "should find the injected jump unbounded"

    bounded = find_real_jump_window(r, cfg=None, min_ratio=2.0, search_end=500)
    assert bounded is None or bounded[2] < 500, (
        f"search returned peak day {bounded[2]} despite bound 500")


def test_jump_search_ignores_a_monotone_vol_spike_without_reversion():
    """A jump that never reverts is not the regime under test and must not be selected."""
    idx = pd.bdate_range("2012-01-02", periods=600)
    rng = np.random.default_rng(1)
    scale = np.concatenate([np.full(400, 0.005), np.full(200, 0.05)])  # steps up, stays up
    r = pd.DataFrame(rng.standard_normal((600, 3)) * scale[:, None], index=idx,
                     columns=["a", "b", "c"])
    assert find_real_jump_window(r, cfg=None, min_ratio=2.0) is None


def test_jump_search_on_real_data_respects_val_end():
    """End-to-end: on the real frame, no selected peak may fall in the sealed test period."""
    from src.data import download_prices
    cfg = _cfg()
    d = cfg["data"]
    try:
        p = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                            min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    except Exception:
        pytest.skip("no market data cache available offline")
    r = to_returns(p)
    val_end = pd.Timestamp(cfg["splits"]["val_end"])
    sr = r.loc[:val_end]
    found = find_real_jump_window(sr, cfg, search_end=len(sr) - 140)
    if found is not None:
        _, _, peak_i = found
        assert sr.index[peak_i] <= val_end, "selected a jump from the sealed test period"


# ------------------------------------------------------------- metrics helper

def test_subwindow_metrics_cvar_takes_the_worst_tail():
    rng = np.random.default_rng(2)
    net = pd.Series(rng.normal(0.001, 0.02, 500))
    m = compute_metrics_from_returns(net)
    r = net.values
    q = np.quantile(r, 0.05)
    assert m["cvar_daily"] == pytest.approx(-r[r <= q].mean())
    assert m["cvar_daily"] > 0, "cvar_daily is reported as a loss magnitude (positive)"


# ------------------------------------------------------------- exposure-matched diagnostic

def test_scale_to_exposure_actually_hits_the_target():
    """The whole diagnostic rests on this: scaling by k must move exposure to k*from."""
    from experiments.exposure_matched import scale_to_exposure
    idx = pd.bdate_range("2020-01-01", periods=300)
    net = pd.Series(np.random.default_rng(0).normal(0.0005, 0.01, 300), index=idx)
    out = scale_to_exposure(net, 0.5, 1.0)
    assert out.std(ddof=1) == pytest.approx(net.std(ddof=1) * 2.0, rel=1e-9)
    assert out.mean() == pytest.approx(net.mean() * 2.0, rel=1e-9)
    # scaling to the SAME exposure is the identity
    same = scale_to_exposure(net, 0.7, 0.7)
    pd.testing.assert_series_equal(same, net)


def test_scale_to_exposure_respects_a_risk_free_rate():
    """With rf > 0 the scaling must be rf + k*(net - rf), not simply k*net."""
    from experiments.exposure_matched import scale_to_exposure
    idx = pd.bdate_range("2020-01-01", periods=50)
    net = pd.Series(np.full(50, 0.01), index=idx)
    rf = 0.0001
    out = scale_to_exposure(net, 1.0, 2.0, rf_daily=rf)
    assert out.iloc[0] == pytest.approx(rf + 2.0 * (0.01 - rf))
    # a constant-rf series must return exactly rf at zero exposure change
    flat = pd.Series(np.full(50, rf), index=idx)
    assert scale_to_exposure(flat, 1.0, 1.0, rf_daily=rf).iloc[0] == pytest.approx(rf)


def test_mean_exposure_is_computed_from_weights_and_is_causal():
    """Exposure must come from the realized weights, and must not read the future."""
    from experiments.exposure_matched import mean_exposure, synth_result
    idx = pd.bdate_range("2020-01-01", periods=10)
    w = pd.DataFrame({"a": [0.5] * 5 + [0.1] * 5, "b": [0.5] * 5 + [0.1] * 5}, index=idx)
    net = pd.Series(0.0, index=idx)
    res = synth_result(net)
    res.weights = w
    assert mean_exposure(res) == pytest.approx(0.6)
    # truncating the future must not change the mean over the retained prefix
    res_short = synth_result(net.iloc[:5])
    res_short.weights = w.iloc[:5]
    assert mean_exposure(res_short) == pytest.approx(1.0)


def test_exposure_matched_uses_the_minimum_exposure_not_the_maximum():
    """Matching up to the highest exposure would reward the most aggressive strategy.

    The script must match to the LOWEST realized exposure, so nobody is advantaged by having
    taken more risk than the comparison.
    """
    import inspect
    from experiments import exposure_matched
    src = inspect.getsource(exposure_matched.main)
    assert "min(exps.values())" in src, "must match to the lowest realized exposure"
    assert "max(exps.values())" not in src


def test_exposure_matched_flags_the_cvar_sign_convention():
    """cvar_daily is a NEGATIVE loss magnitude; a sign slip would invert every verdict."""
    import inspect
    from experiments import exposure_matched
    src = inspect.getsource(exposure_matched)
    assert "NEGATIVE loss magnitude" in src, "must document the CVaR sign convention"
