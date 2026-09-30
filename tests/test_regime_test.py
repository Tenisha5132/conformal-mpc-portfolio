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
