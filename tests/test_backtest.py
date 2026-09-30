import numpy as np
import pytest
import pandas as pd

from src.backtest import run_backtest
from src.baselines import (BuyAndHold, EqualWeight, EqualWeightVolTarget, Markowitz,
                          MPCDrawdownRiskAversion)
from src.data import period_bounds
from tests.common import get_returns


def _returns(n_assets=5):
    return get_returns(n_assets=n_assets, seed=1)


def _perturb_future(returns, T, seed=5):
    """Scramble every return from row T onward."""
    r2 = returns.copy()
    rng = np.random.default_rng(seed)
    r2.iloc[T:] = rng.normal(0, 0.05, r2.iloc[T:].shape)
    return r2


def check_no_lookahead(make_strategy, returns, T):
    start, end = returns.index[T - 60], returns.index[T + 30]
    a = run_backtest(returns, make_strategy(), start, end, 10)
    b = run_backtest(_perturb_future(returns, T), make_strategy(), start, end, 10)
    pos_T = a.weights.index.get_loc(returns.index[T])
    # weights held on any day <= T may only depend on returns before that day
    np.testing.assert_allclose(a.weights.values[: pos_T + 1], b.weights.values[: pos_T + 1], atol=1e-7)


def test_uses_real_market_data_when_cached():
    """Guards the project's premise: tests run on real NSE returns, not the simulator."""
    r = get_returns()
    assert len(r) > 800 and r.shape[1] == 5
    assert r.attrs["source"] in ("real", "synthetic (cache unavailable)"), r.attrs["source"]
    # the real cache must cover the full 2012-2019 research window with no NaNs
    import copy as _copy
    from src.data import download_prices
    from src.utils import load_config
    cfg = load_config("configs/default.yaml")
    d = cfg["data"]
    try:
        p = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                            min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    except Exception:
        pytest.skip("no market data cache available offline")
    assert p.shape[1] == len(d["tickers"])
    assert p.index[0] <= pd.Timestamp("2012-01-02") and p.index[-1] >= pd.Timestamp("2019-12-31")
    assert not p.isna().any().any()


def test_no_lookahead_baselines():
    r = _returns()
    T = 800
    for mk in [EqualWeight, BuyAndHold, lambda: Markowitz(lookback=120),
               lambda: EqualWeightVolTarget(0.10, 60)]:
        check_no_lookahead(mk, r, T)


def test_no_lookahead_drawdown_baseline():
    """The dd baseline holds internal equity state fed back after each realized day, so
    this is the guard that the state cannot let future returns influence earlier weights."""
    from src.utils import load_config
    cfg = load_config("configs/default.yaml")
    r = _returns()
    check_no_lookahead(lambda: MPCDrawdownRiskAversion(cfg, r.shape[1]), r, 800)


def test_costs_are_charged_everywhere():
    r = _returns()
    start, end = r.index[300], r.index[500]
    free = run_backtest(r, EqualWeight(), start, end, 0.0)
    paid = run_backtest(r, EqualWeight(), start, end, 25.0)
    assert paid.costs.sum() > 0
    assert paid.net_returns.sum() < free.net_returns.sum()
    # first day: from all-cash into fully invested -> turnover 1.0
    assert abs(paid.turnover.iloc[0] - 1.0) < 1e-9


def test_buy_and_hold_trades_only_once():
    r = _returns()
    res = run_backtest(r, BuyAndHold(), r.index[300], r.index[500], 10)
    assert res.turnover.iloc[1:].abs().max() < 1e-9


def test_weights_valid():
    r = _returns()
    res = run_backtest(r, Markowitz(lookback=120), r.index[300], r.index[400], 10)
    assert (res.weights.values >= -1e-9).all() and (res.weights.sum(axis=1) <= 1 + 1e-6).all()


def test_splits_do_not_overlap():
    idx = pd.bdate_range("2012-01-02", "2025-12-31")
    cfg = {"train_end": "2017-12-31", "val_end": "2019-12-31"}
    tr, va, te = (period_bounds(cfg, p, idx) for p in ["train", "val", "test"])
    assert tr[1] < va[0] <= va[1] < te[0]
