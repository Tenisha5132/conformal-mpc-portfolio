import numpy as np
import pytest

from src.backtest import run_backtest
from src.conformal import AdaptiveConformal
from src.forecaster import RidgeForecaster
from src.mpc import MPCController, _to_HN, tighten_limits
from src.regime import VolRegimeDetector
from src.strategies import MPCStrategy
from src.utils import load_config
from tests.common import get_returns
from tests.test_backtest import check_no_lookahead


def _r():
    return get_returns(n_assets=5, seed=2)


def test_mpc_respects_constraints():
    N = 4
    cov = np.diag([1e-4] * N)
    ctrl = MPCController(horizon=3, risk_aversion=5, cost_bps=10, max_weight=0.4)
    w = ctrl.solve(np.zeros(N), np.array([1e-3, 5e-4, 0, -1e-3]), cov, max_exposure=0.7)
    tol = 1e-4  # solver tolerance
    assert (w >= -tol).all() and (w <= 0.4 + tol).all() and w.sum() <= 0.7 + tol


def test_robust_penalty_reduces_uncertain_asset():
    N = 2
    cov = np.diag([1e-4] * N)
    ctrl = MPCController(horizon=2, risk_aversion=2, cost_bps=1, max_weight=1.0)
    mu = np.array([1e-3, 1e-3])
    w_plain = ctrl.solve(np.zeros(N), mu, cov)
    w_rob = ctrl.solve(np.zeros(N), mu, cov, halfwidth=np.array([5e-3, 0.0]), kappa=1.0)
    assert w_rob[0] < w_plain[0] and w_rob[1] >= w_plain[1] - 1e-6


def test_forecast_persistence_decays_across_horizon():
    """A 1-day forecast must NOT be held flat for all H steps (that made the MPC H times bolder)."""
    mu = np.array([1e-3, 2e-3])
    flat = _to_HN(mu, 4, 2, persistence=1.0)
    decayed = _to_HN(mu, 4, 2, persistence=0.5)
    np.testing.assert_allclose(decayed[0], mu)
    np.testing.assert_allclose(decayed[1:], mu[None, :] * np.array([0.5, 0.25, 0.125])[:, None])
    assert (decayed[1:] < flat[1:]).all()
    # a 2-D input is passed through unchanged
    np.testing.assert_allclose(_to_HN(np.vstack([mu, -mu]), 4, 2, 0.5)[0], mu)


def test_tighten_limits_monotone():
    e1, _ = tighten_limits(1.0, 1.0, None, 1.0, 0.2)
    e2, _ = tighten_limits(2.0, 1.0, None, 1.0, 0.2)
    e3, _ = tighten_limits(10.0, 1.0, None, 1.0, 0.2)
    assert e1 == 1.0 and e2 < e1 and e3 <= e2 and e3 >= 0.2


def test_adaptive_conformal_coverage():
    rng = np.random.default_rng(0)
    ac = AdaptiveConformal(n_assets=3, alpha=0.1, gamma=0.01, window=250, min_cal=30)
    for t in range(3000):
        sigma = 1.0 if t < 1500 else 3.0                     # distribution shift halfway
        ac.update(sigma * rng.standard_normal(3), np.zeros(3))
    assert 0.86 <= ac.coverage() <= 0.94


def test_regime_no_lookahead_and_detects_stress():
    r = _r()
    det = VolRegimeDetector(21, 0.8, 100)
    T = 600
    a = det.update(r.iloc[:T])
    r2 = r.copy(); r2.iloc[T:] = 0.5
    b = det.update(r2.iloc[:T])
    assert a == b
    calm = r.copy() * 0.2
    stressed = calm.copy(); stressed.iloc[-10:] *= 30
    assert det.update(stressed) == 1


def test_ridge_forecaster_uses_only_history():
    r = _r()
    f1, f2 = RidgeForecaster(10, 500, 63, 10.0), RidgeForecaster(10, 500, 63, 10.0)
    T = 700
    p1 = f1.predict(r.iloc[:T])
    r2 = r.copy(); r2.iloc[T:] = 1.0
    p2 = f2.predict(r2.iloc[:T])
    np.testing.assert_allclose(p1, p2)


@pytest.mark.parametrize("flags", [
    {},
    dict(use_forecaster=True),
    dict(use_forecaster=True, use_conformal=True, use_tighten=True, use_regime=True),
])
def test_mpc_strategies_no_lookahead(flags):
    cfg = load_config("configs/default.yaml")
    cfg["backtest"]["lookback"] = 120
    cfg["mpc"]["horizon"] = 2
    cfg["forecaster"].update(kind="ridge", train_window=400, retrain_every=40)
    r = _r()
    n = r.shape[1]
    check_no_lookahead(lambda: MPCStrategy(cfg, n, **flags), r, 800)


def test_gru_forecaster_smoke():
    try:
        import torch  # noqa: F401
    except Exception:
        pytest.skip("torch not available")
    from src.forecaster import GRUForecaster
    r = _r()
    f = GRUForecaster(r.shape[1], lookback=10, hidden=8, epochs=2, train_window=300)
    p = f.predict(r.iloc[:400])
    assert p.shape == (r.shape[1],) and np.all(np.isfinite(p))
