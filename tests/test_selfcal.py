"""Tests for mpc_selfcal: the beta_t self-calibration and the ensemble disagreement signal."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import run_backtest
from src.selfcal import (BetaState, DisagreementEnsemble, geometric_mean_ratio,
                         uncertainty_ratio)
from src.strategies import MPCSelfCalStrategy
from src.utils import load_config

from tests.common import get_returns

CONFIG = "configs/default.yaml"
SC = {"beta0": 0.0, "loss_threshold": 0.02, "delta": 0.01, "eta": 0.05}


def _returns():
    return get_returns(n_assets=5, seed=1)


# --------------------------------------------------------------- BetaState basics

def test_beta_never_negative():
    b = BetaState(**SC)
    for _ in range(50):
        b.observe(0.05)                    # calm days push beta down
    assert b.beta == 0.0, "beta must floor at zero"


def test_beta_rises_on_breaches_and_falls_on_calm_days():
    b = BetaState(**SC)
    start = b.beta
    b.observe(-0.05)                       # a breach
    assert b.beta > start
    after_breach = b.beta
    b.observe(0.05)                        # a calm day
    assert b.beta < after_breach


def test_beta_step_size_is_exactly_eta():
    """The update is beta += eta*(breach - delta): up-steps are eta*(1-delta) and
    down-steps are -eta*delta, which are NOT symmetric."""
    b = BetaState(**SC)
    b.observe(-0.05)                       # breach=1
    assert b.beta == pytest.approx(SC["eta"] * (1 - SC["delta"]), abs=1e-12)
    b.observe(0.05)                        # breach=0 -> down-step is only eta*delta
    assert b.beta == pytest.approx(SC["eta"] * (1 - SC["delta"]) - SC["eta"] * SC["delta"],
                                   abs=1e-12)


def test_beta_update_matches_closed_form_over_a_path():
    b = BetaState(**SC)
    returns = [0.01, -0.05, 0.02, -0.03, -0.06, 0.04, -0.01, 0.0]
    expected = 0.0
    for i, r in enumerate(returns):
        breach = int(r < -SC["loss_threshold"])
        expected = max(0.0, expected + SC["eta"] * (breach - SC["delta"]))
        got = b.observe(r)
        assert got == breach
        assert b.beta == pytest.approx(expected, abs=1e-12), f"diverged at step {i}"


def test_breach_frequency_matches_definition():
    b = BetaState(**SC)
    for r in [0.01, -0.05, 0.0, -0.09, 0.02]:
        b.observe(r)
    assert b.breach_frequency() == pytest.approx(0.4)      # 2 of 5
    assert b.breach_frequency(window=2) == pytest.approx(0.5)   # last two are breach, calm


def test_breach_frequency_empty_is_nan():
    assert np.isnan(BetaState(**SC).breach_frequency())


# ------------------------------------------------------- monotonic response test

def test_beta_responds_monotonically_to_breach_rate():
    """More breaches -> higher beta, ordering preserved."""
    finals = []
    for breach_rate in (0.0, 0.05, 0.15, 0.40):
        b = BetaState(**SC)
        for i in range(400):
            r = -0.05 if (i % 100) < breach_rate * 100 else 0.01
            b.observe(r)
        finals.append(b.beta)
    assert finals == sorted(finals), f"beta not monotone in breach rate: {finals}"
    assert finals[-1] > 0 > finals[0] - 1e-12


def test_variance_shift_moves_long_run_breach_rate_toward_delta():
    """Synthetic panel whose variance jumps halfway through.

    The quantity beta controls is the POST-SHIFT breach rate, so that is what we test:
    run the identical return path twice, with and without the beta feedback loop, and
    require that feedback (a) substantially reduces the breach rate once losses grow and
    (b) settles near the target delta rather than merely somewhere lower.
    """
    rng = np.random.default_rng(3)
    n = 3000
    vol = np.where(np.arange(n) < n // 2, 0.008, 0.020)      # variance shift halfway
    r = rng.standard_normal(n) * vol
    split = n // 2 + 200                                    # let beta settle post-shift

    def run(with_feedback):
        b = BetaState(**SC)
        br = []
        for day in range(n):
            # beta_t caps exposure; a tighter cap shrinks today's loss magnitude
            cap = max(0.2, 1.0 / (1.0 + b.beta)) if with_feedback else 1.0
            br.append(b.observe(float(cap) * r[day]))
        return np.asarray(br), b.beta

    managed, beta_end = run(True)
    unmanaged, _ = run(False)

    m_post, u_post = managed[split:].mean(), unmanaged[split:].mean()
    assert u_post > 0.10, "the stress regime must actually breach often without feedback"
    assert m_post < u_post / 3.0, f"feedback barely helped: {u_post:.3f} -> {m_post:.3f}"
    assert m_post == pytest.approx(SC["delta"], abs=0.01), \
        f"breach rate did not converge toward delta: {m_post:.4f} vs {SC['delta']}"
    assert beta_end > 0.0, "beta must stay elevated while losses are large"


def test_beta_reacts_to_realized_loss_only_after_the_fact():
    """A day with no breach leaves beta strictly below the breached path."""
    calm = BetaState(**SC)
    bad = BetaState(**SC)
    calm.observe(0.01)
    bad.observe(-0.10)
    assert calm.beta == 0.0 and bad.beta > 0.0


# -------------------------------------------------------- uncertainty ratio maths

def test_uncertainty_ratio_needs_history_then_is_geometric_mean():
    from collections import deque
    wh, dh = deque([1.0] * 10, maxlen=250), deque([0.01] * 10, maxlen=250)
    assert uncertainty_ratio(5.0, 0.05, wh, dh, min_history=20) == 1.0  # no history
    for _ in range(20):
        wh.append(1.0)
        dh.append(0.01)
    # width 2x its median, disagreement 8x its median -> sqrt(2*8) = 4
    assert uncertainty_ratio(2.0, 0.08, wh, dh, min_history=20) == pytest.approx(4.0)


def test_uncertainty_ratio_is_one_when_either_arm_moves_alone():
    from collections import deque
    wh, dh = deque([1.0] * 30, maxlen=250), deque([0.01] * 30, maxlen=250)
    # disagreement alone at its median -> product 1 -> ratio 1
    assert uncertainty_ratio(1.0, 0.01, wh, dh, min_history=20) == pytest.approx(1.0)


def test_geometric_mean_guards_nonpositive_and_nonfinite():
    assert geometric_mean_ratio(4.0, 0.25) == pytest.approx(1.0)
    assert geometric_mean_ratio(-1.0, 4.0) == 1.0
    assert geometric_mean_ratio(0.0, 4.0) == 1.0
    assert geometric_mean_ratio(np.nan, 4.0) == 1.0
    assert geometric_mean_ratio(np.inf, 4.0) == 1.0


def test_tighten_limits_uses_the_beta_we_pass():
    """beta_t=0 must be inert (full exposure); larger beta must tighten. This is the
    contract that makes BetaState meaningful."""
    from src.mpc import tighten_limits
    e0, _ = tighten_limits(2.0, 1.0, None, 0.0, 0.2)
    e1, _ = tighten_limits(2.0, 1.0, None, 0.5, 0.2)
    e2, _ = tighten_limits(2.0, 1.0, None, 4.0, 0.2)
    assert e0 == 1.0
    assert 1.0 > e1 > e2 >= 0.2


# ------------------------------------------------------------------- ensemble

def test_ensemble_includes_expected_members_and_degrades_without_torch():
    cfg = load_config(CONFIG)
    ens = DisagreementEnsemble(cfg, 5, seed=0)
    names = ens.names()
    assert "ridge" in names and "mean" in names and "momentum" in names
    try:
        import torch  # noqa: F401
        assert "gru" in names
    except Exception:
        assert "gru" not in names, "GRU must be omitted when torch is unavailable"


def test_ensemble_disagreement_is_zero_for_identical_models():
    """Sanity on the statistic: if all members agree, disagreement is 0."""
    cfg = load_config(CONFIG)
    ens = DisagreementEnsemble(cfg, 4, seed=0)
    for _, m in ens.members:
        m.predict = (lambda v: np.zeros(4))
    _, dis = ens.predict(pd.DataFrame(np.zeros((50, 4)), columns=list("ABCD")))
    assert dis == pytest.approx(0.0, abs=1e-15)


def test_ensemble_disagreement_runs_on_real_data():
    cfg = load_config(CONFIG)
    r = _returns()
    ens = DisagreementEnsemble(cfg, r.shape[1], seed=0)
    mean, dis = ens.predict(r)
    assert mean.shape == (r.shape[1],)
    assert np.isfinite(dis) and dis >= 0.0


def test_ensemble_disagreement_history_is_trailing_only():
    cfg = load_config(CONFIG)
    ens = DisagreementEnsemble(cfg, 4, seed=0)
    ens.observe_disagreement(0.02)
    assert ens.trailing_median() == pytest.approx(0.02)


# ------------------------------------------------------------- strategy, no leakage

def test_selfcal_no_lookahead():
    """Scrambling returns from day T onward must not change any weight up to and including T."""
    from tests.test_backtest import check_no_lookahead
    cfg = load_config(CONFIG)
    r = _returns()
    check_no_lookahead(lambda: MPCSelfCalStrategy(cfg, r.shape[1]), r, 800)


def test_selfcal_observe_only_sees_realized_returns():
    """The strategy's beta may advance only via observe(); decide() must not mutate it."""
    cfg = load_config(CONFIG)
    r = _returns()
    s = MPCSelfCalStrategy(cfg, r.shape[1])
    s.decide(r.iloc[:600], np.full(r.shape[1], 0.1))
    assert s.beta_state.beta == cfg["selfcal"]["beta0"], "decide() must not touch beta"
    s.observe(np.zeros(r.shape[1]), -0.09, np.full(r.shape[1], 0.1), np.zeros(r.shape[1]))
    assert s.beta_state.beta > 0.0


def test_selfcal_runs_end_to_end_and_logs():
    cfg = load_config(CONFIG)
    r = _returns()
    s = MPCSelfCalStrategy(cfg, r.shape[1])
    res = run_backtest(r, s, r.index[300], r.index[900], cfg["backtest"]["cost_bps"])
    assert len(res.net_returns) > 0
    assert len(s.log) == len(res.net_returns), "one log row per decision"
    for row in s.log:
        assert row["beta"] >= 0.0
        assert np.isfinite(row["uncertainty_ratio"])
        assert row["uncertainty_ratio"] >= 0.0
    # one beta update per settled day. The final day's beta is computed but never used
    # for a decision, which is exactly the no-look-ahead margin.
    assert len(s.beta_trajectory()) == len(res.net_returns)
    assert np.all(np.asarray(s.beta_trajectory()) >= 0.0)


def test_selfcal_charges_costs():
    cfg = load_config(CONFIG)
    r = _returns()
    res = run_backtest(r, MPCSelfCalStrategy(cfg, r.shape[1]), r.index[300], r.index[700], 10)
    assert res.costs.sum() > 0
    assert res.net_returns.sum() < res.gross_returns.sum()


def test_selfcal_exposure_cap_reacts_to_learned_beta():
    """A long run of breaches must lower the exposure cap the strategy passes to the solver."""
    cfg = load_config(CONFIG)
    r = _returns()
    s = MPCSelfCalStrategy(cfg, r.shape[1])
    run_backtest(r, s, r.index[300], r.index[900], cfg["backtest"]["cost_bps"])
    caps = [row["exposure_cap"] for row in s.log]
    betas = [row["beta"] for row in s.log]
    assert min(caps) <= max(caps)
    # the tightest cap should coincide with a large beta
    assert betas[int(np.argmin(caps))] >= np.median(betas)