"""Tests for experiments/jump_events.py.

The critical property is the SEALED-PERIOD GUARD. experiments/jump_events.py
enumerates every real vol-jump event in train+validation, which is exactly the kind
of search that can silently reach into the test period and pick the COVID crash
(the largest jump in the sample, x6.94) to flatter the method. These tests pin that
guard down.

Run: python -m pytest tests/test_jump_events.py -q
"""
import numpy as np
import pandas as pd
import pytest

from experiments.jump_events import find_jump_events
from src.utils import load_config


def _synthetic_jumps(n=900, seed=0):
    """A return series with two clean vol-jump-and-revert episodes."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.008, n)
    # episode 1: calm, then a big spike, then decay
    r[300:320] *= 12.0
    # episode 2: same, far enough away to survive de-clustering
    r[700:720] *= 12.0
    return pd.DataFrame({"MKT": r})


def test_finds_the_planted_jumps():
    ev = find_jump_events(_synthetic_jumps(), min_ratio=2.0, search_end=760, cooldown=140)
    assert len(ev) >= 2, f"expected >=2 events, got {ev}"


def test_declusters_adjacent_detections():
    """A long crisis must not be counted as many independent events."""
    ev = find_jump_events(_synthetic_jumps(), min_ratio=2.0, search_end=760, cooldown=140)
    peaks = [p for _, _, p in ev]
    assert peaks == sorted(peaks), "events must be returned in chronological order"
    for a, b in zip(peaks, peaks[1:]):
        assert b - a >= 140, f"events too close together: {a} and {b}"


def test_search_end_hard_bounds_the_search():
    """THE KEY GUARD: an event after search_end must never be returned.

    The COVID crash sits after the validation boundary. Clipping the search must
    exclude it, so this fails loudly if the bound is ever removed or ignored.
    """
    r = _synthetic_jumps(n=1200, seed=1)
    # plant a huge jump far past the bound
    r.iloc[1000:1020, 0] *= 25.0
    bound = 900
    ev = find_jump_events(r, min_ratio=2.0, search_end=bound, cooldown=140)
    assert ev, "control: expected an event before the bound"
    for _, i, peak_i in ev:
        assert peak_i < bound, f"event at {peak_i} leaked past search_end={bound}"
        assert i < bound, f"candidate start {i} leaked past search_end={bound}"


def test_unbounded_search_reaches_further_than_bounded():
    """Sanity check that the bound is what stops the search, not the data.

    The same series searched without a bound finds strictly more (or equal)
    candidates than the bounded search, which is exactly the leak we are guarding.
    """
    r = _synthetic_jumps(n=1200, seed=1)
    r.iloc[1000:1020, 0] *= 25.0
    bounded = find_jump_events(r, min_ratio=2.0, search_end=900, cooldown=140)
    unbounded = find_jump_events(r, min_ratio=2.0, search_end=None, cooldown=140)
    assert len(unbounded) >= len(bounded)
    assert max((p for _, _, p in bounded), default=-1) < 900


def test_search_end_beyond_data_raises():
    r = _synthetic_jumps()
    with pytest.raises(ValueError):
        find_jump_events(r, min_ratio=2.0, search_end=len(r) + 500)


def test_config_val_end_is_after_train_end():
    """The bound the runner uses must be a real, non-degenerate validation end."""
    cfg = load_config("configs/default.yaml")
    tr = pd.Timestamp(cfg["splits"]["train_end"])
    va = pd.Timestamp(cfg["splits"]["val_end"])
    assert tr < va
