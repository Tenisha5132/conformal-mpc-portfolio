"""Stationary (Politis & Romano 1994) block bootstrap.

WHAT IT DOES
------------
Resamples the OBSERVED daily returns of the real validation window in overlapping blocks,
so each resample is a synthetic *path through real data* rather than a draw from a fitted
distribution. That distinction matters for this project: it keeps the whole study on real
market data (no simulated DGP) while still exposing sampling uncertainty.

Blocks are drawn with geometric lengths: an end-of-block is triggered with hazard
p = 1 / mean_block_length, giving E[block length] = mean_block_length exactly.

PAIRING
-------
One index sequence per replicate is generated ONCE and applied to EVERY strategy's return
series. Differences are then taken within-replicate, which removes the common path noise
and is the only correct way to get a paired interval. Unpaired intervals would be far too
wide and would understate the evidence for a difference.
"""
import numpy as np
import pandas as pd


def stationary_bootstrap_indices(n: int, mean_block: float, rng: np.random.Generator) -> np.ndarray:
    """Draw n indices by walking a circle and cutting into blocks of geometric length.

    With hazard p = 1/mean_block, P(block length = k) = p (1-p)^(k-1), so the mean block
    length is exactly 1/p = mean_block.
    """
    if n <= 0:
        return np.empty(0, dtype=int)
    p = 1.0 / float(mean_block)
    idx = np.empty(n, dtype=int)
    i = int(rng.integers(n))
    for t in range(n):
        idx[t] = i
        if rng.random() < p:
            i = int(rng.integers(n))
        else:
            i = (i + 1) % n
    return idx


def realized_block_length(sample_idx: np.ndarray) -> float:
    """Mean length of the runs in a sampled index sequence (empirical check on L)."""
    if len(sample_idx) == 0:
        return float("nan")
    breaks = np.flatnonzero(np.diff(sample_idx) != 1) + 1
    starts = np.concatenate([[0], breaks])
    ends = np.concatenate([breaks, [len(sample_idx)]])
    return float(np.mean(ends - starts))


def _max_drawdown(r: np.ndarray) -> float:
    eq = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())


def _sharpe(r: np.ndarray, trading_days: int, risk_free: float) -> float:
    excess = r - risk_free
    vol = excess.std(ddof=1)
    if not np.isfinite(vol) or vol <= 0:
        return float("nan")
    return float(excess.mean() * trading_days / (vol * np.sqrt(trading_days)))


def _cvar(r: np.ndarray, alpha: float) -> float:
    q = np.quantile(r, alpha)
    tail = r[r <= q]
    return float(tail.mean()) if len(tail) else float("nan")


def bootstrap_metrics(returns_by_strategy: dict, mean_block: float, n_resamples: int,
                      seed: int, trading_days: int = 252, risk_free: float = 0.0,
                      cvar_alpha: float = 0.05) -> tuple:
    """Resample the common date axis once per replicate and score every strategy on it.

    Returns (point_metrics, dist) where `dist[strategy][metric]` is an array of length
    n_resamples. All strategies share the identical resampled index sequence per replicate,
    so any two columns of `dist` are paired and can be differenced directly.
    """
    names = list(returns_by_strategy)
    lengths = {len(v) for v in returns_by_strategy.values()}
    if len(lengths) != 1:
        raise ValueError("all strategies must share the same date axis for a paired bootstrap")
    n = lengths.pop()
    # align to a common DatetimeIndex
    idx = returns_by_strategy[names[0]].index
    mats = {k: np.asarray(returns_by_strategy[k].reindex(idx).to_numpy(), dtype=float)
            for k in names}

    rng = np.random.default_rng(seed)
    point = {}
    for k, v in returns_by_strategy.items():
        a = mats[k]
        point[k] = {
            "sharpe": _sharpe(a, trading_days, risk_free),
            "cvar_daily": _cvar(a, cvar_alpha),
            "max_drawdown": _max_drawdown(a),
        }

    dist = {k: {"sharpe": np.empty(n_resamples),
                "cvar_daily": np.empty(n_resamples),
                "max_drawdown": np.empty(n_resamples)} for k in names}
    for b in range(n_resamples):
        sel = stationary_bootstrap_indices(n, mean_block, rng)
        for k in names:
            a = mats[k][sel]
            dist[k]["sharpe"][b] = _sharpe(a, trading_days, risk_free)
            dist[k]["cvar_daily"][b] = _cvar(a, cvar_alpha)
            dist[k]["max_drawdown"][b] = _max_drawdown(a)
    return point, dist


def percentile_ci(samples: np.ndarray, confidence: float = 0.95) -> tuple:
    """Two-sided percentile interval over bootstrap replicates."""
    a = np.asarray(samples, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return float("nan"), float("nan")
    lo = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(a, lo)), float(np.percentile(a, 100.0 - lo))


def paired_difference_ci(target: np.ndarray, reference: np.ndarray,
                         confidence: float = 0.95) -> dict:
    """Interval for target - reference using WITHIN-replicate differences.

    Both inputs must come from the same bootstrap run (same replicate ordering), so the
    common path noise cancels and the interval reflects the difference alone.

    The p-value is a two-sided percentile-bootstrap test. `d` is already the bootstrap
    distribution of the difference statistic, so H0 is tested by asking what fraction of
    those replicates falls on the far side of zero:

        p = 2 * min( frac(d <= 0), frac(d >= 0) )

    The replicate counts are floored at 1/n so a result is never reported as p = 0.
    """
    target = np.asarray(target, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if target.shape != reference.shape:
        raise ValueError("paired bootstrap inputs must have the same length")
    d = target - reference
    finite = np.isfinite(d)
    d = d[finite]
    lo, hi = percentile_ci(d, confidence)
    n = int(d.size)
    out = {"mean_diff": float("nan"), "ci_low": lo, "ci_high": hi,
           "p_value": float("nan"), "excludes_zero": False, "n": n}
    if n == 0:
        return out
    mu = float(d.mean())
    out["mean_diff"] = mu
    if float(np.max(d) - np.min(d)) <= 0.0:
        # difference is exactly constant -> no variability, no evidence either way
        out["note"] = "constant difference; not testable"
        return out
    n_f = max(1, n)
    frac_below = max(float(np.count_nonzero(d <= 0.0)), 1.0) / n_f
    frac_above = max(float(np.count_nonzero(d >= 0.0)), 1.0) / n_f
    out["p_value"] = float(min(1.0, 2.0 * min(frac_below, frac_above)))
    out["excludes_zero"] = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0))
    return out
