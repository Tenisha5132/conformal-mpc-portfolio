"""Self-calibrating constraint tightening (mpc_selfcal).

Two mechanisms, both driven only by realized information:

1. SELF-CALIBRATING BETA. The tightening coefficient beta in `tighten_limits` is no longer
   a constant. After each realized day the controller asks "did I lose more than I was
   willing to lose?", and nudges beta_toward a target breach frequency:

       breach_t   = 1 if net_return_t < -loss_threshold else 0
       beta_{t+1} = max(0, beta_t + eta * (breach_t - delta))

   Frequent breaches push beta up, which shrinks the risk limits via the SAME
   `tighten_limits` function the fixed-beta variants use. Quiet stretches relax it back
   toward zero. `delta` is the target daily breach frequency.

   NO LOOK-AHEAD: the update at day t consumes the net return REALIZED ON DAY t, and it is
   applied only from day t+1 onward. The backtest's `observe()` hook (called after a day is
   fully realized) drives it, exactly as for the drawdown baseline.

2. ENSEMBLE DISAGREEMENT. A single forecaster's interval width can be confidently wrong.
   We forecast with several cheap models (ridge, historical mean, momentum, and GRU if torch
   is installed) and measure how much they disagree:

       disagreement_t = mean over assets of std across models of the predicted return

   Disagreement rises when the market stops obeying whichever structure a given model
   assumes, so it is a useful second opinion on uncertainty that does not depend on the
   conformal residuals at all.

   The final tightening signal is the GEOMETRIC mean of the two uncertainty ratios, each
   normalized by its own trailing median so the two scales are comparable:

       ratio_t = sqrt( (width_t / median(width)) * (disagreement_t / median(disagreement)) )

   The geometric mean is used deliberately: it moves only when BOTH components move, so a
   single noisy input cannot by itself trigger a tightening episode.
"""
from collections import deque

import numpy as np
import pandas as pd

from src.forecaster import GRUForecaster, RidgeForecaster


def _available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------- beta state

class BetaState:
    """beta_t >= 0, adapted online toward a target breach frequency.

    beta_t is what `tighten_limits` consumes on day t+1. It is only ever moved by
    realized breaches, never by a future observation.
    """

    def __init__(self, beta0=0.0, eta=0.05, delta=0.01, loss_threshold=0.02):
        self.beta = float(max(0.0, beta0))
        self.eta = float(eta)
        self.delta = float(delta)
        self.loss_threshold = float(loss_threshold)
        self.breaches = []          # realized breach indicators, one per settled day
        self.history = []           # beta AFTER each update, for logging/plots

    def observe(self, net_return: float) -> int:
        """Settle the PREVIOUS day with its realized net return, then step beta forward.

        Returns the breach indicator for the day just settled. Called only after that day
        is fully realized.
        """
        breach = int(float(net_return) < -self.loss_threshold)
        self.breaches.append(breach)
        self.beta = max(0.0, self.beta + self.eta * (breach - self.delta))
        self.history.append(self.beta)
        return breach

    def breach_frequency(self, window: int | None = None) -> float:
        """Realized breach frequency, optionally over the last `window` days."""
        b = self.breaches if window is None else self.breaches[-int(window):]
        return float(np.mean(b)) if len(b) else float("nan")


# --------------------------------------------------------------------- ensemble

class _HistoricalMean:
    """Constant per-asset forecast: the trailing mean return. Trivial but honest baseline."""

    def __init__(self, lookback=252):
        self.lookback = int(lookback)

    def predict(self, hist):
        V = hist.values
        w = V[-self.lookback:] if len(V) >= self.lookback else V
        return w.mean(axis=0)

    @property
    def resid_std(self):
        return None


class _Momentum:
    """Lag-k return carried forward. Assumes short-horizon persistence, which is wrong
    exactly in turbulent markets - that is the point of including it in the ensemble."""

    def __init__(self, lookback=252, lag=5):
        self.lookback = int(lookback)
        self.lag = int(lag)

    def predict(self, hist):
        V = hist.values
        w = V[-self.lookback:] if len(V) >= self.lookback else V
        if len(w) < self.lag:
            return np.zeros(w.shape[1])
        return w[-self.lag:].mean(axis=0)

    @property
    def resid_std(self):
        return None


class DisagreementEnsemble:
    """Several cheap forecasters, scored by their cross-model dispersion.

    `members` is a list of (name, model) with a `predict(hist) -> np.ndarray` interface.
    The GRU is included only when torch is importable, so the ensemble degrades gracefully
    on a torch-free machine instead of crashing.
    """

    def __init__(self, cfg, n_assets, seed=0):
        f = cfg["forecaster"]
        self.lookback = int(cfg["backtest"]["lookback"])
        self.members = [
            ("ridge", RidgeForecaster(f["lookback"], f["train_window"],
                                      f["retrain_every"], f["ridge_alpha"])),
            ("mean", _HistoricalMean(self.lookback)),
            ("momentum", _Momentum(self.lookback, cfg["ensemble"]["momentum_lag"])),
        ]
        self.has_gru = _available()
        if self.has_gru and cfg["ensemble"].get("use_gru", True):
            self.members.append(
                ("gru", GRUForecaster(n_assets, f["lookback"], f["gru_hidden"],
                                       f["gru_epochs"], f["gru_lr"], f["train_window"],
                                       f["retrain_every"], seed, f["gru_batch"],
                                       f["gru_weight_decay"])))
        self._last = None
        self.disagreement = deque(maxlen=int(cfg["ensemble"]["history"]))
        self.disagreement_log = []

    def names(self):
        return [n for n, _ in self.members]

    def predict(self, hist):
        """Ensemble-mean forecast plus the cross-model dispersion."""
        preds = []
        for _, m in self.members:
            p = np.asarray(m.predict(hist), dtype=float)
            if p.shape == (hist.shape[1],) and np.all(np.isfinite(p)):
                preds.append(p)
        if not preds:
            raise RuntimeError("no ensemble member produced a usable forecast")
        P = np.vstack(preds)
        mean = P.mean(axis=0)
        # per-asset std ACROSS models, then averaged over assets
        dis = float(np.mean(P.std(axis=0, ddof=0))) if len(preds) > 1 else 0.0
        self._last = {"mean": mean, "disagreement": dis, "n_members": len(preds)}
        return mean, dis

    def observe_disagreement(self, dis: float):
        """Append today's disagreement to the trailing history used for normalization."""
        self.disagreement.append(float(dis))
        self.disagreement_log.append({"disagreement": float(dis),
                                      "median": self.trailing_median()})

    def trailing_median(self, min_history: int = 1) -> float:
        if len(self.disagreement) < min_history:
            return float("nan")
        return float(np.median(self.disagreement))


# ------------------------------------------------------------- uncertainty ratio

def uncertainty_ratio(width: float, dis: float, width_history: deque,
                      dis_history: deque, min_history: int = 20) -> float:
    """Geometric mean of the two normalized uncertainty ratios.

    ratio = sqrt( (width / med(width)) * (dis / med(dis)) )

    Both medians are trailing (past values only). Returns 1.0 - meaning "no tightening" -
    until enough history exists, so the controller starts from the un-tightened limits
    rather than from an arbitrary guess.
    """
    if len(width_history) < min_history or len(dis_history) < min_history:
        return 1.0
    mw = float(np.median(width_history))
    md = float(np.median(dis_history))
    if not np.isfinite(mw) or not np.isfinite(md) or mw <= 0 or md <= 0:
        return 1.0
    r_w = float(width) / mw
    r_d = float(dis) / md
    # geometric mean; guard both arms against non-positive or non-finite values
    if not (np.isfinite(r_w) and np.isfinite(r_d)) or r_w <= 0 or r_d <= 0:
        return 1.0
    return float(np.sqrt(r_w * r_d))


def geometric_mean_ratio(r1: float, r2: float) -> float:
    """Standalone geometric mean of two ratios, guarded for non-positive inputs."""
    if not (np.isfinite(r1) and np.isfinite(r2)) or r1 <= 0 or r2 <= 0:
        return 1.0
    return float(np.sqrt(r1 * r2))