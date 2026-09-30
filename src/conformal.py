"""Adaptive conformal prediction (Gibbs & Candes 2021) for per-asset return forecasts.

Online protocol, no look-ahead:
  1. forecast made at day t-1 for day t is stored
  2. when day t's return is observed, update(y_t, pred_t) is called
  3. halfwidth() gives the interval half-width used for day t+1
Per-asset miscoverage level alpha_t is adapted:  alpha_{t+1} = alpha_t + gamma*(alpha - err_t).
"""
from collections import deque

import numpy as np


class AdaptiveConformal:
    def __init__(self, n_assets, alpha=0.10, gamma=0.005, window=250, min_cal=30):
        self.n = n_assets
        self.alpha = alpha
        self.gamma = gamma
        self.min_cal = min_cal
        self.alpha_t = np.full(n_assets, alpha, dtype=float)
        self.scores = [deque(maxlen=window) for _ in range(n_assets)]
        self.hits = []                                # 1.0 if covered, per calibrated day

    def _q(self, i):
        sc = np.asarray(self.scores[i])
        m = len(sc)
        if m < self.min_cal:
            return np.nan
        level = 1.0 - self.alpha_t[i]
        if level >= 1.0:
            return float(sc.max())
        if level <= 0.0:
            return 0.0
        k = min(int(np.ceil((m + 1) * level)), m)
        return float(np.sort(sc)[k - 1])

    def halfwidth(self, fallback=None) -> np.ndarray:
        hw = np.array([self._q(i) for i in range(self.n)])
        if fallback is not None:
            fb = np.broadcast_to(np.asarray(fallback, dtype=float), hw.shape)
            hw = np.where(np.isnan(hw), fb, hw)
        return hw

    def update(self, y, pred):
        y, pred = np.asarray(y, float), np.asarray(pred, float)
        issued = self.halfwidth(fallback=None)        # interval that WAS available before seeing y
        s = np.abs(y - pred)
        cov = np.full(self.n, np.nan)
        for i in range(self.n):
            if not np.isnan(issued[i]):
                err = float(s[i] > issued[i])
                cov[i] = 1.0 - err
                self.alpha_t[i] += self.gamma * (self.alpha - err)
            self.scores[i].append(s[i])
        if not np.all(np.isnan(cov)):
            self.hits.append(cov)

    def coverage(self) -> float:
        return float(np.nanmean(np.vstack(self.hits))) if self.hits else float("nan")
