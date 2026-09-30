"""Market-mood detector (#1): calm (0) vs stress (1) from realized market volatility.

The stress threshold is an EXPANDING quantile of past volatility values only (no look-ahead).
"""
import numpy as np
import pandas as pd


class VolRegimeDetector:
    def __init__(self, vol_window=21, quantile=0.80, min_history=252, min_vol_history=30):
        self.w, self.q, self.min_history = vol_window, quantile, min_history
        self.min_vol_history = min_vol_history

    def update(self, hist: pd.DataFrame) -> int:
        if len(hist) < self.min_history:
            return 0
        mkt = hist.mean(axis=1)                       # equal-weight market proxy
        vol = mkt.rolling(self.w).std().dropna()
        if len(vol) < self.min_vol_history:
            return 0
        cur = vol.iloc[-1]
        thr = np.quantile(vol.iloc[:-1].values, self.q)
        return int(cur > thr)
