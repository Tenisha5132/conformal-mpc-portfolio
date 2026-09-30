"""Data loading, synthetic data for offline testing, and split logic."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def download_prices(tickers, start, end, cache_dir="data/raw", force=False,
                    min_coverage=0.95, ffill_limit=3) -> pd.DataFrame:
    """Adjusted close prices from Yahoo Finance, cached on disk."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(("|".join(sorted(tickers)) + start + end).encode()).hexdigest()[:10]
    fp = cache / f"prices_{key}.csv"
    if fp.exists() and not force:
        return pd.read_csv(fp, index_col=0, parse_dates=True)

    import yfinance as yf
    raw = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    prices = prices.sort_index()
    # drop tickers with too many gaps, forward-fill short gaps, drop remaining NaN rows
    prices = prices.dropna(axis=1, thresh=int(min_coverage * len(prices)))
    prices = prices.ffill(limit=ffill_limit).dropna()
    prices.to_csv(fp)
    return prices


def make_synthetic_prices(cfg, seed=0) -> pd.DataFrame:
    """Simulated two-regime market. OFFLINE TEST FIXTURE ONLY.

    Every generator parameter comes from cfg['synthetic'] - there are no magic numbers
    in this function. Never report results computed on this data; the paper uses the
    real tickers in cfg['data'].
    """
    s = cfg["synthetic"]
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(s["start"], s["end"])
    n, n_assets = len(idx), s["n_assets"]
    regime = np.zeros(n, dtype=int)
    for t in range(1, n):
        p_stay = s["p_stay_calm"] if regime[t - 1] == 0 else s["p_stay_stress"]
        regime[t] = regime[t - 1] if rng.random() < p_stay else 1 - regime[t - 1]
    vol = np.where(regime == 0, s["vol_calm"], s["vol_stress"])
    mkt = s["drift"] + vol * rng.standard_normal(n)
    beta = rng.uniform(s["beta_low"], s["beta_high"], n_assets)
    idio = s["idio_vol"] * rng.standard_normal((n, n_assets))
    r = mkt[:, None] * beta[None, :] + idio
    r[1:] += s["momentum"] * r[:-1]                # weak AR(1) so a forecaster has something to find
    prices = s["price_base"] * np.cumprod(1 + r, axis=0)
    return pd.DataFrame(prices, index=idx, columns=[f"A{i}" for i in range(n_assets)])


def to_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns."""
    return prices.pct_change().dropna()


def period_bounds(splits_cfg: dict, period: str, index: pd.DatetimeIndex):
    """(start_date, end_date) inclusive for 'train' | 'val' | 'test'."""
    train_end = pd.Timestamp(splits_cfg["train_end"])
    val_end = pd.Timestamp(splits_cfg["val_end"])
    one = pd.Timedelta(days=1)
    if period == "train":
        return index[0], train_end
    if period == "val":
        return train_end + one, val_end
    if period == "test":
        return val_end + one, index[-1]
    raise ValueError(f"unknown period: {period}")
