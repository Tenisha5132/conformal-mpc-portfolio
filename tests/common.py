"""Shared test-data helper.

Tests run on REAL market data from the on-disk cache (data/raw) whenever it covers
the requested window, so the no-look-ahead guarantees are verified against actual
NSE returns. When the cache is absent (fresh clone, no network) the suite falls back
to the config-driven synthetic generator so it still runs offline.

The data source actually used is recorded on the returned Series as `.attrs['source']`
and printed by `pytest -q`, so it is never ambiguous which data a test exercised.
"""
import copy
import functools

import pandas as pd

from src.data import download_prices, make_synthetic_prices, to_returns
from src.utils import load_config

CONFIG = "configs/default.yaml"


@functools.lru_cache(maxsize=None)
def _cached_real_returns():
    cfg = load_config(CONFIG)
    d = cfg["data"]
    try:
        prices = download_prices(d["tickers"], d["start"], d["end"], d["cache_dir"],
                                 min_coverage=d["min_coverage"], ffill_limit=d["ffill_limit"])
    except Exception:
        return None
    if prices.shape[1] < 3 or prices.shape[0] < 500:
        return None
    return to_returns(prices)


def get_returns(start="2012-01-02", end="2016-12-30", n_assets=5, seed=1):
    """Daily returns for tests. Real cached data if available, else the synthetic fixture."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    real = _cached_real_returns()
    # to_returns() drops the first day (pct_change), so allow a few days of slack on the left.
    if real is not None and real.index[0] <= lo + pd.Timedelta(days=5) and real.index[-1] >= hi:
        r = real.loc[lo:hi].iloc[:, :n_assets].copy()
        r.attrs["source"] = "real"
        return r

    cfg = copy.deepcopy(load_config(CONFIG))
    cfg["synthetic"].update(start=start, end=end, n_assets=n_assets)
    r = to_returns(make_synthetic_prices(cfg, seed=seed))
    r.attrs["source"] = "synthetic (cache unavailable)"
    return r
