"""INDEPENDENT stress labelling for evaluating the controller.

WHY THIS FILE EXISTS
--------------------
Evaluating the regime layer (contribution #3) requires ground-truth stress labels. Those
labels must NOT be produced by `src/regime.VolRegimeDetector`, otherwise the evaluation is
circular: the detector would be graded against its own decisions.

So the rule here is built from statistics that `VolRegimeDetector` never looks at:

    VolRegimeDetector : 21d rolling STD of the equal-weight market return
                        vs an EXPANDING quantile of that same series.

    This module      : (a) CROSS-SECTIONAL DISPERSION across assets (disagreement)
                       (b) the equal-weight MARKET RETURN hitting its own train-window
                           lower tail (common shock)
                        with BOTH thresholds fitted on the TRAIN window only and frozen.

Neither criterion is a rolling volatility statistic, so a day cannot be "stress" here
merely because the detector called it stress. `compare_with_regime_detector()` reports the
overlap so a reader can see they are related-but-distinct rather than interchangeable.

A drawdown-from-running-peak criterion was evaluated first and REJECTED: it is sticky, so
it labels a multi-year bear market as one giant "episode" (28.8% of train days at a 5%
threshold, with a single 194-day span across 2015-2016). Peak-to-trough drawdown is still
reported per episode as context, but it does not define the label.

NO LOOK-AHEAD CONTRACT
----------------------
`dd_t` uses a running maximum, which only ever references days <= t. The dispersion
threshold is a scalar fitted once on the train window and then applied unchanged to every
later day, so no validation or test information leaks into the labelling.
"""
import numpy as np
import pandas as pd


def market_index(returns: pd.DataFrame) -> pd.Series:
    """Equal-weight market proxy: the mean cross-asset return each day."""
    return returns.mean(axis=1)


def dispersion(returns: pd.DataFrame) -> pd.Series:
    """Cross-sectional dispersion: std ACROSS assets of that day's returns.

    Measures asset disagreement. It is not a volatility estimate of any index and is
    blind to whether the market rose or fell on the day.
    """
    return returns.std(axis=1, ddof=1)


def drawdown(returns: pd.DataFrame) -> pd.Series:
    """Drawdown of the equal-weight market from its running peak (causal running max)."""
    wealth = (1.0 + market_index(returns)).cumprod()
    return 1.0 - wealth / wealth.cummax()


def fit_thresholds(returns: pd.DataFrame, train_end, cfg: dict) -> dict:
    """Fit both thresholds on the TRAIN window ONLY, then freeze them.

    `train_end` is passed explicitly (rather than taken from the global split config) so
    the caller cannot accidentally fit on validation data.
    """
    train_end = pd.Timestamp(train_end)
    train = returns.loc[returns.index <= train_end]
    if len(train) < 30:
        raise ValueError("training window too short to fit stress thresholds")
    qd = float(cfg["dispersion_quantile"])
    qm = float(cfg["market_return_quantile"])
    return {
        "disp_threshold": float(np.quantile(dispersion(train), qd)),
        "market_return_threshold": float(np.quantile(market_index(train), qm)),
        "dispersion_quantile": qd,
        "market_return_quantile": qm,
        "train_end": train_end,
        "n_train_days": int(len(train)),
    }


def label_stress(returns: pd.DataFrame, thresholds: dict, cfg: dict) -> pd.Series:
    """Boolean stress label per day, from the FROZEN thresholds.

        stress(t) = dispersion(t)      >= train_disp_p80
                 OR market_return(t)   <= train_market_q10

    The first criterion catches days when assets disagree violently; the second catches
    days when everything sells off together (which is precisely when dispersion is LOW
    and a dispersion-only rule would miss it).
    """
    disp = dispersion(returns)
    mkt = market_index(returns)
    stress = ((disp >= thresholds["disp_threshold"]) |
              (mkt <= thresholds["market_return_threshold"]))

    warmup = int(cfg["warmup"])
    if warmup > 0:
        stress = stress.copy()
        stress.iloc[:warmup] = False
    stress.name = "stress"
    return stress.astype(bool)


def find_episodes(stress: pd.Series, cfg: dict) -> pd.DataFrame:
    """Maximal runs of stress days, merged across short calm gaps, with a minimum length.

    Returns a DataFrame with columns: start, end, n_stress_days, n_span_days.
    """
    min_days = int(cfg["min_episode_days"])
    merge_gap = int(cfg["merge_gap_days"])

    flags = stress.to_numpy(dtype=bool)
    dates = stress.index
    runs = []
    i = 0
    n = len(flags)
    while i < n:
        if not flags[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and flags[j + 1]:
            j += 1
        runs.append((i, j))
        i = j + 1

    # merge runs separated by <= merge_gap calm days
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1] - 1) <= merge_gap:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    rows = []
    for a, b in merged:
        span = b - a + 1
        if span < min_days:
            continue
        rows.append({
            "start": dates[a],
            "end": dates[b],
            "n_span_days": span,
        })
    if not rows:
        return pd.DataFrame(columns=["start", "end", "n_span_days"])
    out = pd.DataFrame(rows).set_index("start")
    out["n_stress_days"] = [
        int(flags[dates.get_loc(s): dates.get_loc(e) + 1].sum()) for s, e in zip(out.index, out["end"])
    ]
    return out


def episode_market_drawdown(returns: pd.DataFrame, episodes: pd.DataFrame) -> pd.Series:
    """Peak-to-trough market drawdown WITHIN each episode span - reporting context only.

    Deliberately NOT part of the label. A drawdown measured from the episode's own first
    day avoids the stickiness of a running peak carried across a long bear market.

    The running peak is anchored at the wealth level of 1.0 *before* the episode opens.
    Without that anchor the first day of an episode is always its own running max, which
    forces dd = 0 on day one and makes min(dd) identically zero for every episode.
    """
    mkt = market_index(returns)
    out = {}
    for start, end in zip(episodes.index, episodes["end"]):
        span = mkt.loc[start:end]
        if span.empty:
            out[start] = float("nan")
            continue
        wealth = (1.0 + span).cumprod()
        peak = wealth.cummax().clip(lower=1.0)   # never a peak below the pre-episode level
        out[start] = float((1.0 - wealth / peak).min())
    return pd.Series(out, name="episode_max_drawdown")


def compare_with_regime_detector(returns: pd.DataFrame, stress: pd.Series,
                                 cfg: dict) -> dict:
    """DIAGNOSTIC ONLY - quantify overlap with the in-controller detector.

    This never feeds back into the labels. It exists so the paper can state honestly how
    related the independent definition is to what the controller actually keys on.
    """
    from src.regime import VolRegimeDetector

    r = cfg["regime"]
    det = VolRegimeDetector(r["vol_window"], r["quantile"], r["min_history"], r["min_vol_history"])
    warmup = int(r["min_history"])
    flags = [det.update(returns.iloc[:i]) == 1 for i in range(warmup, len(returns))]
    common = stress.index[warmup:]
    if len(common) == 0 or len(flags) == 0 or len(flags) != len(common):
        return {"overlap": float("nan"), "n_days": 0}
    det_flags = pd.Series(flags, index=common, dtype=bool)
    sub = stress.loc[common].astype(bool)
    both = int((det_flags & sub).sum())
    union = int((det_flags | sub).sum())
    return {
        "overlap": float(both / union) if union else float("nan"),
        "detector_rate": float(det_flags.mean()),
        "label_rate": float(sub.mean()),
        "n_days": int(len(common)),
    }
