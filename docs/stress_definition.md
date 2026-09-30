# Stress-day definition (independent of the controller's regime detector)

This document defines what counts as a **stress day**, so that the regime layer
(`src/regime.py`, contribution #3) can be scored against labels it did not produce itself.

## Why this cannot come from `src/regime.py`

Grading a detector against its own decisions is circular: it would trivially agree with
itself. The rule below therefore uses **no statistic that `VolRegimeDetector` computes**.

| | `VolRegimeDetector` (in-controller) | this document (ground truth) |
|---|---|---|
| statistic | 21-day **rolling std** of the equal-weight market return | **cross-sectional dispersion** across assets, and the market return's **level** |
| threshold | **expanding** quantile (q80) of its own history | **frozen** quantile fitted once on train (2012-2015) |
| character | persistent level ("we are in a volatile regime") | episodic shock ("something happened today") |

The two are related — market turmoil does raise dispersion — but they are not
interchangeable. Measured overlap is reported below so a reader can judge the distance.

## The rule

For each day `t`, from the real NSE daily return matrix `R` (10 large caps):

```
disp_t       = std across the 10 assets of that day's returns      (ddof=1)
mkt_t        = mean across the 10 assets of that day's returns
```

Thresholds are fitted **on the training window only** (2012-01-03 → 2015-12-31, 982 days)
and then frozen — validation data plays no part in defining the labels:

```
disp_threshold       = quantile(disp, 0.80) over train  = 0.017613
market_return_thresh  = quantile(mkt,  0.10) over train  = -0.010790
```

A day is **stress** if either criterion fires:

```
stress_t = (disp_t >= 0.017613)  OR  (mkt_t <= -0.010790)
```

The first arm catches **disagreement** — assets diverging violently. The second catches
**common shocks** — everything selling off together, which is precisely when dispersion is
*low* and a dispersion-only rule would miss it. Both arms are episodic by construction.

## Episodes

```
runs        = maximal consecutive stress-day runs
merged      = runs joined when separated by <= 2 calm days
episodes    = merged runs of >= 3 days   (shorter ones dropped)
```

Merging stops a single calm day from splitting one crisis into several episodes.

## No look-ahead

- `disp_t` and `mkt_t` are same-day quantities, used to *label* days after the fact. They
  are never inputs to any trading decision, so they cannot leak into returns.
- Both thresholds are scalars fitted once on train and applied unchanged to every later
  day. No expanding/rolling refit, so no validation or test information enters the labels.
- Implementation: `src/stress.py` (`fit_thresholds`, `label_stress`, `find_episodes`).

## A criterion that was rejected

A **market drawdown ≥ 5% below running peak** rule was tried first and **discarded**. It
is *sticky*: once the market enters a bear market it stays below its peak, so it labels a
multi-year decline as one giant "stress episode".

Measured on train: 28.8% of days at a 5% threshold, collapsing into a **single 194-day
span** spanning 2015-08 → 2016-06. Peak-to-trough drawdown is still reported per episode as
*descriptive context* (`episode_market_drawdown`), but it does not define the label.

> Note on that function: the per-episode running peak is anchored at wealth `1.0` measured
> **before** the episode opens. Without that anchor the first day of an episode is always
> its own running maximum, which forces drawdown `= 0` on day one and makes the minimum
> drawdown identically zero for *every* episode. That bug was caught because the median and
> the maximum both came back as exactly `0.0%`.

## Resulting counts (real NSE, train+val only — test never touched)

Window `2012-01-03 → 2019-12-31`, 1965 days.

| quantity | value |
|---|---|
| stress days | 448 (**22.8%**) |
| episodes (≥3d, gap ≤2d merged) | **83** |
| stress days inside kept episodes | 344 |
| episode span: min / median / max | 3 / 5 / 27 days |
| episode market drawdown: median / p90 / max | 0.0% / 1.5% / 2.1% |
| validation-window stress days | 216 / 983 (**22.0%**), 44 episodes |

67.5% of episodes end at a new high for the market, which is consistent with the rule: many
stress episodes are *dispersion* shocks (idiosyncratic turmoil) rather than market crashes.
Episode market drawdowns are therefore modest by construction and should be read as
severity context, not as the definition.

### Longest 15 episodes

| start | end | split | span (d) | stress (d) | mkt DD |
|---|---|---|---|---|---|
| 2013-07-12 | 2013-08-21 | train | 27 | 16 | 0.0% |
| 2013-08-27 | 2013-10-03 | train | 26 | 15 | 0.0% |
| 2015-01-06 | 2015-02-03 | train | 20 | 12 | 0.0% |
| 2019-09-17 | 2019-10-10 | val | 16 | 11 | 0.0% |
| 2017-10-11 | 2017-11-01 | val | 15 | 7 | 0.0% |
| 2016-05-13 | 2016-06-01 | val | 14 | 7 | 0.0% |
| 2016-01-13 | 2016-02-01 | val | 13 | 9 | 0.0% |
| 2016-11-02 | 2016-11-21 | val | 13 | 7 | 0.5% |
| 2013-04-11 | 2013-04-30 | train | 12 | 8 | 0.0% |
| 2015-02-26 | 2015-03-13 | train | 11 | 8 | 0.0% |
| 2018-09-28 | 2018-10-15 | val | 11 | 10 | 0.0% |
| 2013-06-20 | 2013-07-03 | train | 10 | 6 | 0.0% |
| 2013-12-05 | 2013-12-17 | train | 9 | 6 | 0.0% |
| 2016-03-28 | 2016-04-07 | val | 9 | 5 | 0.0% |
| 2014-06-11 | 2014-06-23 | train | 9 | 4 | 0.0% |

The 2013 clusters are the taper tantrum; 2015-01 and 2015-08 are the sharp
NSE corrections around the rupee/FII episode; 2016 is demonetization; 2018-09→10 is the
lead-up to the IL&FS crisis.

## Independence diagnostic (not used to build labels)

Overlapping window where the detector is warm (1713 days):

| quantity | value |
|---|---|
| Jaccard overlap of stress sets | **0.166** |
| detector stress rate | 0.170 |
| this rule's stress rate | 0.240 |

An overlap of 0.17 means the two definitions agree on only ~17% of the union of their
stress days. They are clearly related but far from equivalent, which is exactly what is
required for the evaluation to be informative. Produced by
`src.stress.compare_with_regime_detector`, which is used **only** for reporting.

## Reproduce

```bash
python -m experiments.run_all --period val          # point estimates
python -m experiments.regime_conditional            # step 2
python -m experiments.block_bootstrap              # step 1
```