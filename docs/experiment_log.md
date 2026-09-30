# Experiment log

Every run, newest last. All numbers computed on **real NSE data** (10 large caps, cached in
`data/raw`). The test period (2020-01-01 onward) has **never** been read by any command
logged here. Synthetic data remains only the offline test fallback.

---

## 2026-09-30 — stress / bootstrap study on real data

### Step 0 — new baselines and resplit

**Splits changed.** `train_end` 2017-12-31 → **2015-12-31**, `val_end` unchanged at
2019-12-31. So train = 2012-01-03…2015-12-31 (982 days), validation = 2016-01-01…2019-12-31
(983 days), test = 2020-01-01 onward (sealed).

This invalidates every pre-existing `results/*` directory, which was scored on the old
2018–2019 validation window. Those runs are stale and must not be quoted.

**Two baselines added** (`src/baselines.py`):

- `equal_weight_voltarget` — vol-scaled equal weight, 10% annualized target, 60d window.
- `mpc_dd_riskaversion` — naive-mean MPC whose risk aversion is ×3 once **its own** equity
  drawdown exceeds 5%. It needs internal equity state, so `run_backtest` gained an optional
  `observe()` hook (`src/backtest.py:49`), called only *after* a day is realized. This
  preserves the no-look-ahead contract: the drawdown state used on day `d` reflects days
  ≤ d−1. Verified by a future-perturbation test.

> **Bug in `equal_weight_voltarget`, found and fixed.** It compared a **daily** volatility
> estimate against an **annualized** target, inflating the scale factor by √252 ≈ 16×. The
> scale was therefore pinned at its cap on every single day and the strategy returned
> returns byte-identical to `equal_weight`. My first explanation for that coincidence —
> "realized vol is below target, so vol-targeting never engages" — was **wrong**; the real
> cause was this units bug. After fixing the annualization the baseline behaves as intended
> and is no longer inert: avg exposure 0.73, validation maxDD **−10.7%** vs equal weight's
> −13.1%, at the cost of return (Sharpe 1.30 vs 1.38).

### Stress definition — correction to an earlier claim

An earlier message in this session reported "16 episodes / 277 stress days". Those came
from running `src/regime.VolRegimeDetector` and counting its own `True` runs — circular,
since that detector is the thing being evaluated. **Discarded.**

Replacement rule, full rationale and counts in `docs/stress_definition.md`. A
drawdown-from-running-peak criterion was tried first and **rejected** for stickiness
(28.8% of train days, collapsing into a single 194-day span over 2015–2016).

Final rule, thresholds fitted on **train only** then frozen:
`stress = (dispersion >= train p80 = 0.01761) OR (market return <= train q10 = -0.01079)`.

Result: **448 stress days (22.8%), 83 episodes** in 2012-01-03…2019-12-31; episode span
min/median/max = 3/5/27 days. Jaccard overlap with the controller's own detector: **0.166**.

**Bug found and fixed** in `episode_market_drawdown`: anchoring each episode's running peak
at the first day of the episode makes day one its own maximum, forcing drawdown = 0 and
making the minimum drawdown identically zero for every episode. Caught because median and
max both returned exactly `0.0%`. The peak is now anchored at wealth 1.0 measured *before*
the episode opens; median/p90/max are now 0.0% / 1.5% / 2.1%.

### Step 1 — stationary block bootstrap

`results/20260930-224541-val-bootstrap`. Stationary (Politis–Romano) bootstrap, mean block
10 days, 2000 resamples, seed 42, on real validation returns. Verified the resampler first:
empirical mean block length **9.93** (target 10), lag-1 autocorrelation of the index
sequence 0.886 — dependence preserved. One index sequence per replicate is shared across all
strategies, so differences are properly paired.

Marginal 95% intervals (Sharpe):

| strategy | point | lo | hi |
|---|---|---|---|
| equal_weight | 1.385 | 0.408 | 2.353 |
| buy_hold | 1.325 | 0.336 | 2.301 |
| markowitz | 0.967 | −0.009 | 1.935 |
| equal_weight_voltarget | 1.302 | 0.259 | 2.321 |
| mpc_naive | 1.252 | 0.291 | 2.226 |
| mpc_dd_riskaversion | 1.103 | 0.105 | 2.137 |
| mpc_fc | 1.124 | 0.168 | 2.098 |
| mpc_fc_robust | 1.115 | 0.145 | 2.109 |
| mpc_fc_tight | 1.073 | 0.107 | 2.067 |
| mpc_full | 1.160 | 0.175 | 2.159 |

**No Sharpe difference anywhere is significant.** Every paired Sharpe CI spans zero, against
both references. Bootstrap noise in Sharpe is ≈0.9 while the differences are ≈0.1–0.4, so
this window simply cannot separate these strategies on risk-adjusted return.

Significant paired differences, all in **risk**, none in return:

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| mpc_full − mpc_naive | CVaR | **+0.002** | [+0.001, +0.003] | 0.001 |
| mpc_dd_riskaversion − mpc_naive | CVaR | +0.001 | [+0.000, +0.002] | 0.001 |
| equal_weight_voltarget − mpc_naive | CVaR | +0.004 | [+0.001, +0.007] | 0.002 |
| markowitz − mpc_naive | CVaR | **−0.003** | [−0.005, −0.001] | 0.001 |

versus `equal_weight_voltarget` (negative diff = voltarget is better):

| comparison | metric | diff | 95% CI | p |
|---|---|---|---|---|
| mpc_full − voltarget | CVaR | −0.002 | [−0.004, +0.000] | 0.055 |
| mpc_fc_tight − voltarget | CVaR | −0.003 | [−0.006, −0.001] | 0.004 |
| mpc_naive − voltarget | CVaR | −0.004 | [−0.007, −0.001] | 0.002 |
| markowitz − voltarget | maxDD | −0.059 | [−0.177, +0.032] | 0.179 |
| equal_weight − voltarget | maxDD | −0.025 | [−0.052, −0.003] | 0.034 |

> **Bug found and fixed** in the bootstrap p-value. Two successive errors: (1) `2*min(frac≤0,
> frac≥0)` on the *mean-centred* differences double-counted the mass straddling zero and
> returned p ≈ 0.97–0.99 for everything; (2) the percentile test compares the **uncentred**
> replicate distribution against zero. Verified against synthetic cases with known effect
> sizes — p falls monotonically (0.97 → 0.96 → 0.74 → 0.31 for δ = 0, 0.05, 0.3, 1.0) and
> CIs match theory.

### Step 2 — regime-conditional performance

`results/20260930-224813-val-stress`. Validation only, independent labels (216 stress days,
767 calm).

| strategy | Sharpe stress | Sharpe calm | CVaR stress | CVaR calm | worst stress | worst calm |
|---|---|---|---|---|---|---|
| equal_weight | −1.37 | 3.25 | −0.0233 | −0.0095 | −0.0297 | −0.0107 |
| buy_hold | −1.54 | 3.30 | −0.0225 | −0.0093 | −0.0308 | −0.0132 |
| markowitz | −2.19 | 2.79 | −0.0275 | −0.0127 | −0.0458 | −0.0238 |
| **equal_weight_voltarget** | −1.33 | 3.04 | **−0.0173** | **−0.0076** | **−0.0211** | −0.0103 |
| mpc_naive | −2.56 | 3.06 | −0.0259 | −0.0120 | −0.0450 | −0.0244 |
| mpc_dd_riskaversion | −2.45 | 2.93 | −0.0232 | −0.0109 | −0.0450 | −0.0213 |
| mpc_fc | −2.84 | 3.27 | −0.0280 | −0.0113 | −0.0470 | −0.0236 |
| mpc_fc_robust | −2.88 | 3.23 | −0.0276 | −0.0112 | −0.0458 | −0.0233 |
| mpc_fc_tight | −2.91 | 3.17 | −0.0265 | −0.0108 | −0.0441 | −0.0200 |
| mpc_full | −3.05 | 3.27 | −0.0241 | −0.0100 | −0.0421 | −0.0138 |

Max drawdown is deliberately omitted on these subsets: on a set of scattered days the
peak-to-trough sequence is not a path, so the number would be meaningless.

Read honestly: **`equal_weight_voltarget` has the best stress-day tail of everything tested,
and it beats the entire MPC family.** On stress days its CVaR is −0.0173 against `mpc_full`
−0.0241 and `mpc_naive` −0.0259; its worst day is −0.0211 against −0.0421. It also has the
best validation maxDD (−10.7%).

What the MPC ladder *does* show, within its own family, is a monotone tail improvement from
`mpc_fc` (−0.0280) → `mpc_fc_robust` (−0.0276) → `mpc_fc_tight` (−0.0265) → `mpc_full`
(−0.0241), with worst-day loss shrinking −0.0470 → −0.0421. Lower exposure buys that, and it
costs return.

### Step 3 — per-episode exposure

`results/20260930-224813-val-stress/episode_exposure_all.png` (all **83** episodes, shared
axes, red shading = episode span) and `episode_exposure_mean.png` (average trajectory
aligned at episode start).

Mean exposure at episode start / +10d / +20d:

| strategy | start | +10d | +20d | full-window avg |
|---|---|---|---|---|
| equal_weight | 1.000 | 1.000 | 1.000 | 1.00 |
| equal_weight_voltarget | 0.721 | 0.669 | 0.699 | 0.73 |
| mpc_naive | 0.880 | 0.864 | 0.891 | 0.91 |
| mpc_dd_riskaversion | 0.708 | 0.686 | 0.706 | 0.75 |
| mpc_fc_tight | 0.838 | 0.805 | 0.839 | 0.86 |
| mpc_full | 0.789 | 0.751 | 0.765 | 0.81 |

Exposure does **not** dip further at +10d/+20d than at the episode start for any strategy —
the de-risking is a standing posture driven by a slow rolling signal, not a reaction to the
shock. `equal_weight_voltarget` scales fastest and sits lowest of all.

---

## Net reading of this study

1. The forecaster has no skill (established earlier); nothing here overturns it.
2. On 2016–2019 real data, **no strategy significantly beats another on Sharpe.** Every
   paired Sharpe interval spans zero against both references.
3. The significant effects are all in **tail risk and drawdown**. Within the MPC family the
   tightening and regime layers do improve CVaR relative to `mpc_naive`
   (`mpc_full` +0.002, p = 0.001).
4. **But a vol-targeted equal-weight portfolio beats the whole MPC family on tails** — better
   stress-day CVaR, better worst day, better validation max drawdown — and does it with less
   machinery.

### Honest implication for the paper

The data do **not** support a claim that uncertainty-driven constraint tightening improves
risk-adjusted performance over a simple vol-targeted equal weight. On this window the
generic risk control wins. Two readings are available and neither is settled by this study:

- *Tightening helps the MPC controller relative to its own un-tightened version* — supported
  (`mpc_full` vs `mpc_naive`, p = 0.001), but a weak claim.
- *Tightening helps in windows where volatility targeting fails* — untested. Vol-targeting
  wins here because this period's volatility was persistent enough for a slow estimator to
  track it. The mechanism would matter more where vol is mean-reverting or jumps, which this
  window does not contain.

Claiming superiority over equal weight on the basis of these numbers would be wrong. The
defensible contribution remains narrower: tightening makes an unreliable forecaster
*harmless*, and the tail improvements it delivers inside the MPC family are real and
statistically supported. Testing the second reading would require more regimes, which is the
natural next piece of work.
---

## 2026-09-30 — `mpc_selfcal`: self-calibrating tightening + ensemble disagreement

**Artifacts:** `results/20260930-225856-val/` (ladder), `results/20260930-230706-val-bootstrap/`.
Validation only (2016-01-01 → 2019-12-31). Test period untouched; no `--period test` was run.

### What was added
`src/selfcal.py` with two independent mechanisms, and `MPCSelfCalStrategy` in `src/strategies.py`
inheriting MPC/forecaster/conformal unchanged so the ladder comparison stays controlled.

1. **Learned `beta_t`.** `BetaState` advances `beta_{t+1} = max(0, beta_t + eta*(breach_t - delta))`
   with `breach_t = 1{net_return_t < -loss_threshold}`. This learned scalar replaces the fixed
   `tighten.beta` constant in the SAME `tighten_limits` call, so nothing else in the controller
   changes. Driven only by `observe()`, which the backtest calls after a day is fully realized.
2. **Ensemble disagreement.** `ridge + historical-mean + momentum (+GRU if torch)`. The signal is
   the mean over assets of the cross-model standard deviation of the predicted return. Final
   tightening signal is the geometric mean of the conformal-width ratio and the disagreement
   ratio, each divided by its own trailing median.

Config: `beta0=0.0`, `loss_threshold=0.02`, `delta=0.01`, `eta=0.05`, `ensemble.history=250`.

### Validation results (all costs at 10 bps)

| strategy | Sharpe | maxDD | turnover |
|---|---|---|---|
| mpc_naive | 1.25 | -15.5% | 0.008 |
| mpc_fc_tight | 1.07 | -14.6% | 0.033 |
| **mpc_selfcal** | **1.09** | **-15.2%** | 0.036 |
| mpc_full | 1.16 | -13.6% | 0.033 |

**The calibration itself worked.** Realized breach frequency over the validation window was
**0.0112** against a target `delta = 0.01`, and 0.0119 over the last 250 days. `beta_t` averaged
0.097, peaked at 0.322, and ended at 0.249 — it engaged and stayed engaged rather than
collapsing to zero.

**The performance effect did not materialize.** Paired bootstrap (2000 stationary-block
resamples, mean block 10d) vs the two references:

| vs | metric | diff | 95% CI | p |
|---|---|---|---|---|
| mpc_naive | CVaR | +0.0003 | [-0.0003, +0.0010] | 0.331 |
| mpc_naive | Sharpe | -0.161 | [-0.394, +0.082] | 0.194 |
| mpc_fc_tight-equivalent voltarget | CVaR | -0.0035 | [-0.0061, -0.0011] | 0.002 |

No Sharpe or max-drawdown difference is significant against either reference. Against
`equal_weight_voltarget` the CVaR difference is *negative and significant* (`p=0.002`), i.e.
`mpc_selfcal` is worse on tails than vol-targeting — consistent with the pre-existing finding that
vol-targeting dominates this validation window.

### Honest diagnosis of the null result

The mechanism is active but the tightening is far too weak to move risk, and I verified this
rather than assuming it:

- mean exposure cap over the run was **0.9908** (min 0.8501) — the controller is essentially
  unconstrained, so `beta_t` has almost nothing to tighten;
- `corr(beta_t, exposure_cap) = -0.489`: beta does pull the cap down, just not far;
- mean uncertainty ratio was **1.0156** (max 1.7634), so the geometric mean is near 1 nearly always.
  Taking a geometric mean of two trailing-median-normalized ratios means the signal only moves
  when *both* arms move together, which is a deliberately conservative choice and here it is
  too conservative.

Two further contributors: (a) with `loss_threshold=0.02` and `delta=0.01`, breaches are rare, so
`beta` saw ~11 breach events in 983 days — the loop is starved of feedback; (b) mean gross
exposure was 0.839 for `mpc_selfcal` vs 0.815 for `mpc_fc_tight`, i.e. it was not even taking
less risk, it was just slower.

`delta=0.01` is a target frequency, not a claim about optimality, and it was fixed a priori from
the user's specification, not tuned on validation. **I did not tune `eta`, `loss_threshold`, or
`delta` against validation returns after seeing this null result**, because that would be fitting
the evaluation window. If the paper wants a self-calibration ablation to show an effect, the
defensible knobs to vary a priori are `delta` (a stated design target, e.g. 0.02–0.05) and a
stronger response function than the geometric mean (e.g. arithmetic mean, or raising the tightening
slope in `mpc.tighten`). That is a design decision to fix before looking at results, not after.

### Bugs caught while testing this
Four test failures in the first draft of `tests/test_selfcal.py`, all mine, all fixed:
1. Assumed the calm-day decrement was `eta`; it is `eta*delta`. Up-steps (`eta*(1-delta)`) and
   down-steps (`eta*delta`) are deliberately asymmetric.
2. Misread a trailing-window breach count (1 of 2, not 0 of 2).
3. Off-by-one on the beta trajectory length — there is one update per *settled* day, so the last
   day's beta is computed but never used for a decision.
4. **Substantive:** the variance-shift test originally compared the post-shift breach rate against
   the *calm pre-shift* baseline, which is not what the feedback loop controls. Rewrote it to run the
   identical return path with and without feedback and compare post-shift against post-shift.
   Unmanaged post-shift breach rate is 0.174; with feedback it settles at 0.010, i.e. exactly
   `delta`. The mechanism is sound — the null result is about signal strength, not correctness.

Also verified: the ensemble omits the GRU cleanly when torch is unavailable rather than raising, and
its disagreement statistic is exactly 0 when all members are forced to agree.

### Test coverage added (`tests/test_selfcal.py`, 22 tests)
beta flooring at zero; exact closed-form agreement over a 8-day path; monotonic response to breach
rate; asyymmetric step sizes; variance-shift convergence to `delta`; geometric-mean arithmetic;
`tighten_limits` beta-response contract (beta=0 inert, larger beta tighter); ensemble membership and
torch-free degradation; zero disagreement for identical models; **no look-ahead** via the shared
`check_no_lookahead` harness plus an explicit assertion that `decide()` never mutates `beta`;
costs charged; one log row per decision.
