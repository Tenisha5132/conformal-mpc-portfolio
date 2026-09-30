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

## 2026-10-01 — `mpc_tailbudget`: the title mechanism actually implemented

The paper title claims **tail-risk budgeting**. Until now nothing in the objective had a
tail term: `src/mpc.py` maximized mean return minus a quadratic variance penalty minus
turnover. `kappa` shrinks the *mean* by a half-width (a worst-case-mean box adjustment),
which is not a tail risk measure. So "budgeting" was aspirational.

Added: a hard cap on the daily CVaR(5%) of the loss, via the Rockafellar & Uryasev epigraph

    min_eta { eta + 1/(alpha(1-alpha)) * E[(L-eta)_+] }  <=  budget

imposed as a CONSTRAINT with `eta` free, over a fixed 200-scenario Gaussian set
(`scenario_returns`, `r = mu + chol(cov) z`, `z` drawn once from a fixed seed). Because the
shocks are fixed, the path loss is LINEAR in `w`, so this adds only linear constraints and
keeps the problem a **QP** — it does not become an SOCP. CLARABEL with tight tolerances is
used for the budget path; the default solver returns `optimal_inaccurate` and *violates* the
budget by ~20%.

Budget: `budget_t = vol_mult * sigma_t * scale_t`, `sigma_t` = trailing equal-weight vol over
the lookback (no look-ahead), `scale_t = 1/(1 + beta_t*max(ratio_t-1,0))` — the same learned
`beta_t` and uncertainty ratio as `mpc_selfcal`, applied to the TAIL instead of the mean.
`vol_mult=1.6` is TRAIN-calibrated: realized CVaR/vol is ~2.0–2.17 in both train and val, so
1.6 sits just below the normal level and the constraint is LIVE at full investment rather
than permanently slack (the `risk_aversion>=20` failure mode already in this log).

`mpc_tailbudget` differs from `mpc_selfcal` in exactly ONE line: the `_compute_budget` hook.
Everything else (conformal, ensemble, beta_t, costs) is inherited, so the ablation is clean.

### Three bugs hit on the way (all silent)

1. **`cvar_of_paths` sign error.** Used `quantile(alpha)` and took `losses >= q`, which
   measures the BEST-case tail. For `loss = -return` the threshold is the `(1-alpha)`
   quantile and the tail is the upper end. Symptom: the budget never bound and utilization
   sat at a constant -0.00001 regardless of budget.
2. **Missing `1/S`.** The R-U term is `1/(alpha(1-alpha)) * E[(L-eta)_+]` where `E` is the
   empirical MEAN. Omitting `1/S` inflated the term 200x, so the solver returned
   `optimal_inaccurate` and produced solutions ~20% OVER budget. Fixed and asserted in
   `test_budget_is_respected_when_it_binds`.
3. **Infeasible-budget fallback returned `w0`**, i.e. fully invested — the exact opposite of
   risk-off. Now returns CASH, but ONLY on the budget path; the legacy no-budget path still
   holds `w0` so existing strategies are byte-for-byte unchanged.

### Result (validation 2016-2019, real NSE, 10 bps, bootstrap 2000 blocks)

| strategy | Sharpe | ann vol | CVaR 5% | maxDD | turnover |
|---|---|---|---|---|---|
| equal_weight_voltarget | 1.302 | 0.1050 | **-0.0131** | -10.7% | 0.014 |
| mpc_fc_tight | 1.073 | 0.1175 | -0.0163 | -14.7% | 0.033 |
| mpc_selfcal | 1.091 | 0.1204 | -0.0166 | -15.2% | 0.036 |
| **mpc_tailbudget** | **1.196** | **0.1113** | **-0.0150** | **-14.6%** | 0.036 |

Budget diagnostics: mean budget 0.01387, **binding on 54.2% of days**, mean utilization
0.869, **0 violations in 983 days** (max util 0.995).

Paired bootstrap, `mpc_tailbudget - mpc_selfcal` (the one-line ablation):

| metric | diff | 95% CI | p | sig |
|---|---|---|---|---|
| CVaR 5% | +0.00159 | [0.00101, 0.00214] | 0.001 | **YES** |
| Sharpe | +0.105 | [-0.072, 0.279] | 0.237 | no |
| maxDD | +0.0059 | [-0.0057, 0.0388] | 0.184 | no |

**The tail-risk budget beats the exposure cap on the tail, significantly, and that is exactly
the prediction the mechanism makes.** Sharpe/drawdown are not significantly different.

### What is still NOT claimed

- It does **not** beat `equal_weight_voltarget` on CVaR (diff -0.0019, p=0.082). The simple
  volatility-targeted baseline remains the hardest thing to beat on tails, and no MPC variant
  is best on Sharpe or CVaR. Honest claim: the budget is the best of the MPC family on tails,
  and it is the mechanism that finally makes "tail-risk budgeting" literally true.
- `docs/hypothesis.md` remains FALSIFIED as written: "at matched annualized volatility...
  lower CVaR than volatility-targeted equal-weight" is still false. Rewriting it is a
  separate decision and was NOT done here.
- vol_mult was fixed from TRAIN and never tuned on val. Do not tune it on val returns.
- `beta_t` alone is inert (scale uses `max(ratio-1,0)`, so beta with ratio=1 does nothing);
  the levers are multiplicative, matching `tighten_limits`.

## 2026-10-01 — the forecaster-off control: what the paper can and cannot claim

The `mpc_tailbudget` ablation showed the CVaR budget beats the exposure cap on the tail
(p=0.001), but `equal_weight_voltarget` still beat BOTH on CVaR. That raised an obvious
question this log had not answered: **is the MPC layer adding anything, or is the forecaster
just paying for itself with turnover?**

Since the ridge forecaster has NO skill here (mean per-asset corr(pred, realized) ~= -0.035),
the forecast entering the objective is close to noise. If it is noise, then removing it should
not hurt - and if the MPC still loses to a scalar vol-targeting rule without it, the whole
optimization layer is decorative.

Added `mpc_tailbudget_nofc`: identical to `mpc_tailbudget` EXCEPT `_blend_mu` returns the
historical mean instead of `mu_shrink*pred + (1-mu_shrink)*hist_mu`. It keeps the conformal
budget, the learned beta_t, the ensemble disagreement, the scenario set, and the cost model.
Only the return forecast is removed. This isolates the forecast, not the uncertainty loop.

### Result (validation 2016-2019, real NSE, 10bps, 2000-block paired bootstrap)

| strategy | Sharpe | ann vol | CVaR 5% | maxDD | turnover | total cost |
|---|---|---|---|---|---|---|
| equal_weight_voltarget | 1.302 | 0.1050 | -0.0131 | -10.7% | 0.014 | 1.34% |
| mpc_fc (forecaster, no budget) | 1.124 | 0.1251 | -0.0171 | -15.6% | 0.035 | 3.43% |
| mpc_selfcal | 1.091 | 0.1204 | -0.0166 | -15.2% | 0.036 | 3.54% |
| mpc_tailbudget (forecast ON) | 1.196 | 0.1113 | -0.0150 | -14.6% | 0.036 | 3.54% |
| **mpc_tailbudget_nofc (forecast OFF)** | **1.265** | 0.1105 | **-0.0148** | -14.7% | **0.009** | **0.89%** |

Paired bootstrap:

- nofc vs mpc_tailbudget (forecast ON vs OFF): CVaR +0.00025, p=0.606; Sharpe +0.069, p=0.622;
  maxDD p=0.869. **The forecast makes NO significant difference to any metric.**
- nofc vs mpc_selfcal (budget vs exposure cap): CVaR +0.00185, p=0.001. Significant.
- nofc vs equal_weight_voltarget: CVaR -0.0017, p=0.142; Sharpe p=0.953; maxDD p=0.836.
  **Still NOT significantly better than the scalar baseline.**

Stress-conditional (independent train-fitted rule, val): nofc stress CVaR -0.0225, worst day
-0.0417 (vs tailbudget -0.0234/-0.0395, selfcal -0.0259/-0.0436, voltarget -0.017/-0.021).
The nofc control has the best stress CVaR of the three MPC budget variants.

### VERDICT: the forecaster is a drag, and the MPC layer does not beat vol-targeting

1. **Removing the forecast is free.** CVaR, Sharpe, and maxDD are statistically identical
   (all p>0.6) with the forecast OFF, and it cuts turnover 4x (0.036 -> 0.009) and cost 4x
   (3.54% -> 0.89%). So the ridge forecaster contributes exactly nothing here except cost.
   This is the sharpest confirmation yet of the "forecasting does not add alpha here" finding.
2. **The tail-risk budget is the real contribution.** The one robust, significant effect is
   budget-vs-exposure-cap on CVaR (p=0.001, twice now: vs selfcal and vs nofc). Targeting the
   tail via R-U genuinely beats shrinking a mean-exposure cap, and this survives with the
   forecast removed.
3. **BUT the MPC layer still does not beat `equal_weight_voltarget`** on CVaR (p=0.142, not
   significant), Sharpe (p=0.953), or maxDD (p=0.836) - even with the forecast removed. The
   simple inverse-vol-scaled equal-weight baseline remains statistically indistinguishable
   from (and point-estimate-better than) the whole MPC stack on every headline metric.

### What this means for the paper title

"Self-Calibrating Uncertainty-Aware MPC for Tail-Risk Budgeting in Portfolios" is NOT
supported as a performance claim. It is supported as a MECHANISM claim only:
- Self-calibrating: YES - beta_t learned online, budget shrinks with breaches + uncertainty.
- Uncertainty-aware: YES - ACI drives the budget, calibrated (0.898 vs 0.90 target).
- Tail-risk budgeting: YES - hard R-U CVaR cap, live (binds 52-54% of days), 0 violations,
  beats the exposure-cap ablation at p=0.001.
- "...for portfolios" implying it improves portfolio risk: NO - it does not significantly beat
  a scalar vol-targeting baseline on this data/window.

The title needs either (a) softening to a mechanism/framework claim, or (b) more data/universe
where MPC might win. It should NOT stand as a superiority claim. This is a falsification of
the headline, recorded before touching test. Do not rescue it by tuning vol_mult on val.

## 2026-10-01 — vol-jump regime test: the hypothesis is FALSIFIED (twice)

The `mpc_tailbudget_nofc` result said the forecaster is a drag and the MPC still loses to
vol-targeting. I hypothesised the reason was the REGIME: 2016-2019 volatility was persistent,
which favours a trailing-vol estimator. In a regime where vol JUMPS and then mean-reverts, a
reactive tail-risk budget should beat scalar vol-targeting. I built the test to check this
rather than assuming it. It does not hold.

### 1. Synthetic mechanism test (`--synthetic`, NOT reportable)

`src/data.make_regime_prices`: deterministic block schedule calm -> spike -> decay, vol 0.008
-> 0.040 (5x) -> geometric decay. `momentum=0.0` so predictability cannot confound the risk
control. Full-sample: `equal_weight_voltarget` Sharpe 0.338 / CVaR -0.0176, `mpc_tailbudget_nofc`
Sharpe 0.235 / CVaR -0.0190, `mpc_tailbudget` Sharpe 0.072 / CVaR -0.0196. On the SPIKE phase
the MPC variants are marginally better (CVaR -0.0062 vs voltarget -0.0102), but in the DECAY
phase voltarget wins outright (-0.0241 vs -0.0262) and the vol-target's shallower drawdown
dominates. Net: hypothesis NOT supported even on the fixture built to favour it.

### 2. Real data, and a test-period trap

`find_real_jump_window` selects jump+revert events by a rule fixed BEFORE seeing any strategy
result. On the FULL frame the largest jump is **x6.94, peaking 2020-03 - the COVID crash,
which is SEALED TEST data.** An unbounded search would have silently grabbed the single event
most likely to flatter the method, from data we agreed never to touch. `search_end` now HARD
-BOUNDS the search to val_end, and `tests/test_regime_test.py` asserts the bound is respected
(the guard test fails if someone removes it).

Within train+val the best available jump is only **x2.16 (Oct 2018)** - a much weaker event
than COVID. De-clustered to peaks >40d apart, there are 5 qualifying jump+revert events.

### 3. Real result: MPC loses in 3 of 5 jump events

| event (peak) | voltarget CVaR | nofc CVaR | tailbudget CVaR |
|---|---|---|---|
| 2013-09-13 | 0.0147 | 0.0210 | 0.0202 |
| 2015-09-09 | 0.0176 | 0.0158 | 0.0161 |
| 2016-03-02 | 0.0139 | 0.0078 | 0.0096 |
| 2017-11-08 | 0.0131 | 0.0146 | 0.0150 |
| 2018-10-29 | 0.0132 | 0.0214 | 0.0205 |

nofc beats voltarget on CVaR in **2/5** events, mean diff **+0.00164 (worse)**. On the largest
event (2018-10-29) the MPC is dramatically worse (0.0214 vs 0.0132). The single best window
run in isolation (2018-07..2019-05) shows the same: voltarget Sharpe 1.73 / CVaR -0.0134 vs
nofc Sharpe 1.14 / CVaR -0.0219. n=5 events is small and cannot prove a null, but the
direction is consistent with the synthetic result and the full-period result.

### WHY the hypothesis failed (the real lesson)

The budget is denominated in TRAILING vol (`vol_mult * sigma_t`). When vol jumps, sigma_t lags,
so the budget is set from stale risk. It has the SAME trailing-window weakness as the vol
target it is meant to beat, just on the tail instead of the mean. A genuinely forward-looking
risk signal (implied vol, a jump model, an option surface) would be needed to actually lead
the jump - and none is available here. This is a design limitation, not a tuning problem, and
it is worth stating plainly rather than papering over.

### Bottom line for the paper

The title now has THREE falsification records, all on validation, all before test:
1. `docs/hypothesis.md` (matched-vol, beat voltarget) - falsified on the full val period.
2. `mpc_tailbudget_nofc` (strip the forecast) - forecaster is a pure drag, MPC still ties
   voltarget (p=0.14 CVaR).
3. Vol-jump regime test - the "regime where we win" hypothesis is falsified on both the
   fixture and real data (2/5 events, mean worse).

The tail-risk budget is still a real, calibrated, live mechanism that beats the exposure-cap
ablation at p=0.001. But there is NO regime found in which the MPC stack beats a simple
volatility-targeted equal-weight baseline. The honest, defensible paper is a MECHANISM /
negative-result paper, not a performance claim. Do not rescue the title by tuning vol_mult,
eta, or the event window on these results.

## 2026-10-01 — option 2: a FORWARD-LOOKING risk signal. Signal works, MPC still loses.

The vol-jump experiment above concluded the budget loses because it is denominated in
TRAILING vol. The only mechanism that could break a vol-targeting rule on a jump is a
forward-looking risk signal, so I built one and tested it PROPERLY.

### The signal: jump-augmented GARCH(1,1)-t (`src/fwdvol.py`)

Fitted by MLE with `scipy` (no new dependency: no `arch`/`statsmodels` installed). Student-t
innovations for fat tails plus a jump multiplier on the shock. The 1-step conditional
variance `h_{t+1} = omega + alpha*eps_t^2 + beta*h_t` is a statement about the FUTURE, and it
responds to the newest shock, which a 60-day trailing window cannot do at full speed.

Four real bugs found while building it, all of which had silently produced a WORTHLESS
signal while looking plausible:
1. **Inverted likelihood sign** - accumulated -loglik then returned -ll, so the optimizer
   MAXIMIZED the loss. Fit returned the starting point.
2. **`_pack` exponentiated all four params** including omega and nu, so nu=8 became e^8~3000
   and a variance of 0.05 became something else. Now only alpha/beta are log-parameterized.
3. **Additive jump term** injected ~1.0 into a daily variance of ~1e-3, swamping GARCH and
   pinning the forecast at the cap. Now a multiplier on the SHOCK, same units.
4. **Off-by-one in the forecast recursion** (`range(len(e)-1)`) - the most recent return was
   never propagated, so `h_{t+1}` ignored exactly the shock it is supposed to react to. The
   forecast was bit-identical for 0.5-sigma and 3-sigma shocks. This is the one that mattered.

Plus a **bound-collapse guard**: L-BFGS-B on this flat surface walks a good grid point down
onto the lower bounds of alpha/beta, where the objective is numerically better but the model
is degenerate (a flat, useless forecast). We reject candidates sitting on a bound.

### Is the signal actually forward-looking? (TRAIN ONLY, 2012-2015)

| predictor | corr with \|r_t\| | corr with \|r_t+1\| (1-step ahead) | Spearman |
|---|---|---|---|
| GARCH 1-step conditional vol | 0.193 | **0.188** | 0.091 |
| trailing 60d vol | 0.173 | 0.170 | 0.130 |

On synthetic data generated BY a GARCH the forecast beats trailing vol clearly (0.25 vs
0.18). On REAL NSE data it is only marginally better in Pearson correlation and clearly
WORSE in rank correlation. Real returns are not GARCH. So the signal is real but weak -
stated plainly rather than quoted from the favourable synthetic case.

### MATCHED-INFORMATION experiment (the part that matters)

Giving the MPC a better signal and then beating a baseline that lacks it would prove
nothing. So `equal_weight_voltarget_fwd` was added: the SAME GARCH signal driving a plain
scalar vol-target rule. Comparison is then at EQUAL information and isolates what the MPC
adds on top of the signal.

**Full validation 2016-2019:**

| strategy | CVaR 5% | Sharpe | CVaR/vol |
|---|---|---|---|
| equal_weight_voltarget (trailing) | 0.0131 | 1.302 | 1.984 |
| **equal_weight_voltarget_fwd (GARCH)** | **0.0117** | 1.299 | 1.994 |
| mpc_tailbudget (trailing vol) | 0.0150 | 1.196 | 2.146 |
| mpc_tailbudget_nofc | 0.0148 | 1.265 | 2.126 |
| mpc_tailbudget_fwdvol | 0.0156 | 1.241 | 2.146 |

**The 5 real jump+revert events (train+val), CVaR:**

| event | voltarget | voltarget_fwd | mpc_nofc | mpc_fwdvol |
|---|---|---|---|---|
| 2013-09-13 | 0.0147 | 0.0152 | 0.0210 | 0.0220 |
| 2015-09-09 | 0.0176 | 0.0171 | 0.0158 | 0.0162 |
| 2016-03-02 | 0.0139 | 0.0133 | 0.0078 | 0.0078 |
| 2017-11-08 | 0.0131 | 0.0111 | 0.0146 | 0.0162 |
| 2018-10-29 | 0.0132 | 0.0131 | 0.0214 | 0.0221 |

- Forward signal helps the SCALAR rule: 4/5 events, mean CVaR diff **-0.00052**. The premise
  of option 2 was CORRECT - a forward-looking signal is a better risk signal here.
- Forward signal does NOT help the MPC: mpc_fwdvol beats voltarget_fwd in only 2/5 events,
  mean diff **+0.00291 (worse)**. It is also worse than the trailing-vol MPC (+0.00164).

### Interpretation: the MPC DILUTES a good signal

This is the finding, and it is more interesting than a simple loss. The best risk controller
in the whole study is the dumbest one: `equal_weight_voltarget_fwd` at CVaR 0.0117. The MPC
layers - R-U epigraph, scenario set, self-calibration, exposure cap - convert a good scalar
risk estimate into a worse portfolio decision. Every MPC variant lands at 0.0148-0.0156
against the scalar rule's 0.0117. The tail-risk budget is not failing to react; it is
over-reacting relative to how much risk the (better) signal says is present.

Consistent with the AGENTS.md trap already recorded: the budget is denominated in a vol
estimate and then MULTIPLIED by vol_mult and shrunk by beta_t*(ratio-1). A better vol
estimate does not make that composition better; it changes the level of a constraint whose
interaction with the exposure cap and the objective was never calibrated for the new scale.

### Bottom line (fourth falsification record)

The forward-looking signal is the right idea and it works as a signal - but it makes the
SIMPLE rule better and the MPC worse. The paper title cannot be rescued this way either. The
defensible claims are now: (i) a hard CVaR budget beats an exposure cap at p=0.001, (ii) the
learned forecaster is a pure drag, (iii) a volatility-targeted equal-weight rule with a
forward-looking vol estimate is the strongest baseline found in this study, and (iv) added
MPC structure dilutes rather than improves a good risk signal. That is a coherent
negative-result paper with a clear, generalizable lesson. Do NOT tune vol_mult or jump_scale
against these numbers to manufacture a win.
