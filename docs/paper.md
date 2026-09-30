# Self-Calibrating Uncertainty-Aware MPC for Tail-Risk Budgeting in Portfolios

> **Draft status.** Manuscript skeleton. Every number below is a validation-period (2016-01-01
> to 2019-12-31) measurement produced by the committed code; the sealed test period
> (2020-01-01 onward) has never been examined. Provenance for each figure is given in
> `docs/experiment_log.md`. Claims that our own experiments falsified are stated as
> limitations, not quietly dropped — see §6.

---

## Abstract

Portfolio risk limits are usually written as a single scalar: cap the exposure, or target a
volatility. Both are blunt. They are indifferent to the *shape* of the loss distribution, and
they cannot tighten in response to a forecast that turns out to be unreliable. We study the
alternative — a **hard budget on the tail**, imposed as an explicit constraint rather than a
scaled heuristic — and ask which risk instrument is actually doing the work.

Our contribution is a controlled instrument comparison rather than a performance claim. Using
a convex model-predictive-control allocator with adaptive conformal return intervals and a
self-calibrating uncertainty feedback, we hold the entire learning-and-control stack fixed and
swap *only* the risk instrument: a daily CVaR(5%) budget (Rockafellar–Uryasev, over a fixed
200-scenario Gaussian set) versus the scalar exposure cap it replaces. The tail budget reduces
5% CVaR by **+0.00159** (95% CI [0.00101, 0.00214], **p = 0.001**) and the effect replicates
under a forecast-ablation control (**p = 0.001**). This is our one robust, significant result,
and it is a mechanism claim, exactly as the title states.

We then report four hypotheses of our own that the data **falsified**, because a paper whose
only claim is the favourable one is not a paper. (i) The learned return forecaster is a pure
drag: removing it leaves every risk metric statistically unchanged (CVaR p = 0.606) while
cutting turnover and cost roughly fourfold. (ii) A volatility-jump regime, where a
trailing-volatility rule should be weakest, is not where the MPC wins (2/5 real events).
(iii) A forward-looking jump-augmented GARCH(1,1)-t volatility — given to the *baseline* as
well, so the comparison is at matched information — improves the simple rule
(CVaR 0.0117 vs 0.0131) and makes the MPC **worse**. (iv) The MPC's residual advantage is not
an artifact of de-risking: at matched realized exposure the MPC holds *more* risk (1.18×) for
*less* return.

The negative results are the substance. In this data the strongest risk controller we found is
the simplest one: volatility-targeted equal weight with a forward-looking volatility estimate.
Our contribution is to establish *which risk instrument matters* (the tail, not the scalar
cap), to show the mechanism is live and calibrated rather than slack, and to document the
failure modes of adding MPC structure on top of a working risk signal.

---

## 1. Introduction

Risk limits in a portfolio allocator are almost always a single number. A maximum exposure, a
volatility target, a drawdown-triggered de-scaling — each collapses risk into one scalar that
is then tuned. This is defensible, and in our experiments it turns out to be hard to beat.

The idea we study is that the *tail* deserves its own instrument. A portfolio's 5% CVaR is not
a function of its volatility: a book of many small losses and a book of one large loss can
share a volatility and differ enormously in tail. A scalar exposure cap cannot distinguish
them. A **tail-risk budget** can: it is a constraint on the quantity we actually care about,
and it is falsifiable in a way a tuned scalar is not.

The second ingredient is that a risk limit should respond to *how much we trust the
forecast*. If a conformal interval is wide, the forecast is uncertain; a controller that
ignores this sizes the same position regardless. We therefore close the loop: an online
adaptive-conformal layer produces per-asset intervals, a learned coefficient updates from
observed interval breaches, and both feed the risk instrument.

This yields a natural question that, as far as we can tell, is under-asked: **when you have a
learned forecast and an optimizer, does it matter what shape the risk limit takes?** The
forecast machinery, the optimizer, the transaction costs, and the data are all held fixed.
Only the risk instrument changes. This isolates the thing the title names.

Our answer: the instrument matters, and the tail is the right one — but only relative to the
*scalar cap it replaces*, not relative to a competent volatility-targeting baseline. That
distinction is the paper, and §6 states it without hedging.

### 1.1 Contributions

1. **A tail-risk budget as a first-class MPC constraint.** A daily CVaR(5%) budget via the
   Rockafellar–Uryasev epigraph over a fixed scenario set, calibrated so the constraint is
   *live* at full investment rather than slack (§3.3). Against the scalar exposure cap it
   replaces, the tail budget improves 5% CVaR at **p = 0.001**, replicated under a
   forecast-ablation control.
2. **A self-calibrating uncertainty loop that is verifiably live.** Online adaptive conformal
   intervals per asset, a learned breach-feedback coefficient, and a geometric uncertainty
   ratio driving the budget. We report the coverage achieved and the fraction of days the
   constraint binds, and we show the loop is not inert.
3. **A negative result, documented rather than buried.** Four self-generated hypotheses
   falsified (§6), each with a paired-bootstrap significance statement: the learned forecaster
   is a drag; no volatility-jump regime rescues the MPC; a forward-looking volatility estimate
   helps the simple rule and hurts the MPC; and the MPC's edge is not a de-risking artifact.
4. **A reproducibility discipline that made (3) possible.** Every strategy shares data,
   period, costs, and lookback; a no-look-ahead test runs for every new feature; a
   regime-search bound is unit-tested so a sealed-test event cannot be selected by accident;
   and the parameterization traps we hit (four in the volatility model) are recorded because
   each produced a plausible-looking, worthless signal.

---

## 2. Related work

*(To be written. Anchors in `references.bib`; none is yet cited in prose — see the gap list
in §7. The nearest prior work is decision-focused learning for portfolio optimization
(Leake & Lodha), with which our negative result should be positioned carefully.)*

---

## 3. Method

### 3.1 Allocator

*(Skeleton. Convex MPC over a fixed Gaussian scenario set, long-only simplex with exposure cap,
CLARABEL with tight tolerances, fixed scenario shocks by seed so the program stays a QP and the
backtest is reproducible.)*

### 3.2 Adaptive conformal return intervals

*(Skeleton. Per-asset online ACI; reported empirical coverage vs target on validation.)*

### 3.3 The tail-risk budget

The risk instrument. Write the daily loss of weights **w** over a fixed scenario set
**ε**¹…**ε**ᴿ as the sample CVaR at level α = 0.05, and impose it as a budget:

```
budget_t = vol_mult · σ_t · scale_t ,
scale_t  = 1 / (1 + β_t · max(ratio_t − 1, 0))          (floored at min_scale)
```

with σ_t the equal-weight volatility estimate, β_t the learned breach-feedback coefficient, and
ratio_t the conformal-width / ensemble-disagreement uncertainty ratio. The constraint itself is
the Rockafellar–Uryasev epigraph, which keeps the program convex:

```
CVaR_α(loss(w)) = 1/(α(1−α)R) · Σ_r u_r ,   u_r ≥ loss_r(w),  u_r ≥ 0
```

Because the scenario shocks are fixed, `loss_r(w)` is **linear** in **w** and the whole program
stays a QP. Two implementation details are load-bearing and were both learned the hard way:
the `1/R` factor must be retained (omitting it inflates the CVaR term R-fold and yields
over-budget solutions), and an infeasible budget must fall back to **cash** — never to the
previous weights, which are fully invested, i.e. the opposite of risk-off.

`vol_mult` is calibrated on **train only** (realized CVaR/vol ≈ 2.0–2.17) and set to 1.6, so
the budget sits just under the observed tail and binds on a substantial fraction of days
instead of being permanently slack. We report the binding fraction as a diagnostic: a
constraint that never binds is not a result.

### 3.4 Design note — what "self-calibrating" does and does not mean

The loop is verifiably *active* (calibrated breach frequency matches the target δ = 0.01), but
its effect on realized exposure is small, and we say so. The geometric ratio requires both the
conformal arm and the ensemble-disagreement arm to move before the budget shrinks, and δ = 0.01
starves the loop of feedback. We report the measured exposure cap and uncertainty ratio rather
than presenting the loop as if it were delivering large gains.

---

## 4. Experimental design

*(Skeleton. Ten NSE large caps, 2012-01-03 to 2015-12-31 train, 2016-01-01 to 2019-12-31
validation, 2020 onward sealed. Transaction costs on in every backtest including baselines
(10 bps one way). Fixed seed 42. Real prices only; the simulated generator exists so the test
suite runs offline and its runs print a warning banner. All hyperparameters in
`configs/default.yaml`.)*

### 4.1 The strategy ladder

`equal_weight`, `buy_hold`, `markowitz`, `equal_weight_voltarget`, `equal_weight_voltarget_fwd`,
`mpc_naive`, `mpc_dd_riskaversion`, `mpc_fc`, `mpc_fc_robust`, `mpc_fc_tight`, `mpc_selfcal`,
`mpc_tailbudget`, `mpc_tailbudget_nofc`, `mpc_tailbudget_fwdvol`, `mpc_full`.

Two controls deserve emphasis because they are what make the negative results credible:

- **`mpc_tailbudget_nofc`** removes the return forecast and changes nothing else — budget,
  conformal loop, scenarios, costs all retained. The forecast-off versus forecast-on difference
  is therefore attributable to the forecast alone.
- **`equal_weight_voltarget_fwd`** is a *matched-information* control: the scalar
  volatility-targeting rule driven by the **same** GARCH signal as the MPC. Without it, a
  "win" for the MPC would only show that a good signal beats no signal.

---

## 5. Results

### 5.1 The tail budget beats the scalar exposure cap (the surviving claim)

Holding the entire stack fixed and swapping only the risk instrument:

| comparison | ΔCVaR | 95% CI | p |
|---|---|---|---|
| `mpc_tailbudget` − `mpc_selfcal` | **+0.00159** | [0.00101, 0.00214] | **0.001** |
| `mpc_tailbudget_nofc` − `mpc_selfcal` | **+0.00185** | [0.00108, 0.00268] | **0.001** |

Paired stationary bootstrap. Both comparisons are MPC-versus-MPC with identical tightening
machinery, so neither is confounded by exposure level. This is the claim the title makes.

### 5.2 The forecaster is a pure drag

| strategy | Sharpe | ann. vol | CVaR | turnover | cost |
|---|---|---|---|---|---|
| `mpc_tailbudget_nofc` | 1.265 | 0.1105 | −0.0148 | 0.009 | 0.89% |
| `mpc_tailbudget` | 1.196 | 0.1113 | −0.0150 | 0.036 | — |

`mpc_tailbudget_nofc` − `mpc_tailbudget`: CVaR p = 0.606, Sharpe p = 0.622, max drawdown
p = 0.869. Removing the forecaster leaves risk unchanged and costs roughly a quarter of the
turnover. Per-asset correlation between prediction and realization is ≈ −0.035 on these names;
true daily AR(1) averages ≈ 0.004. There is no forecast skill here, and the paper does not
claim any.

### 5.3 A forward-looking signal helps the simple rule and hurts the MPC

Jump-augmented GARCH(1,1)-t, 1-step conditional volatility, train-only diagnostic: correlation
with |r| one step ahead 0.188 vs 0.170 for a 60-day trailing window, though rank correlation is
*worse* (0.091 vs 0.130) — real returns are not GARCH, and we report this rather than quoting
the synthetic case where the advantage is large.

Full validation, matched information:

| strategy | CVaR | Sharpe | maxDD |
|---|---|---|---|
| `equal_weight_voltarget_fwd` | **0.0117** | 1.299 | **−0.0937** |
| `equal_weight_voltarget` | 0.0131 | 1.302 | −0.1066 |
| `mpc_tailbudget_nofc` | 0.0148 | 1.265 | −0.1467 |
| `mpc_tailbudget_fwdvol` | 0.0156 | 1.241 | −0.1545 |

The forward signal improves the **simple** rule and degrades the MPC. On 5 de-clustered real
volatility-jump events in train+validation, the forward signal helps the scalar rule in 4/5 and
the MPC in 2/5.

### 5.4 The residual edge is not de-risking

Measured realized mean gross exposure, and every strategy rescaled to a common exposure:

| strategy | mean gross exposure |
|---|---|
| `equal_weight_voltarget_fwd` | 0.686 |
| `equal_weight_voltarget` | 0.767 |
| `mpc_tailbudget_nofc` | 0.767 |
| `mpc_tailbudget_fwdvol` | 0.784 |
| `mpc_tailbudget` | 0.812 |

The MPC holds **more** exposure than the baseline, up to 1.18×. At matched exposure the baseline
remains better on CVaR, Sharpe and drawdown with indistinguishable volatility. Between 11% and
69% of the raw CVaR gap was an exposure artifact; the remainder still favours the baseline. The
rescaling is an analytical normalization, not a tradeable backtest, so no cost conclusion is
drawn from it.

---

## 6. What we got wrong, and what it means

Four hypotheses of our own, all falsified on validation, all stated with the evidence that
killed them.

| # | Hypothesis | Verdict | Decisive evidence |
|---|---|---|---|
| 1 | The learned forecaster adds value | **False** | Forecast-off ties on risk (p = 0.606) at ¼ the turnover |
| 2 | The MPC wins when volatility jumps | **False** | 2/5 real events; mean CVaR **worse**; also false on the fixture built to favour it |
| 3 | A forward-looking risk signal rescues the MPC | **False** | Signal helps the scalar rule (4/5), MPC worse in 3/5 |
| 4 | The MPC's wins are just de-risking | **False** | It holds *more* risk (1.18×) for less return |

Hypotheses 2 and 3 share a root cause worth isolating, because it is the most transferable
finding here: **the budget is denominated in a volatility estimate, and so is the baseline.**
`budget_t = vol_mult · σ_t` inherits the same estimation lag as the volatility-targeting rule
it is meant to beat — differing only in that it constrains the tail rather than the mean. A
better σ does not fix that composition; it changes the level of a constraint whose interaction
with the exposure cap and the objective was never calibrated for the new scale. Beating a
competent scalar rule on a jump requires a risk signal that *leads* the jump, not one that
summarizes the recent past more cleverly.

The most likely reason the MPC loses while holding more risk: the added structure — epigraph,
scenario set, self-calibration, exposure cap — converts a good scalar risk estimate into a
worse allocation decision. This is the same lesson the volatility-managed-portfolio literature
reports, arriving from the opposite direction.

**The claim the title makes, and the one it does not.** The title is a mechanism claim, and it
is supported: *given* this allocator, budgeting the tail beats the scalar exposure cap at
p = 0.001, replicated under ablation. The title is **not** a claim that this MPC beats
volatility-targeting, and it must not be read that way. In our data a volatility-targeted
equal-weight rule with a forward-looking volatility estimate is the strongest risk controller
we found.

---

## 7. Limitations and open work

1. **The test period is unexamined.** All numbers are validation. A single 2020+ run would be
   the natural confirmation, and we deliberately have not taken it; the regime search is
   hard-bounded to `val_end` with a regression test, because the largest volatility jump in the
   sample (6.9×, Dec-2019→Mar-2020) is COVID and sits in the sealed period.
2. **One market, ten large caps, four years.** No cross-market replication.
3. **Sharpe is seed-sensitive**; tail metrics are the more reliable comparison here.
4. **The learned component is not carrying the result.** We report this rather than presenting
   the conformal loop as if it were contributing materially.
5. **Related work is not yet positioned.** No entry in `references.bib` is currently cited in
   prose. This must be fixed before submission, in particular against decision-focused learning
   for portfolio optimization and volatility-managed portfolios.

---

## Provenance

Numbers: `docs/experiment_log.md` (chronological, with the parameterization bugs recorded).
Figures: `docs/figures/`. Config: `configs/default.yaml`. Experiments: `experiments/run_all.py`
(main ladder), `experiments/regime_test.py` (jump windows), `experiments/exposure_matched.py`
(exposure control). Tests: `tests/` (124 passing, including no-look-ahead for every feature).
