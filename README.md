# Self-Calibrating Uncertainty-Aware MPC for Tail-Risk Budgeting in Portfolios

**Manuscript:** [`paper.tex`](paper.tex) — IEEE journal format, title retained exactly.
Build with `pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex`.
Prose notes and the same numbers live in [`docs/paper.md`](docs/paper.md).

A learned return forecaster feeds adaptive conformal prediction intervals; those intervals
drive **uncertainty-dependent constraint tightening** inside a CVXPY model-predictive
controller, which re-solves a convex programme every trading day and imposes a hard daily
CVaR(5%) budget via the Rockafellar–Uryasev epigraph. When forecast uncertainty spikes, the
risk limits tighten automatically.

> **The honest headline: this project does not show that forecasting helps, and it does not
> show that the MPC beats a simple risk rule.** Four hypotheses of ours are falsified on
> purpose, and the negative results are the contribution:
> 1. The forecaster is a **pure drag** — removing it leaves risk unchanged (`p = 0.606`) at
>    one quarter of the turnover.
> 2. The MPC does **not** win in volatility-jump regimes (**1/4** real events, mean CVaR worse).
> 3. A forward-looking GARCH signal, given to the *baseline too* so the comparison is at
>    **matched information**, significantly **improves the simple rule** (`p = 0.001`) and
>    significantly **degrades the MPC** (`p = 0.001`). The MPC dilutes a good signal.
> 4. The MPC's tail advantage is **not** de-risking — at matched exposure it holds up to
>    **1.18× more risk** for less return.
>
> What survives is narrower and defensible: **adding a tail-risk budget to this allocator
> improves its tail** (`+0.00159`, CI `[0.00101, 0.00214]`, `p = 0.001`), replicated under a
> forecast-ablation control. That is a mechanism claim, not a superiority claim. See
> [What this does not show](#what-this-does-not-show).

---

## Table of contents

- [Quick start](#quick-start)
- [The pipeline](#the-pipeline)
- [Data and splits](#data-and-splits)
- [The strategy ladder](#the-strategy-ladder)
- [Results](#results)
- [The matched-information test](#the-matched-information-test)
- [Exposure-matched control](#exposure-matched-control)
- [Volatility-jump regime study](#volatility-jump-regime-study)
- [Stress-conditional analysis](#stress-conditional-analysis)
- [Self-calibration ablation](#self-calibration-ablation)
- [What this does not show](#what-this-does-not-show)
- [Bugs found and fixed](#bugs-found-and-fixed)
- [How the study was built, step by step](#how-the-study-was-built-step-by-step)
- [Repository layout](#repository-layout)
- [Reproducing](#reproducing)
- [Project rules](#project-rules)

---

## Quick start

```bash
make setup && source .venv/bin/activate

make test        # 130 tests
make synthetic   # offline smoke test on SIMULATED data — never reportable
make val         # real NSE data (10 tickers), validation period 2016–2019
```

`make synthetic` exists so the suite runs with no network. Its output must never appear in
the paper; `AGENTS.md` rule: *reported numbers come from real NSE prices only.*

---

## The pipeline

```mermaid
%% Canonical source: docs/flow.mmd — edit that file, then re-render.
%% Prose explanation follows the diagram.
flowchart TD
    RET["returns[:d]<br/>10 NSE large caps"] --> HIST["history window<br/>lookback L = 60 days"]

    HIST --> FOR["RidgeForecaster<br/>walk-forward, retrain every 63d<br/>corr with realized ≈ −0.035"]
    HIST --> ENS["DisagreementEnsemble<br/>ridge · historical mean<br/>momentum · GRU if torch"]
    HIST --> COV["shrunk covariance<br/>Ledoit–Wolf style"]

    FOR --> ACI["AdaptiveConformal<br/>per-asset α_t<br/>α_t+1 = α_t + γ(α − err_t)"]
    ACI --> HW["half-widths hw<br/>quantile of |y − ŷ| at level 1 − α_t"]

    HW --> RHO["uncertainty ratio ρ_t"]
    ENS --> RHO
    RHO --> TIGHT["tighten_limits — CORE<br/>exposure_cap = E_max / (1 + β·(ρ_t − 1))"]
    BETA["BetaState — selfcal only<br/>β_t+1 = max(0, β_t + η(breach_t − δ))"] --> TIGHT

    FOR --> MU["expected return μ<br/>μ_shrink·ŷ + (1 − μ_shrink)·mean"]
    HIST --> MU
    COV --> QP["MPC — convex QP (CVXPY)"]
    HW --> QP
    MU --> QP
    TIGHT --> QP
    QP --> W["weights w_d"] --> TURN["turnover"] --> COST["cost 10 bps"] --> REAL["realised net return"]

    REAL -. "conformal score · breach signal" .-> ACI
    REAL -. "β_t update" .-> BETA

    classDef core fill:#fff3cd,stroke:#d39e00,stroke-width:3px,color:#000
    classDef weak fill:#f8d7da,stroke:#842029,stroke-width:2px,color:#000,stroke-dasharray:4 3
    class TIGHT,RHO core
    class FOR weak
```

The full diagram — including the regime layer, the execution path, and the complete
feedback loop — lives in [`docs/flow.mmd`](docs/flow.mmd) as editable Mermaid source,
with a rendered copy at `docs/figures/flow.svg`.
`tests/test_figures.py` checks that the node ids there still match the seven blocks in
`docs/figures/block_diagram.png`, so the two renderings cannot silently drift apart.

**Core mechanism — `tighten_limits` in `src/mpc.py`.** Given an uncertainty ratio `ρ_t ≥ 1`
(half-widths relative to their trailing median), the exposure and volatility caps are shrunk:

```
exposure_cap_t = max_exposure · 1 / (1 + β·(ρ_t − 1))
```

`β` is the tightening strength. `mpc_fc_tight` uses a fixed `β`; `mpc_selfcal` *learns* `β_t`
online from realized breaches. Higher uncertainty ⇒ lower cap ⇒ smaller positions ⇒ less
drawdown risk. The optimisation stays a convex QP.

**Tail-risk budget — `mpc_tailbudget`, the paper's title mechanism.** `tighten_limits` shrinks
a *mean* exposure cap, which is a blunt instrument: it does not distinguish a mild drawdown
from a tail event. `mpc_tailbudget` instead caps the **CVaR of the daily loss** directly, using
the Rockafellar–Uryasev epigraph (Rockafellar & Uryasev 2000) over a fixed Gaussian scenario
set:

```
budget_t = vol_mult · σ_t · 1/(1 + β_t·(ρ_t − 1))          σ_t = trailing equal-weight vol
CVaR_α( loss ) ≤ budget_t                                  α = 0.05, 200 scenarios
```

The epigraph is imposed as a **constraint** with `η` free, so it adds only linear constraints
and the problem stays a **QP** (the scenarios are fixed draws, making the loss linear in `w`).
The same learned `β_t` and uncertainty ratio `ρ_t` that shrink the exposure cap instead shrink
the *tail* budget. `vol_mult = 1.6` is TRAIN-calibrated: realized CVaR/vol is ~2.0–2.17, so the
constraint is **live** at full investment (binds on 54% of days) rather than permanently slack.
See `docs/experiment_log.md` for the three silent bugs this took.

**Adaptive conformal prediction (`src/conformal.py`).** Gibbs & Candès (2021) ACI. Per-asset
miscoverage adapts online, `α_{t+1} = α_t + γ(α − err_t)`, so the interval widens on assets
where the model keeps being wrong and narrows where it is calibrated.

---

## Data and splits

| | window | trading days | use |
|---|---|---|---|
| train | 2012-01-03 → 2015-12-31 | 982 | fitting, threshold estimation |
| **validation** | **2016-01-01 → 2019-12-31** | **983** | **all reported numbers** |
| test | 2020-01-01 → | — | **sealed — never read** |

Real adjusted-close prices for 10 NSE large caps, 2012-01-02 → 2025-12-30, cached in
`data/raw/` (git-ignored; re-downloadable). One-way transaction costs **10 bps**, applied in
every backtest including baselines. Seed **42**; 252 trading days/year.

Every hyperparameter lives in `configs/default.yaml`. There are no magic numbers in code.

**Splits are a governance artifact, not a convenience.** `experiments/run_all.py` refuses
`--period test` unless invoked with an explicit `--final` instruction. No command in this
repository has ever run on the test period.

---

## The strategy ladder

Each rung adds exactly one element, so its contribution is measurable. `ALL` in
`experiments/run_all.py` is the source of truth.

| strategy | what it adds |
|---|---|
| `equal_weight` | 1/N, rebalanced |
| `buy_hold` | buy at the start, never touch |
| `markowitz` | mean–variance, shrunk covariance |
| `equal_weight_voltarget` | 1/N scaled to a 10% vol target, 60d |
| `mpc_naive` | MPC, no forecast, no conformal, no tightening |
| `mpc_dd_riskaversion` | MPC + drawdown-adaptive risk aversion |
| `mpc_fc` | + forecaster |
| `mpc_fc_robust` | + κ-penalised worst-case μ |
| `mpc_fc_tight` | **+ uncertainty-driven tightening (core)** |
| `mpc_selfcal` | learned `β_t` + ensemble disagreement |
| **`mpc_tailbudget`** | **+ hard CVaR(5%) tail-risk budget (title mechanism)** |
| `mpc_tailbudget_nofc` | the same, **forecast removed** — isolating control |
| `mpc_tailbudget_fwdvol` | budget denominated in GARCH forward vol, not trailing vol |
| `equal_weight_voltarget_fwd` | **matched-information scalar rule** — same GARCH signal as the MPC above |
| `mpc_full` | + regime layer |

`mpc_tailbudget` **subclasses** `mpc_selfcal` (`src/strategies.py:204`), so the two differ by
exactly one constraint: the CVaR budget is **added to** the same uncertainty-driven exposure
cap, not substituted for it. That is what makes their comparison a clean instrument test.

`equal_weight_voltarget_fwd` and `mpc_tailbudget_fwdvol` exist specifically as a
**matched-information pair**. Giving the same forward signal to both rules out the objection
that a "win" merely shows a good signal beating no signal.

---

## Results

Validation period 2016-01-01 → 2019-12-31, 983 days, costs on. Sharpe/CVaR/max-drawdown
intervals are 95% stationary block bootstrap (2000 resamples, mean block 10 days), seed 42.

| strategy | Sharpe | 95% CI | CVaR 5% | maxDD |
|---|---|---|---|---|
| `equal_weight` | 1.385 | [0.41, 2.35] | −0.0172 | −13.1% |
| `buy_hold` | 1.325 | [0.34, 2.30] | −0.0170 | −13.1% |
| `markowitz` | 0.967 | [−0.01, 1.94] | −0.0199 | −15.6% |
| **`equal_weight_voltarget`** | 1.302 | [0.26, 2.32] | **−0.0131** | **−10.7%** |
| `mpc_naive` | 1.252 | [0.29, 2.23] | −0.0170 | −15.5% |
| `mpc_dd_riskaversion` | 1.103 | [0.11, 2.14] | −0.0158 | −15.2% |
| `mpc_fc` | 1.124 | [0.17, 2.10] | −0.0171 | −15.6% |
| `mpc_fc_robust` | 1.115 | [0.14, 2.11] | −0.0170 | −15.5% |
| `mpc_fc_tight` (core) | 1.073 | [0.11, 2.07] | −0.0163 | −14.6% |
| `mpc_selfcal` | 1.091 | [0.12, 2.09] | −0.0166 | −15.2% |
| **`mpc_tailbudget`** | **1.196** | [0.21, 2.20] | **−0.0150** | **−14.6%** |
| `mpc_tailbudget_nofc` | 1.265 | [0.23, 2.30] | −0.0148 | −14.7% |
| `mpc_tailbudget_fwdvol` | 1.241 | — | −0.0156 | −15.5% |
| **`equal_weight_voltarget_fwd`** | 1.299 | — | **−0.0117** | **−9.4%** |
| `mpc_full` | 1.160 | [0.18, 2.16] | −0.0151 | −13.6% |

`equal_weight_voltarget_fwd` has the best tail of every strategy tested. See
[The matched-information test](#the-matched-information-test).

**No Sharpe difference is statistically significant.** Bootstrap Sharpe noise is roughly ±0.9;
every observed gap is 0.1–0.4. The significant effects are all in risk:

- `mpc_full` − `mpc_naive`: CVaR **+0.002**, paired bootstrap `p = 0.001`. The core layer helps
  *within* the MPC family.
- `mpc_tailbudget` − `mpc_selfcal`: CVaR **+0.00159**, paired bootstrap `p = 0.001`
  (CI [0.00101, 0.00214]). `mpc_tailbudget` **subclasses** `mpc_selfcal`, so these two differ
  by exactly one constraint — a hard CVaR budget *added to* the same uncertainty-driven
  exposure cap — so this is the cleanest evidence that *budgeting the tail* helps. Sharpe
  (+0.11, p = 0.24) and maxDD (p = 0.18) do not differ significantly.
- versus `equal_weight_voltarget`, `mpc_fc_tight` is **worse** on CVaR (p = 0.004), and
  `mpc_tailbudget` is still behind it (diff −0.0019, **p = 0.082**, not significant). The
  simple volatility-targeted baseline remains the hardest thing to beat on tails.

**The forecaster is a drag, and we can now prove it.** `mpc_tailbudget_nofc` is
`mpc_tailbudget` with exactly one change: `_blend_mu` returns the historical mean instead of
the forecast. It keeps the budget, `beta_t`, the ensemble, the scenarios, and the costs.

| nofc vs … | CVaR diff | p | Sharpe diff | p |
|---|---|---|---|---|
| `mpc_tailbudget` (forecast on) | +0.00025 | 0.61 | +0.069 | 0.62 |
| `mpc_selfcal` (exposure cap, no budget) | **+0.00185** | **0.001** | +0.174 | 0.17 |
| `equal_weight_voltarget` | −0.0017 | 0.14 | −0.037 | 0.95 |

Removing the forecast changes **nothing** statistically (every p > 0.6) while cutting turnover
4× (0.036 → 0.009) and cost 4× (3.54% → 0.89%). So the forecaster contributes nothing here
except cost — and the one significant effect remains budget-vs-exposure-cap on the tail. But
even with the forecast removed, the MPC stack **still does not significantly beat
`equal_weight_voltarget`** on CVaR (p = 0.14), Sharpe (p = 0.95), or maxDD (p = 0.84).

**Conformal calibration works.** ACI empirical coverage is **0.8984** against a 0.90 target on
this validation window; per-asset `α_t` settles in [0.066, 0.121] around a mean of 0.0925.

**The budget is live, not slack.** A budget that never binds is not a result, so we report the
whole utilization distribution (`CVaR_used / budget` over 983 days): mean **0.869**, median
**0.985**, max **0.995**, and **zero violations**. It binds (utilization ≥ 0.98) on **54.2%**
of days. `vol_mult = 1.6` is calibrated on **train only** from the observed realized
CVaR/vol ratio (~2.0–2.17) so the constraint is live at full investment.

---

## The matched-information test

The strongest result in the project, and the one that settles the most. If the MPC loses
because its budget is denominated in *trailing* volatility, the obvious remedy is a
forward-looking risk signal — so we built one and then **gave it to the baseline too**.

`src/fwdvol.py` is a jump-augmented GARCH(1,1) with Student-*t* innovations, fitted by
maximum likelihood via SciPy (no new dependency). The conditional variance
`h_{t+1} = ω + α·ε_t² + β·h_t` reacts to the newest shock, which a 60-day trailing window
cannot do at full speed.

| comparison (paired bootstrap, 2000 resamples) | ΔCVaR | 95% CI | p |
|---|---|---|---|
| fwd-vol scalar − trailing scalar | **+0.00145** | [0.00087, 0.00195] | **0.001** |
| … same, max drawdown | **+0.01296** | [0.00146, 0.03557] | **0.028** |
| MPC (fwd vol) − fwd-vol scalar | **−0.00395** | [−0.00692, −0.00143] | **0.001** |
| MPC (no fc) − fwd-vol scalar | −0.00313 | [−0.00583, −0.00086] | 0.005 |
| `mpc_tailbudget` − fwd-vol scalar | −0.00338 | [−0.00596, −0.00129] | 0.002 |
| MPC (fwd vol) − trailing scalar | −0.00250 | [−0.00533, +0.00000] | 0.051 *(ns)* |

Both directions are significant and they point **opposite ways**. The forward signal makes
the **simple** rule significantly better, and the MPC consuming that *same signal* is
significantly worse. The MPC **dilutes** a good signal. The best risk controller in this
study is the dumbest one.

> **Do not quote a correlation coefficient for this signal.** The standalone 1-step-ahead
> `corr(fwdvol, |r_{t+1}|)` is not robust: Pearson 0.240/0.184 at fit window 400 but
> 0.091/0.063 at 500, and at the *configured* `lookback: 1000` it cannot be measured on train
> at all (train is 982 days). An earlier draft quoted ".188 vs .170"; that is **retracted**.
> Argue from the decision-level matched-information test only.

---

## Exposure-matched control

A strategy that simply holds less risk always looks better on a downside-tail statistic. That
is not evidence of skill, so we measured realized mean gross exposure from the backtest
weights and rescaled every strategy to a common exposure (`experiments/exposure_matched.py`).

| strategy | exposure | ratio | Sharpe | CVaR | maxDD |
|---|---|---|---|---|---|
| `equal_weight_voltarget_fwd` | 0.686 | 1.00× | 1.299 | **0.0117** | **−9.4%** |
| `equal_weight_voltarget` | 0.767 | 1.12× | **1.302** | **0.0117** | −9.6% |
| `mpc_tailbudget_nofc` | 0.767 | 1.12× | 1.265 | 0.0132 | −13.2% |
| `mpc_tailbudget_fwdvol` | 0.784 | 1.14× | 1.241 | 0.0137 | −13.6% |
| `mpc_tailbudget` | 0.812 | **1.18×** | 1.196 | 0.0127 | −12.5% |

The de-risking explanation is falsified in the direction that matters: the MPC does **not**
hold less risk, it holds up to **1.18× more**, and at matched exposure the baseline is still
better on CVaR, Sharpe, *and* drawdown with indistinguishable volatility. Between 11% and 69%
of the raw CVaR gap was an exposure artifact; the remainder still favours the baseline. The
MPC takes more risk for less return.

*(Rescaling is an analytical normalization, not a tradeable backtest — turnover and cost do
not rescale consistently, so no cost conclusion is drawn here. And `cvar_daily` is a negative
loss magnitude: **more negative is worse**.)*

---

## Volatility-jump regime study

The trailing-vol denominator suggests a regime where the MPC *should* win: a volatility jump
that mean-reverts, where a slow estimator is most exposed. We tested it and it is false.

`experiments/regime_test.py` builds a block-regime fixture (calm → spike → decay, zero
momentum so predictability cannot confound the risk control). **On the fixture built to
favour the MPC, the volatility-targeting rule still wins end to end.**

`experiments/jump_events.py` then enumerates *every* qualifying real event in train+validation
with the rule fixed before any strategy result was inspected: a rolling 20-day volatility rising
≥ 2× the preceding 60-day mean, then falling ≥ 30% from peak within 60 days, de-clustered with
a 140-day cooldown.

| event peak | jump | `voltarget` | `voltarget_fwd` | `mpc_nofc` | `mpc_fwdvol` |
|---|---|---|---|---|---|
| 2013-09-13 | ×2.00 | 0.0173 | 0.0189 | 0.0237 | 0.0246 |
| 2015-09-09 | ×2.00 | 0.0187 | **0.0179** | **0.0175** | 0.0178 |
| 2017-11-08 | ×2.02 | 0.0097 | **0.0091** | 0.0148 | 0.0162 |
| 2018-10-29 | ×2.10 | 0.0165 | **0.0156** | 0.0287 | 0.0298 |

The forward signal helps the scalar rule in **3/4**; the MPC beats its matched scalar in only
**1/4**, with mean CVaR worse by +0.0058 (no fc) and +0.0067 (fwd vol). The largest real
train+val window (2018-07-04 → 2019-05-02, jump ×2.16) is reported separately as an *n*=1
case study, where the scalar rule reaches Sharpe 1.729 / CVaR 0.0134 against
`mpc_tailbudget` 1.342 / 0.0212 — the MPC loses decisively even in the regime built for it.

With 4 events we cannot prove a null; the direction is consistent with the matched-information
result above.

> **Selection trap worth recording.** The largest volatility jump in the *full* sample is
> ×6.94, peaking March 2020 — the COVID crash, in the **sealed test period**. An unbounded
> search would silently select the one event most likely to flatter the method, from data we
> agreed never to touch. The largest jump available in train+validation is only ×2.10. The
> search therefore takes a **hard bound clipped to the validation end**, and regression tests
> fail if that bound is removed (`tests/test_regime_test.py`, `tests/test_jump_events.py`).

> **Correction (2026-10-01).** An earlier version of this study reported "5 events, 4/5 and
> 2/5". Those came from an ad-hoc CSV that no committed script generated. Recomputed with
> `experiments/jump_events.py`: **4 events, 3/4 and 1/4**. Same conclusion, stronger.

---

## Stress-conditional analysis

Conditioning on an **independent** stress label, defined in `docs/stress_definition.md` and
fitted on train only:

```
stress_t  =  (cross-sectional dispersion_t ≥ train p80 = 0.01761)
         OR (equal-weight market return_t ≤ train q10 = −0.01079)
```

60-day warmup; runs separated by ≤2 calm days merged; episodes shorter than 3 days dropped.
This shares **no statistic** with `src/regime.py`'s `VolRegimeDetector` (Jaccard overlap 0.166)
— the first draft of this analysis used that same detector, which made the labels circular and
the whole exercise meaningless. Drawdown-from-running-peak was also tried and **rejected**: it
is sticky, collapsing into one 194-day "episode" across the 2015–16 bear market.

448 stress days (22.8%) in 83 episodes across train+val. In validation: 750 calm, 232 stress.

| strategy | CVaR calm | CVaR stress | worst stress day |
|---|---|---|---|
| **`equal_weight_voltarget`** | −0.008 | **−0.019** | **−0.032** |
| `mpc_naive` | −0.012 | −0.028 | −0.044 |
| `mpc_fc` | −0.011 | −0.029 | −0.046 |
| `mpc_fc_tight` | −0.011 | −0.026 | −0.044 |
| `mpc_selfcal` | −0.011 | −0.026 | −0.044 |
| **`mpc_tailbudget`** | **−0.010** | **−0.023** | **−0.039** |
| `mpc_full` | −0.010 | −0.024 | −0.042 |

Mean gross exposure across all 83 episodes: `equal_weight` 1.000, `mpc_naive` 0.909,
`mpc_selfcal` 0.871, `mpc_tailbudget` 0.84, `mpc_fc_tight` 0.862, `mpc_full` 0.810,
`equal_weight_voltarget` 0.734.
Exposure does **not** dip further at +10d/+20d than at episode start — de-risking here is a
standing posture, not a reaction, because the conformal signal moves slowly.

---

## Self-calibration ablation

`mpc_selfcal` (`src/selfcal.py`) learns the tightening strength instead of fixing it:

```
breach_t   = 1{net_return_t < −0.02}
β_{t+1}    = max(0, β_t + η(breach_t − δ)),    η = 0.05, δ = 0.01
```

It also builds a forecaster ensemble (ridge, historical mean, momentum, GRU if torch is
installed) and measures cross-model disagreement. The tightening signal is the **geometric
mean** of the conformal-width ratio and the disagreement ratio, each divided by its own
trailing median — deliberately conservative, since it moves only when both arms move together.

**The calibration worked; the performance effect did not.**

- realized breach frequency **0.0112** against target `δ = 0.01` (0.0119 over the last 250 days)
- `β_t` mean 0.097, peak 0.322, final 0.249 — engaged and stayed engaged
- mean exposure cap **0.9908** (min 0.850) — almost nothing to tighten
- mean uncertainty ratio **1.0156** (max 1.763) — the geometric mean sits near 1 nearly always
- `corr(β_t, exposure_cap) = −0.489`: β does pull the cap down, just not far
- mean gross exposure 0.826 over validation vs 0.815 for `mpc_fc_tight` — it was not even
  taking less risk, only acting more slowly

`η`, `loss_threshold`, and `δ` were **not** tuned against validation returns after seeing this
null result; doing so would be fitting the evaluation window. The defensible knobs to vary
a priori are `δ` as a stated design target and a stronger response function (arithmetic mean,
or a steeper tightening slope).

---

## What this does not show

Stated plainly, because these are the reviewer questions:

1. **Forecasting does not add alpha here.** Mean per-asset `corr(ŷ, r)` ≈ **−0.035** on this
   window; true daily AR(1) averages ~0.004. Sharpe degrades monotonically as `mu_shrink` rises,
   which is what you expect when the forecast is near-noise. Do not claim the forecaster helps.
2. **A much simpler baseline wins on tails — and the MPC layer still cannot beat it, even
   without the forecaster.** `equal_weight_voltarget` has the best CVaR (−0.0131) and the
   shallowest max drawdown (−10.7%) of every strategy tested, and no MPC variant beats it on
   stress-day CVaR. This is not a forecasting artifact: `mpc_tailbudget_nofc` strips the
   forecast out and the MPC stack is *still* statistically indistinguishable from (and
   point-estimate-better by) the scalar baseline (CVaR p = 0.14, Sharpe p = 0.95, maxDD
   p = 0.84). The MPC ladder improves *within itself*, but not against vol-targeting. A
   reviewer will read `equal_weight_voltarget` and ask why the MPC exists; the current honest
   answer is "it costs more and does not yet do better here."
3. **Tested the "right regime" explanation, and it is also falsified.** I hypothesised the MPC
   loses only because 2016–19 volatility was persistent, and built a vol-jump testbed
   (`experiments/regime_test.py`, `src/data.make_regime_prices`) to check. On the fixture the
   vol-target still wins, and across the real jump+revert events in train+val the MPC beats
   its matched scalar in only **1/4** with a *worse* mean CVaR (+0.0058). See the log for the
   COVID near-miss: the largest jump in the sample is sealed test data, so the search is
   hard-bounded to val_end and a regression test guards that bound.
   **Correction (2026-10-01):** an earlier version of this bullet said "5 events, 2/5, +0.0016".
   Those numbers came from an ad-hoc run that no committed script reproduces. Recomputed with
   `experiments/jump_events.py` (hard-bounded, de-clustered): **4 events, 1/4, +0.0058**.
4. **Why the right regime doesn't help:** the budget is denominated in *trailing* vol
   (`vol_mult · σ_t`), so when vol jumps it is set from stale risk — the same trailing-window
   weakness as the baseline, on the tail instead of the mean. Beating a vol-target on a jump
   needs a genuinely forward-looking risk signal. **We built one and tested it**: a
   jump-augmented GARCH(1,1)-t. At matched information it *significantly helps the scalar rule*
   and *significantly hurts the MPC* (both `p = 0.001`). This is an architectural problem,
   not a tuning problem — the added MPC structure dilutes the good signal.
5. **The MPC's tail advantage is not de-risking.** At matched realized exposure the MPC holds
   up to **1.18× more** risk for less return, and remains worse on CVaR, Sharpe, and drawdown.
   The obvious escape route is closed. See [Exposure-matched control](#exposure-matched-control).
6. **The self-calibration loop is not carrying the result.** The loop is verifiably *active*
   (realized breach frequency 0.0112 against target `δ = 0.01`) but its effect is near-null:
   mean exposure cap 0.991, mean uncertainty ratio 1.016, so there is almost nothing to
   tighten. We did not tune `η`/`δ` against validation returns to fix this, because that would
   fit the evaluation window.
7. **Single market, single window, four years, ten large caps.** No cross-asset or
   cross-market replication.
8. **Sharpe is dominated by seed noise.** Intervals of ±0.9 make this a tail-risk study, not a
   return study.
9. **The test period has never been examined.** None of the above is confirmed out of sample.
   (The vol-jump search is explicitly hard-bounded to `val_end` for exactly this reason.)
10. **The title is a mechanism claim, not a performance claim.** "Self-Calibrating
   Uncertainty-Aware MPC for Tail-Risk Budgeting in Portfolios" is supported as a *mechanism*
   — the budget is real, live (binds 54.2% of days, zero violations), calibrated, and adding it
   improves the tail at `p = 0.001`, replicated under a forecast-ablation control. It is **not**
   a claim that this improves portfolios over a simple baseline. Taken as a superiority claim,
   the title is falsified. Soften it, or gather data where the MPC has room to win — do not
   rescue it by tuning `vol_mult` on validation returns.

### The one surviving claim

`mpc_tailbudget` subclasses `mpc_selfcal`, so the two differ by exactly one constraint. Adding
a hard CVaR(5%) budget improves the tail:

| comparison | ΔCVaR | 95% CI | p |
|---|---|---|---|
| `mpc_tailbudget` − `mpc_selfcal` | **+0.00159** | [0.00101, 0.00214] | **0.001** |
| `mpc_tailbudget_nofc` − `mpc_selfcal` | **+0.00185** | [0.00108, 0.00268] | **0.001** |

MPC-against-MPC, identical tightening machinery, so it is not confounded by exposure level in
the way a cross-family comparison would be. This is the mechanism the title names, and it is
the only robustly significant positive effect in the project.

---

## Bugs found and fixed

Kept because negative results are only trustworthy if the machinery is auditable. Full
timeline with numbers in [`docs/experiment_log.md`](docs/experiment_log.md).

| bug | effect | fix |
|---|---|---|
| `_to_HN` tiled the 1-day forecast flat across all `H` horizon steps | rewarded a 1-day signal `H` times, made the controller ~`H`× bolder than `risk_aversion` implied; turnover 0.94/day (46% cumulative cost); `mpc_fc` returned −19.8% | `mpc.forecast_persistence` — mean decays per step, half-widths stay flat |
| `equal_weight_voltarget` compared a **daily** σ against an **annualized** target | scale inflated ~16×, baseline pinned at its cap and was inert | annualize σ consistently with the target |
| `episode_market_drawdown` anchored each episode's peak at its own first day | forced dd = 0 there, making min-drawdown identically 0 for every episode | anchor the peak at 1.0 *before* episode start |
| bootstrap p-values compared the **centred** distribution | every p-value ≈ 0.97–0.99, i.e. "nothing is significant" always | compare the uncentred replicate distribution against 0 |
| stress labels derived from the controller's own detector | circular — the "16 episodes / 277 days" figure was meaningless | independent train-fitted rule (above) |
| variance-shift test compared post-shift breach rate to the *calm pre-shift* baseline | tested the wrong quantity; the loop controlled the post-shift rate all along | compare post-shift against post-shift: 0.174 unmanaged → 0.010, exactly `δ` |
| GARCH negative log-likelihood sign inverted | optimizer **maximized** the loss and returned its start point | negate correctly; the fit must improve on its grid-search start |
| `_pack` exponentiated `omega` and `nu`, not just `alpha`/`beta` | `nu = 8` became `e^8 ≈ 3000` | only `alpha` and `beta` are log-parameterized |
| GARCH jump term **added** to the variance | adding ~1.0 to a ~1e-3 daily variance pins the forecast at its cap | the jump must **multiply** the shock, keeping units consistent |
| off-by-one in the GARCH forecast recursion | `range(len(e)-1)` never propagated the newest return, so `h_{t+1}` ignored the very shock it must react to — forecast was bit-identical for 0.5σ and 3σ shocks | propagate the newest observation; guarded by a shock-monotonicity test |
| L-BFGS-B walked a good grid point onto the `alpha`/`beta` lower bounds | better objective, degenerate flat model | explicit at-bound guard rejects the candidate |
| a "5 jump events" table was generated by an ad-hoc session | no committed script reproduced it; the claim was unreproducible | `experiments/jump_events.py` regenerates it (4 events); orphan CSV deleted |
| bibliography entries written from memory | 3 guessed arXiv IDs were **physics papers**; 3 volume/page/year sets were wrong | every entry registry-verified via Crossref/arXiv; fields transcribed from the API response |

The last two matter most. The first is a *mechanism* bug that made a signal look plausible
while being worthless; the second is a *reporting* bug that would have put an unreproducible
number in a paper. Both are why the volatility findings above rest on the decision-level
matched-information test rather than on any correlation coefficient.

---

## How the study was built, step by step

The order matters — each step was gated on the previous one.

1. **Splits.** Fixed train 2012–2015 / validation 2016–2019, test sealed from the outset. The
   old 2018–2019 split invalidated every earlier result, which is why the log is explicit that
   pre-resplit numbers are stale.
2. **Strong baselines.** Added `equal_weight_voltarget` and `mpc_dd_riskaversion` *before*
   building the self-calibration layer, so there was an honest floor to measure against. This
   is what exposed the vol-targeting result rather than burying it.
3. **Block diagram.** `experiments/make_block_diagram.py` renders
   `docs/figures/block_diagram.png` from the live config — no hard-coded values — and
   `tests/test_figures.py` asserts no text collisions or off-canvas elements.
4. **Self-calibration** (`src/selfcal.py` + `mpc_selfcal`), with 22 tests, including the
   no-look-ahead harness shared with the backtest.
5. **Tail-risk budget** (`src/mpc.py` `scenario_returns`/`cvar_of_paths` + `mpc_tailbudget`),
   the paper-title mechanism: a hard CVaR(5%) cap via the Rockafellar–Uryasev epigraph that
   keeps the problem a QP. 19 dedicated tests, including budget-respected, budget-binds,
   solver-status, and cash-on-infeasible-budget. The budget needed CLARABEL at tight
   tolerances: at default settings the solver returned `optimal_inaccurate` while violating
   the budget by ~20%.
6. **Exposure confound ruled out** (`experiments/exposure_matched.py`). The escape route for
   every "the MPC wins on tails" claim is that it simply held less risk. Measured and
   falsified in the awkward direction: at matched exposure the MPC holds *more* risk for less
   return. This also forced the `cvar_daily` sign convention to be documented and tested —
   getting it backwards inverts every verdict.
7. **Forward-looking risk signal** (`src/fwdvol.py`, jump-augmented GARCH(1,1)-t) plus the
   **matched-information pair** (`equal_weight_voltarget_fwd` / `mpc_tailbudget_fwdvol`).
   This was the attempt to rescue the MPC, and it produced the sharpest negative result in the
   project. Four parameterization traps are documented in `paper.tex` App. B; two more
   (optimizer walking onto parameter bounds, and the unreproducible event table) are in
   `docs/experiment_log.md`.
8. **Stationary block bootstrap** (`src/bootstrap.py`, Politis & Romano geometric blocks).
   Verified before use: realized mean block length **9.93** against a target of 10, lag-1
   adjacent-index rate 0.886 confirming dependence preservation. Paired differences reuse one
   shared resampled index sequence per replicate.
9. **Independent stress labels** (`src/stress.py`, `docs/stress_definition.md`), train-fitted
   and frozen.
10. **Stress-conditional and per-episode analysis** (`experiments/regime_conditional.py`): all
    83 episodes on shared axes, plus a mean-exposure plot aligned at episode start. Max drawdown
    is deliberately omitted on non-contiguous subsets.
11. **Volatility-jump regime study** (`experiments/regime_test.py` fixture + single real
    window; `experiments/jump_events.py` for the full enumeration). The largest jump in the
    sample is COVID, in the sealed test period, so the event search is hard-bounded to
    `val_end` and the bound is unit-tested in both scripts.
12. **Log everything** in `docs/experiment_log.md`, including the null results, the bugs, and
    the two numbers that had to be **retracted** when a reproducibility audit found no
    committed script could produce them.

---

## Repository layout

```
paper.tex                   IEEE journal manuscript (the title above); no LaTeX in this env
references.bib              17 entries, each verified against Crossref / arXiv / PMLR
configs/default.yaml        every hyperparameter; splits, costs, seeds, ACI, selfcal, fwdvol
src/
  data.py                   download, synthetic fixtures (sim + block-regime), period bounds
  backtest.py               daily engine — no look-ahead, costs, causal observe() hook
  metrics.py                Sharpe, Sortino, maxDD, CVaR 5%, turnover, costs
  baselines.py              equal_weight, buy_hold, markowitz, + both vol-target rules
  estimators.py             shrunk covariance
  forecaster.py             RidgeForecaster, GRUForecaster (torch optional)
  conformal.py              adaptive conformal prediction (Gibbs & Candès ACI)
  mpc.py                    CVXPY controller, tighten_limits, CVaR epigraph (the core)
  regime.py                 VolRegimeDetector — calm/stress "market mood"
  selfcal.py                BetaState, DisagreementEnsemble, uncertainty ratio
  strategies.py             ablation ladder; MPCTailBudgetStrategy = title mechanism
  stress.py                 independent stress labels + episodes
  bootstrap.py              stationary block bootstrap (Politis & Romano)
  fwdvol.py                 jump-augmented GARCH(1,1)-t forward-vol signal
  utils.py                  config, seeding, run dirs
experiments/
  run_all.py                full 15-strategy ladder; guards the test period
  block_bootstrap.py        Step 1 — bootstrap CIs and paired differences
  regime_conditional.py     Steps 2 & 3 — stress/calm metrics, episode exposure
  regime_test.py            vol-jump fixture + single real window, hard-bounded search
  jump_events.py            every real jump+revert event, de-clustered and hard-bounded
  exposure_matched.py       realized-exposure diagnostic + matched-exposure comparison
  make_block_diagram.py     renders the block diagram from live config
tests/                      130 tests; test_backtest.py holds the no-look-ahead harness
docs/
  paper.md                  manuscript notes and the same numbers, in prose
  experiment_log.md         every run, every bug, every null result, every retraction
  stress_definition.md      the independent stress rule, with rejected alternatives
  hypothesis.md             the one falsifiable claim under test
  flow.mmd                  editable Mermaid source for the pipeline diagram
  figures/block_diagram.png generated from the live config
  figures/flow.svg          rendered from flow.mmd  (STALE — regenerate before submission)
```

---

## Reproducing

```bash
source .venv/bin/activate

python -m pytest -q                                  # 130 passed, 1 skipped
python -m experiments.run_all --period val           # 15-strategy ladder -> results/<ts>-val/
python -m experiments.block_bootstrap                 # Step 1
python -m experiments.regime_conditional              # Steps 2 & 3
python -m experiments.jump_events                     # -> results/jump_events.csv
python -m experiments.exposure_matched                # matched-exposure control
python -m experiments.regime_test                     # fixture + single real jump window
python -m experiments.make_block_diagram              # -> docs/figures/block_diagram.png

pdflatex paper.tex && bibtex paper && pdflatex paper.tex && pdflatex paper.tex
```

Every run writes `results/<timestamp>-*/` with its own `config.yaml` snapshot. Outputs are
git-ignored by design; the numbers in this README and in `docs/experiment_log.md` are the
record until you choose otherwise.

**The manuscript is not yet submission-ready.** Three things must be done first:

1. **Generate the placeholder figures.** Fig. 4 (main forest plot) and Fig. 6 (exposure vs
   CVaR) are `\framebox` placeholders in `paper.tex`.
2. **Regenerate `docs/figures/flow.svg`** — it is stale (2025-09-30) and predates the
   forward-volatility and exposure-control workstreams.
3. **Compile the PDF.** There is no LaTeX toolchain in this environment, so `paper.tex` has
   only been validated *structurally*: brace and environment balance, no dangling `\cref`,
   every `\cite` resolving, all 17 bibliography entries cited. It has never been compiled.

**References.** [`references.bib`](references.bib) holds 17 entries, every one verified against
a live registry — Crossref for journal and conference DOIs, the arXiv API for preprints, PMLR
`citation_*` metadata for the L4DC paper — with the author/venue/volume/page/year fields
transcribed from the API response rather than from memory. Verification matters more than it
sounds: while adding the related-work block, **3 guessed arXiv IDs turned out to be physics
papers** and 3 volume/page/year sets were wrong. A guessed DOI is worse than no citation, so
two papers that could not be confirmed (Safronov; Leake & Lodha) are deliberately **not**
cited. The file carries a one-liner to re-run the verification.

The most important entry is **Chee et al. (2024)**, L4DC: it also drives constraint tightening
from conformal uncertainty, in continuous control rather than portfolio allocation. It is the
nearest prior work and a reviewer will ask why it is not a baseline. The manuscript positions
it explicitly in Related Work.

---

## Project rules

[`AGENTS.md`](AGENTS.md) is the binding brief. The non-negotiables:

1. **No look-ahead.** Everything at day `d` uses `returns[:d]` only. Every new feature gets a
   no-look-ahead test.
2. **The test period is untouchable.** Never run `--period test` without an explicit final
   instruction. Never edit the guard.
3. **Transaction costs on** in every backtest, baselines included.
4. **Reproducible.** Fixed seeds; every run snapshots its config.
5. **Fair comparison.** All strategies share data, period, costs, and lookback.
6. **`src/backtest.py`, the cost model, and split logic are reviewed by hand.** Don't touch
   without asking.
7. **If a result looks too good** (Sharpe > 2 net of costs on real data), assume a bug and hunt
   for leakage first.
8. **Never report a number you did not just compute.** No cherry-picking seeds or periods.
