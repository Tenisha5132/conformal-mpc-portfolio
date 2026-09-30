# Project: Conformal-Calibrated Robust MPC for Risk-Aware Portfolio Allocation

## Goal
Portfolio controller: learned return forecaster -> adaptive conformal intervals -> MPC (CVXPY) whose risk limits
tighten automatically when forecast uncertainty spikes. Target: a publishable paper. Rigor matters more than returns.

## Contributions (in priority order)
1. CORE: uncertainty-driven constraint tightening (src/mpc.py `tighten_limits`, src/strategies.py)
2. #7: adaptive conformal intervals (src/conformal.py) — Gibbs & Candes ACI, online, per asset
3. #1: regime-aware "market mood" MPC (src/regime.py) — calm vs stress parameters
4. (later, NOT now) #2 decision-focused learning. Do not implement unless explicitly asked.

## Stack / commands
- Python 3.10+, numpy, pandas, scipy, cvxpy, torch (optional GRU), pytest. Activate venv: `source .venv/bin/activate`
- Tests: `python -m pytest -q`        Offline smoke test: `make synthetic` (SIMULATED - never reportable)
- Validation run: `make val`          Final test run: `make final` (see rules)
- All hyperparameters live in configs/default.yaml. No magic numbers in code.

## Layout
src/data.py (download, synthetic, splits) | src/backtest.py (engine) | src/metrics.py
src/baselines.py | src/estimators.py | src/mpc.py | src/forecaster.py (ridge + GRU)
src/conformal.py | src/regime.py | src/strategies.py (ablation ladder)
experiments/run_all.py | tests/

## Strategy ladder (ablation)
equal_weight, buy_hold, markowitz, equal_weight_voltarget, mpc_dd_riskaversion, mpc_naive, mpc_fc,
mpc_fc_robust, mpc_fc_tight (core), mpc_selfcal (learned beta), mpc_tailbudget (CVaR budget,
the paper-title mechanism), mpc_tailbudget_nofc (same, forecast removed = isolating control),
mpc_full (+regime)
Source of truth is `ALL` in experiments/run_all.py.

## Figures
- `python -m experiments.make_block_diagram` -> `docs/figures/block_diagram.png`. Rendered from the
  live config (no hard-coded values) and covered by tests/test_figures.py, which also asserts no
  text collisions. Regenerate after any config change.
- `python -m experiments.run_all ...` -> `results/<timestamp>-*/equity_drawdown.png`

## Metrics
Sharpe, Sortino, max drawdown, annualized return/vol, CVaR 5%, turnover, total costs, plus conformal empirical coverage.

## HARD RULES (do not violate, do not "fix" by relaxing)
1. NO LOOK-AHEAD. Everything at day d uses returns[:d] only. tests/test_backtest.py::check_no_lookahead must pass
   for every strategy you add. Add a no-lookahead test for every new feature/model.
2. TEST PERIOD IS UNTOUCHABLE. Tune hyperparameters on the validation period only. Never run `--period test`
   without an explicit instruction from the user containing the word "final". Never edit the guard in run_all.py.
3. TRANSACTION COSTS ON in every backtest, baselines included (backtest.cost_bps).
4. REPRODUCIBLE: fixed seeds; every run saves its config to results/<timestamp>-*/config.yaml.
5. FAIR COMPARISON: all strategies share the same data, period, costs, lookback.
6. Do not change src/backtest.py, the cost model, or split logic unless the user asks. These are finance-critical
   and reviewed by hand.
7. If a result looks too good (Sharpe > 2 net of costs on real data), assume a bug and look for leakage first.
8. Never report numbers you did not just compute. No cherry-picking seeds/periods.

## Findings so far (validation period only - test never touched)
- Daily ridge forecasts on lagged returns have NO skill on these NSE names:
  mean per-asset corr(pred, realized) ~= -0.035 on 2018-2019; true daily AR(1) averages ~0.004.
  Sharpe degrades monotonically in `mu_shrink`, which confirms the forecast is near-noise.
  Do not claim the forecaster adds alpha. The defensible claim is that the conformal +
  tightening + regime layer makes an unreliable forecaster SAFE to put in the loop.
- BUG (fixed): `_to_HN` used to tile the 1-day forecast flat across all H steps, rewarding a
  1-day signal H times and making the controller ~H times bolder than `risk_aversion` implied.
  Turnover hit 0.94/day (46% cumulative cost) and mpc_fc returned -19.8% on real val.
  Fixed with `mpc.forecast_persistence` (mean decays per step; half-widths stay flat).
- TRAP: large `risk_aversion` (>=20) parks realized exposure at 0.3-0.6. Sharpe then looks good
  only because vol is near zero, AND every constraint goes slack, so the core tightening
  mechanism becomes inert. Prefer operating points where the controller runs against its limits.
- `kappa` must stay tiny: daily interval half-widths (~0.028) dwarf daily mean returns (~0.001),
  so a large kappa makes worst-case mu negative for every asset and the robust MPC holds cash.
- Conformal ACI behaves: empirical coverage 0.898 vs 0.90 target, per-asset alpha_t in 0.069-0.119.
- mpc_selfcal (src/selfcal.py): learned beta_t = max(0, beta + eta*(breach - delta)) driving the SAME
  tighten_limits call, plus geometric mean of conformal-width and ensemble-disagreement ratios.
  Calibration WORKS (realized breach freq 0.0112 vs delta=0.01) but the PERFORMANCE effect is null:
  mean exposure cap 0.9908, mean uncertainty ratio 1.0156 -> almost nothing to tighten. The
  geometric mean is too conservative (needs BOTH arms to move) and delta=0.01 starves the loop of
  feedback (~11 breaches in 983d). Do NOT tune eta/loss_threshold/delta against val returns to fix
  this; see docs/experiment_log.md. A weaker baseline (equal_weight_voltarget) beats it on tails.
- mpc_tailbudget (src/mpc.py `scenario_returns`/`cvar_of_paths`, src/strategies.py
  MPCTailBudgetStrategy): HARD daily CVaR(5%) budget via the Rockafellar-Uryasev epigraph over a
  FIXED 200-scenario Gaussian set. Stays a QP (fixed shocks => loss is linear in w) and uses
  CLARABEL with tight tolerances, because the default solver returns `optimal_inaccurate` and
  VIOLATES the budget by ~20%. budget = vol_mult * sigma_t * 1/(1+beta_t*(ratio_t-1)); vol_mult=1.6
  is TRAIN-calibrated (realized CVaR/vol ~2.0-2.17) so the constraint is LIVE, not slack.
  Beats mpc_selfcal on CVaR by 0.0016 (paired bootstrap p=0.001) - the one-line ablation.
  Does NOT beat equal_weight_voltarget (p=0.082). docs/hypothesis.md is still falsified.
  TRAPS: (a) loss = -return, so the CVaR tail is the UPPER end - `quantile(alpha)` measures the
  BEST-case tail and the budget never binds; (b) the R-U term is 1/(a(1-a)*S)*sum(u) - omitting
  1/S inflates it S-fold and yields over-budget solutions; (c) an infeasible budget must go to
  CASH, never back to w0 (which is fully invested = the opposite of risk-off).
- FINDING (2026-10-01, the decisive ablation): `mpc_tailbudget_nofc` (budget ON, forecast OFF)
  has IDENTICAL risk metrics to `mpc_tailbudget` (CVaR p=0.61, Sharpe p=0.62, maxDD p=0.87) but
  4x lower turnover/cost. So the forecaster is a pure drag here. AND the nofc stack still does
  NOT beat equal_weight_voltarget (CVaR p=0.14, Sharpe p=0.95). The ONE robust significant effect
  is budget-vs-exposure-cap on CVaR (p=0.001, vs both mpc_selfcal and nofc). Title is a MECHANISM
  claim only; as a superiority claim it is FALSIFIED. See docs/experiment_log.md.
- DATA POLICY: reported numbers come from real NSE prices only (10 tickers, cached in data/raw).
  The simulated generator (cfg['synthetic']) exists solely so the test suite runs offline; every
  one of its parameters is in the config, and the runner prints a warning banner on synthetic runs.
  tests/common.py picks the real cache automatically and records the source on each frame.

## Working style
- One milestone per session. Read this file, run the tests first, make the change, run the tests again.
- Keep code small and typed where helpful; add a test for every new behavior.
- Ask before adding dependencies.
