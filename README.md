# conformal-mpc-portfolio
See AGENTS.md for the project brief and rules, and BUILD_GUIDE.md for the step-by-step build.

    make setup && source .venv/bin/activate
    make test
    make synthetic     # offline smoke test on SIMULATED data - never reportable
    make val           # real NSE data (10 tickers), validation period 2018-2019

Data: real adjusted-close prices for 10 NSE large caps, 2012-01-02 to 2025-12-30,
cached under `data/raw/`. The simulated generator in `src/data.py` is an offline test
fixture only. Every hyperparameter lives in `configs/default.yaml`; no magic numbers
in code.
