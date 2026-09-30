.PHONY: setup test val synthetic final

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

test:
	python -m pytest -q

# Offline smoke test on SIMULATED data. Fast, no network.
# NEVER report these numbers in the paper - use `make val` (real NSE data).
synthetic:
	python -m experiments.run_all --synthetic --period val --strategies equal_weight buy_hold markowitz mpc_naive

# Real NSE data, validation period 2018-2019. This is where results get reported from.
val:
	python -m experiments.run_all --period val

# Test period: run ONLY when everything is frozen. Never tune on this.
final:
	python -m experiments.run_all --period test --final
