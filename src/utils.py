"""Small helpers: config loading, seeding, run folders."""
import random
import time
from pathlib import Path

import numpy as np
import yaml


def load_config(path="configs/default.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def make_run_dir(base="results", tag="run") -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = Path(base) / f"{stamp}-{tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_config(cfg, run_dir: Path):
    with open(Path(run_dir) / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)
