"""Smoke tests for the paper figures, so they cannot drift from the config/code."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from experiments.make_block_diagram import build
from src.utils import load_config

CFG = "configs/default.yaml"


def _rendered_text(cfg):
    """Build the diagram in memory and return all the text it drew."""
    holder = {}
    original = plt.Figure.savefig
    plt.Figure.savefig = lambda self, *a, **k: holder.setdefault("fig", self)
    try:
        build(cfg, __import__("pathlib").Path("/tmp/_unused.png"))
    finally:
        plt.Figure.savefig = original
    fig = holder["fig"]
    FigureCanvasAgg(fig).draw()
    texts = [t.get_text() for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    plt.close(fig)
    return texts


def test_block_diagram_renders():
    cfg = load_config(CFG)
    texts = _rendered_text(cfg)
    joined = "\n".join(texts)
    # every controller block must be present
    for required in ["MARKET", "MEASUREMENT", "FORECASTER", "ADAPTIVE CONFORMAL",
                     "RISK-LIMIT SCHEDULER", "MPC OPTIMIZER", "ACTUATOR"]:
        assert required in joined, f"block diagram is missing the {required} block"
    # the core contribution is explicitly marked
    assert "CORE CONTRIBUTION" in joined


def test_block_diagram_reflects_config():
    """Figure must show live config values, not hard-coded ones."""
    cfg = load_config(CFG)
    cfg["backtest"]["cost_bps"] = 25.0
    cfg["mpc"]["horizon"] = 9
    joined = "\n".join(_rendered_text(cfg))
    assert "25 bps" in joined, "cost in diagram did not follow the config"
    assert "H = 9 steps" in joined, "horizon in diagram did not follow the config"


def test_block_diagram_has_no_text_collisions():
    """Guard the layout: no overlapping labels, nothing off-canvas."""
    from pathlib import Path
    holder = {}
    original = plt.Figure.savefig
    plt.Figure.savefig = lambda self, *a, **k: holder.setdefault("fig", self)
    try:
        build(load_config(CFG), Path("/tmp/_unused.png"))
    finally:
        plt.Figure.savefig = original
    fig = holder["fig"]
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    r = canvas.get_renderer()
    items = [(t.get_text(), t.get_window_extent(r))
             for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
    W, H = fig.get_size_inches() * fig.dpi
    for txt, bb in items:
        assert bb.x0 >= 2 and bb.x1 <= W - 2 and bb.y0 >= 2 and bb.y1 <= H - 2, \
            f"text falls off the canvas: {txt[:40]!r}"
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a, b = items[i][1], items[j][1]
            ix = max(0, min(a.x1, b.x1) - max(a.x0, b.x0))
            iy = max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
            assert not (ix > 3 and iy > 3), \
                f"overlapping labels: {items[i][0][:30]!r} <> {items[j][0][:30]!r}"
    plt.close(fig)
