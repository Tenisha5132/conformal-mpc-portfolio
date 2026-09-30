"""Smoke tests for the paper figures, so they cannot drift from the config/code."""
import pathlib
import re

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


# --------------------------------------------------------- mermaid flow diagram

FLOW = pathlib.Path("docs/flow.mmd")


def _flow_text():
    return FLOW.read_text() if FLOW.exists() else ""


def test_flow_mmd_exists_and_declares_a_diagram():
    assert FLOW.exists(), "docs/flow.mmd is missing"
    text = _flow_text()
    assert text.lstrip().startswith("%%"), "flow.mmd should open with a comment header"
    assert "flowchart TD" in text.splitlines(), "expected a top-level `flowchart TD` declaration"


def test_flow_mmd_has_no_duplicate_diagram_declarations():
    """A second `flowchart TD` silently breaks the Mermaid parser."""
    assert _flow_text().count("flowchart TD") == 1


def test_flow_mmd_documents_the_no_lookahead_boundary():
    """The one arrow that crosses back from day t+1 must be drawn as feedback only."""
    text = _flow_text()
    assert "returns[:d]" in text
    assert "realised" in text.lower()
    # the feedback path must be dashed (.-), never a solid data edge
    assert text.count("-. ") + text.count("-.->") >= 1, "no dashed feedback edge found"


def test_flow_mmd_marks_the_core_contribution():
    text = _flow_text()
    assert "tighten_limits" in text
    assert "exposure_cap" in text
    assert "core" in text.lower(), "the core block should be styled as such"


def test_flow_mmd_node_ids_match_the_rendered_block_diagram():
    """The .mmd is the textual source of truth; the PNG is generated. They must not drift.

    We check that every stage named in the rendered matplotlib diagram also appears
    as a node in the mermaid file.
    """
    cfg = load_config(CFG)
    joined = "\n".join(_rendered_text(cfg))
    text = _flow_text()
    pairs = {
        "MARKET": "RET",
        "MEASUREMENT": "HIST",
        "FORECASTER": "FOR",
        "ADAPTIVE CONFORMAL": "ACI",
        "RISK-LIMIT SCHEDULER": "TIGHT",
        "MPC OPTIMIZER": "QP",
        "ACTUATOR": "TURN",
    }
    for block, node in pairs.items():
        assert block in joined, f"rendered diagram lost the {block} block"
        assert re.search(rf'^\s*{node}\[', text, re.M), \
            f"flow.mmd has no node id {node} corresponding to the {block} block"


def test_flow_svg_is_committed_and_nontrivial():
    svg = pathlib.Path("docs/figures/flow.svg")
    assert svg.exists(), "docs/figures/flow.svg is missing; re-render docs/flow.mmd"
    text = svg.read_text(errors="ignore")
    assert text.lstrip().startswith("<?xml") or "<svg" in text[:400]
    assert len(text) > 5000, "rendered flow diagram looks truncated"
