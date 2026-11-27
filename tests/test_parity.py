"""Cross-engine contract: quant_lab and VectorBT must agree on the same execution plan."""
import pytest
import strategy_engine as eng

vbt_mod = pytest.importorskip("vectorbt")
import quant_lab_runner
import vectorbt_runner
from pathlib import Path

STRATEGY = Path(__file__).resolve().parents[1] / "strategies" / "ema_trend_15m_1h.yaml"


@pytest.mark.parametrize("flat_window", [False, True])
def test_engines_agree_on_trades_and_final_equity(flat_window):
    """Break caught: the two engines silently implementing different rules (the original -38% vs -18% gap)."""
    c = eng.load_config(STRATEGY)
    c["data"]["end"] = "2026-04-01T00:00:00Z"
    c["costs"]["funding"] = {"mode": "none", "fallback_rate_per_8h": 0.0}   # VectorBT does not model funding
    c["execution"]["flat_before_funding"]["enabled"] = flat_window
    a, ta, _ = quant_lab_runner.run(c, write=False)
    b, tb, _ = vectorbt_runner.run(c, write=False)
    assert a["trades"] == b["trades"] > 20
    assert a["final_equity"] == pytest.approx(b["final_equity"], rel=2e-3)
    assert a["max_drawdown"] == pytest.approx(b["max_drawdown"], abs=5e-3)
