"""Test utilities: synthetic frames and a minimal valid config (new schema)."""
import copy
import gzip
import numpy as np
import pandas as pd

BASE_CFG = {
    "strategy_id": "t",
    "name": "test",
    "data": {"instrument": "BTC-USDT-SWAP", "timeframe": "15m", "higher_timeframe": "1H",
             "start": "2026-01-01T00:00:00Z", "end": "2026-12-31T00:00:00Z"},
    "signals": {"type": "ema_trend", "fast_ema": 2, "slow_ema": 3,
                "higher_timeframe": {"fast_ema": 2, "slow_ema": 3, "require_alignment": True}},
    "execution": {"signal_execution": "next_bar_open", "position_mode": "long_short",
                  "leverage": 1, "pyramiding": False,
                  "flat_before_funding": {"enabled": False, "bars_before": 1,
                                          "funding_hours_utc": [0, 8, 16]}},
    "risk": {"position_sizing": "equity_fraction", "equity_fraction": 1.0, "initial_equity": 10000.0},
    "costs": {"fee_per_side": 0.0, "slippage_per_side": 0.0,
              "funding": {"mode": "none", "fallback_rate_per_8h": 0.0}},
    "validation": {"data_source": "local_okx_cache", "out_of_sample_fraction": 0.3},
}


def cfg(**sections):
    """Deep-copy BASE_CFG and overlay {section: {key: value}} (one level deep, funding/flat handled by caller)."""
    c = copy.deepcopy(BASE_CFG)
    for sec, vals in sections.items():
        if isinstance(vals, dict):
            c[sec].update(vals)
        else:
            c[sec] = vals
    return c


def frame(opens, closes=None, target=None, start="2026-01-01 00:00", freq="15min"):
    """Synthetic 15m frame. target[i] is decided at the close of bar i."""
    n = len(opens)
    idx = pd.date_range(start, periods=n, freq=freq, tz="UTC")
    closes = list(opens) if closes is None else list(closes)
    df = pd.DataFrame({"open": np.asarray(opens, float), "high": np.maximum(opens, closes) * 1.0,
                       "low": np.minimum(opens, closes) * 1.0, "close": np.asarray(closes, float)}, index=idx)
    df["target"] = 0 if target is None else target
    return df


def write_candles(path, rows):
    """rows: list of (iso_ts, o, h, l, c, confirm) -> OKX-style csv.gz."""
    lines = ["ts_ms,timestamp_utc,open,high,low,close,vol_contracts,vol_base,vol_quote,confirm"]
    for ts, o, h, l, c, conf in rows:
        ms = int(pd.Timestamp(ts).value // 1_000_000)
        lines.append(f"{ms},{ts},{o},{h},{l},{c},1,1,1,{conf}")
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
