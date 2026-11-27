#!/usr/bin/env python3
"""Cross-check runner: replays the SAME execution plan through VectorBT.

Differences from quant_lab by design: VectorBT does not model funding (reported as not_modeled), and
exposure > 1 (leverage) is not supported here. Everything else (fills, costs, flat windows, final close)
comes from the shared plan, so any disagreement is a real discrepancy.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_engine import (DATA_DIR, TF_MINUTES, TRADE_COLS, file_sha256, load_config, prepare,  # noqa: E402
                             save_outputs, summarize)

ENGINE = "vectorbt"


def run(cfg, output_root=None, data_dir=DATA_DIR, write=True, config_sha256=None):
    import vectorbt as vbt

    expo = cfg["risk"]["equity_fraction"] * cfg["execution"]["leverage"]
    if expo > 1:
        raise NotImplementedError("vectorbt cross-check supports equity_fraction * leverage <= 1")
    plan, _ = prepare(cfg, data_dir)
    minutes = TF_MINUTES[cfg["data"]["timeframe"]]
    bar = pd.Timedelta(minutes=minutes)
    bidx = plan.index.append(pd.DatetimeIndex([plan.index[-1] + bar]))
    price = pd.Series(np.append(plan["open"].to_numpy(float), plan["close"].iloc[-1]), index=bidx)
    pos = np.append(plan["exec_pos"].to_numpy(int), 0)      # final row: flat -> closes anything still open
    prev = np.r_[0, pos[:-1]]
    sig = lambda m: pd.Series(m, index=bidx)  # noqa: E731
    kwargs = dict(init_cash=cfg["risk"]["initial_equity"], fees=cfg["costs"]["fee_per_side"],
                  slippage=cfg["costs"]["slippage_per_side"], freq=f"{minutes}min", upon_opposite_entry="close")
    if expo < 1:
        kwargs.update(size=expo, size_type="valuepercent")
    pf = vbt.Portfolio.from_signals(
        price, sig((pos == 1) & (prev != 1)), sig((pos != 1) & (prev == 1)),
        short_entries=sig((pos == -1) & (prev != -1)), short_exits=sig((pos != -1) & (prev == -1)), **kwargs)

    rec = pf.trades.records_readable
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(rec["Entry Timestamp"]).map(lambda t: t.isoformat()),
        "exit_time": pd.to_datetime(rec["Exit Timestamp"]).map(lambda t: t.isoformat()),
        "side": rec["Direction"].str.lower(),
        "entry_price": rec["Avg Entry Price"], "exit_price": rec["Avg Exit Price"],
        "notional": rec["Size"] * rec["Avg Entry Price"],
        "fees": rec["Entry Fees"] + rec["Exit Fees"], "funding": 0.0,
        "pnl": rec["PnL"], "return": rec["Return"],
        "bars_held": ((pd.to_datetime(rec["Exit Timestamp"]) - pd.to_datetime(rec["Entry Timestamp"])) / bar).round().astype(int),
        "exit_reason": np.where(pd.to_datetime(rec["Exit Timestamp"]) == bidx[-1], "end_of_data", "signal"),
    }).reindex(columns=TRADE_COLS)
    eq = pf.value().to_numpy(float)
    metrics = summarize(cfg, bidx, price.to_numpy(float), eq, pos, trades)
    metrics.update(engine=ENGINE, funding_modeled=False, funding_mode="not_modeled", funding_coverage=None,
                   engine_assumptions="VectorBT from_signals replaying the shared execution plan; funding not modeled")
    if config_sha256:
        metrics["config_sha256"] = config_sha256
    equity = pd.DataFrame({"timestamp": [t.isoformat() for t in bidx], "equity": eq, "position": pos})
    if write:
        root = Path(output_root) if output_root else Path(__file__).resolve().parents[1] / "results"
        save_outputs(root / cfg["strategy_id"] / ENGINE, metrics, trades, equity)
    return metrics, trades, equity


def main():
    p = argparse.ArgumentParser(description="Run a YAML strategy through VectorBT (cross-check)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--output-root", default=None)
    a = p.parse_args()
    cfg = load_config(a.strategy)
    metrics, _, _ = run(cfg, a.output_root, config_sha256=file_sha256(a.strategy))
    print(metrics)


if __name__ == "__main__":
    main()
