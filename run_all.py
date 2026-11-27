#!/usr/bin/env python3
"""Run every strategy YAML through both engines, then gate on cross-engine parity.

Results are written with the configured funding model (quant_lab) / without it (vectorbt cannot model funding).
The parity check re-runs both engines with funding disabled so the comparison is apples to apples.
"""
import argparse
import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runners"))
from strategy_engine import file_sha256, load_config  # noqa: E402
import quant_lab_runner  # noqa: E402
import vectorbt_runner  # noqa: E402

EQUITY_REL_TOL = 2e-3
DD_ABS_TOL = 5e-3


def pct(x):
    return "n/a" if x is None else f"{x * 100:8.2f}%"


def main():
    p = argparse.ArgumentParser(description="Run every strategy YAML through both engines")
    p.add_argument("--strategy-dir", default="strategies")
    p.add_argument("--output-root", default=None)
    p.add_argument("--no-parity", action="store_true", help="skip the cross-engine parity gate")
    a = p.parse_args()
    failed = False
    for path in sorted((ROOT / a.strategy_dir).glob("*.yaml")):
        cfg, sha = load_config(path), file_sha256(path)
        q, _, _ = quant_lab_runner.run(cfg, a.output_root, config_sha256=sha)
        v, _, _ = vectorbt_runner.run(cfg, a.output_root, config_sha256=sha)
        print(f"\n== {cfg['strategy_id']}  ({q['start'][:10]} -> {q['end'][:10]}, funding={q['funding_mode']}, "
              f"coverage={pct(q['funding_coverage'])})")
        print(f"   quant_lab  ret {pct(q['net_return'])}  maxDD {pct(q['max_drawdown'])}  trades {q['trades']:4d}"
              f"  fees ${q['total_fees']:.0f}  funding ${q['total_funding']:.0f}  B&H {pct(q['buy_and_hold_return'])}")
        print(f"   vectorbt   ret {pct(v['net_return'])}  maxDD {pct(v['max_drawdown'])}  trades {v['trades']:4d}"
              f"  (funding not modeled)")
        for tag in ("in_sample", "out_of_sample"):
            b = q[tag]
            if b:
                print(f"   {tag:13s} ret {pct(b['net_return'])}  maxDD {pct(b['max_drawdown'])}  "
                      f"sharpe {b['sharpe']}  B&H {pct(b['buy_and_hold_return'])}")
        if a.no_parity:
            continue
        nf = copy.deepcopy(cfg)
        nf["costs"]["funding"]["mode"] = "none"
        qa, _, _ = quant_lab_runner.run(nf, write=False)
        va, _, _ = vectorbt_runner.run(nf, write=False)
        rel = abs(qa["final_equity"] - va["final_equity"]) / qa["final_equity"] if qa["final_equity"] else 0.0
        ok = (qa["trades"] == va["trades"] and rel <= EQUITY_REL_TOL
              and abs(qa["max_drawdown"] - va["max_drawdown"]) <= DD_ABS_TOL)
        print(f"   parity (funding off): trades {qa['trades']} vs {va['trades']}, equity diff {rel * 100:.3f}% -> "
              f"{'OK' if ok else 'MISMATCH'}")
        failed |= not ok
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
