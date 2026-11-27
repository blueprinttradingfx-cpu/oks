#!/usr/bin/env python3
"""Reference runner: the engine's own marked-to-market simulator (models funding)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_engine import DATA_DIR, file_sha256, load_config, prepare, save_outputs, simulate  # noqa: E402

ENGINE = "quant_lab"


def run(cfg, output_root=None, data_dir=DATA_DIR, write=True, config_sha256=None):
    plan, funding = prepare(cfg, data_dir)
    metrics, trades, equity = simulate(plan, cfg, funding)
    metrics["engine"] = ENGINE
    metrics["funding_modeled"] = cfg["costs"]["funding"]["mode"] != "none"
    if config_sha256:
        metrics["config_sha256"] = config_sha256
    if write:
        root = Path(output_root) if output_root else Path(__file__).resolve().parents[1] / "results"
        save_outputs(root / cfg["strategy_id"] / ENGINE, metrics, trades, equity)
    return metrics, trades, equity


def main():
    p = argparse.ArgumentParser(description="Run a YAML strategy with the built-in marked-to-market simulator")
    p.add_argument("--strategy", required=True)
    p.add_argument("--output-root", default=None)
    a = p.parse_args()
    cfg = load_config(a.strategy)
    metrics, _, _ = run(cfg, a.output_root, config_sha256=file_sha256(a.strategy))
    print(metrics)


if __name__ == "__main__":
    main()
