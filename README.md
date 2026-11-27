# BTC-USDT Strategy Lab (v2)

Config-driven research package for BTC-USDT perpetual-swap backtests on locally cached OKX data. One shared
execution plan feeds two engines (a built-in marked-to-market simulator and VectorBT), so they cannot drift apart.
See `AUDIT.md` for what changed from v1 and why.

## Layout
- `strategy_engine.py` – config schema (strict), data layer, signals, execution plan, simulator, metrics.
- `runners/quant_lab_runner.py` – reference engine (models funding).
- `runners/vectorbt_runner.py` – cross-check: replays the same plan through VectorBT (no funding).
- `run_all.py` – runs every `strategies/*.yaml` through both engines, then gates on parity.
- `tests/` – pytest suite (`pytest` from this directory). Warnings are errors.
- `data/` – cached OKX 5m / 15m / 1H candles and partial funding history (Aug 26 → Nov 27 only).
- `results/<strategy_id>/<engine>/` – `metrics.json`, `trades.csv`, `equity_curve.csv` (marked to market).
- `results/_archive_pre_refactor/` – v1 outputs, kept only for comparison.

## Run
```bash
pip install -r requirements.txt
pytest                      # 46 tests
python run_all.py           # both engines + parity gate (exit 1 on mismatch)
python runners/quant_lab_runner.py --strategy strategies/ema_trend_15m_1h.yaml
```

## Execution conventions
Signal from bar *i*'s close fills at bar *i+1*'s open. Fills pay `slippage_per_side` against you plus
`fee_per_side` on fill notional. Size is fixed at entry (`equity × equity_fraction × leverage`). Equity is marked
to market every bar. An open position is closed at the last close. Funding settles at 00/08/16 UTC: a position pays
if it was held over the bar that ends at the settlement; long pays when the rate is positive. The full list is at
the top of `strategy_engine.py`.

## Config notes
- Unknown, legacy or unsupported keys raise `ConfigError` (nothing is silently ignored).
- `execution.flat_before_funding` is **off** by default; enabling it adds up to 3 round trips per day.
- `costs.funding.mode`: `none` | `cached_when_available` | `cached_with_fallback` (uses `fallback_rate_per_8h`
  where the cache has no data). `metrics.json` reports `funding_coverage`.
- `validation.out_of_sample_fraction` produces `in_sample` / `out_of_sample` blocks. Parameters are not tuned
  on the first block, so treat it as a stability check, not proof of an edge.
- `leverage` only scales size; liquidation prices and margin are not modelled (a trade that loses all equity is
  floored at zero). The VectorBT cross-check supports `equity_fraction × leverage ≤ 1`.

## Adding a strategy
Copy the YAML, change `strategy_id` and parameters, run `python run_all.py`. A new signal type needs code in
`build_signals` plus a schema entry; the YAML stays declarative.

## Caveats
Hypothetical backtests on one instrument and one year. Funding cache covers ~28% of settlements. Next-open fills
are taker orders. Re-test fees, slippage and parameters out of sample before any paper or live use.
