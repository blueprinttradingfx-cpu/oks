# Audit: v1 → v2

## Headline
The shipped v1 result (quant_lab −38.0%, 501 trades) was mostly an artifact of a force-flat rule that only one
engine implemented. The same strategy under one consistent rule set is **−18.2%** (319 trades) with both engines
agreeing to 0.014% on final equity. The signal has a gross edge (+27.8% at zero cost vs buy-and-hold −30.9%) that
costs consume at the default 5 bp fee + 2 bp slippage.

## Findings and fixes
| # | v1 problem | Where | v2 |
|---|---|---|---|
| 1 | `force_flat_at_utc` exited at the 00:00 open and re-entered at 00:15: 189 of 501 trades were artificial. It did not avoid funding (exit lands on the settlement timestamp; 08:00/16:00 never covered). | engine `simulate` | Replaced by `flat_before_funding` (exits the bar *before* each settlement), tested, **off** by default |
| 2 | VectorBT ignored the rule → −38% vs −18% "disagreement" was a config mismatch | vectorbt runner | Both runners replay one shared plan; parity is a tested gate |
| 3 | Config keys never read: `force_flat_before_funding`, `funding`, `leverage`, `pyramiding`, `position_mode`, `out_of_sample_fraction`, `require_alignment` | engine | All implemented or rejected; unknown keys raise `ConfigError` |
| 4 | Equity only moved at exits; max drawdown understated | engine | Marked-to-market every bar |
| 5 | Funding never applied | engine | Modelled with explicit settlement convention + coverage reporting |
| 6 | VectorBT price series `open.shift(-1)`: trades labelled one bar early, equity off by a bar; `freq` hardcoded | vectorbt runner | Fill-bar timestamps; freq from config |
| 7 | Last bar liquidated at close with fees on entry equity; profit factor 0 when no losses | engine | Explicit `end_of_data` exit; fees on fill notional; `null` when undefined |
| 8 | Hardcoded `_2026` data path; unconfirmed candles not filtered; no duplicate check | data layer | Glob by instrument/timeframe; `confirm==1` only; duplicates raise |
| 9 | Manifest pointed at `/home/ubuntu/...`; no Sharpe/exposure/benchmark; no tests | misc | Relative paths; richer metrics; 46 tests (13 mutations checked) |

## Verified as correct in v1 (unchanged behaviour)
Data has no gaps, duplicates or bad OHLC. Higher-timeframe bias uses completed candles only (no lookahead).
Signals fill at the next open.

## v2 results (shipped config, 2026-01-01 → 2026-11-27)
| Run | Net | Max DD (MTM) | Sharpe | Trades |
|---|---|---|---|---|
| quant_lab, funding as cached | −18.21% | −43.24% | −0.49 | 319 |
| vectorbt (no funding) | −18.42% | −43.30% | – | 319 |
| zero costs (gross) | +27.83% | −23.81% | 0.98 | 319 |
| fee 2 bp + slip 1 bp | +5.57% | −32.11% | 0.35 | 319 |
| fee 5 bp + slip 5 bp | −32.48% | −51.92% | −1.11 | 319 |
| `flat_before_funding` on | −61.36% | −71.76% | −2.84 | 820 |

Funding over the 28% of settlements we have data for: +$19 (≈ +0.2 pt). Avoiding it is never worth the turnover.
In-sample (first 70%) −18.8% vs buy-and-hold −20.6%; out-of-sample (last 30%) +0.8% vs −13.0%. Single untuned
config, so this is a stability read only.

## Open items (not done in v2)
- Turnover reduction: minimum hold, EMA-gap/ATR band, limit orders. All 16 EMA pairs tried in the audit were
  gross-positive and net-negative, and slower signals did better, so cost per trade is the lever.
- Funding history before Aug 26 is missing; `cached_with_fallback` needs a deliberate rate.
- Liquidation/margin model for leverage > 1.
- Walk-forward parameter selection and a second instrument/period.
