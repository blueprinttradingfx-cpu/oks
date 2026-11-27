"""Engine tests. Expected values are hand-derived literals (see comments), never computed by the engine."""
import copy
import json
import pandas as pd
import pytest
from helpers import cfg, frame, write_candles, BASE_CFG
import strategy_engine as eng


def V(**sections):
    return eng.validate_config(cfg(**sections))


def plan(df, c):
    return eng.build_positions(df, c)


def run(df, c, funding=None):
    return eng.simulate(plan(df, c), c, funding)


# ---------------------------------------------------------------- positions
def test_exec_position_is_target_delayed_one_bar():
    """Break caught: filling on the same bar the signal was computed (lookahead)."""
    df = frame([100] * 6, target=[0, 1, 1, 0, -1, -1])
    p = plan(df, V())
    assert p["exec_pos"].tolist() == [0, 0, 1, 1, 0, -1]


@pytest.mark.parametrize("start,k,flat_idx", [
    ("2026-01-01 23:00", 1, {3}),      # bar 23:45 ends exactly at 00:00 funding
    ("2026-01-01 23:00", 2, {2, 3}),   # 23:30 and 23:45
    ("2026-01-02 07:00", 1, {3}),      # 07:45 ends at 08:00 funding (hour 8 is configured)
    ("2026-01-02 02:00", 1, set()),    # no funding boundary inside this window
])
def test_flat_before_funding_flattens_only_bars_ending_at_funding(start, k, flat_idx):
    """Break caught: flat window on the wrong bars (old code flattened AT 00:00 and ignored 08:00/16:00)."""
    c = V(execution={"flat_before_funding": {"enabled": True, "bars_before": k,
                                             "funding_hours_utc": [0, 8, 16]}})
    df = frame([100] * 8, target=[1] * 8, start=start)
    p = plan(df, c)
    want = [0 if (j == 0 or j in flat_idx) else 1 for j in range(8)]
    assert p["exec_pos"].tolist() == want
    assert set(p.index[p["flat_window"]]) == {df.index[j] for j in flat_idx}


def test_flat_before_funding_disabled_by_default_keeps_position():
    df = frame([100] * 8, target=[1] * 8, start="2026-01-01 23:00")
    assert plan(df, V())["exec_pos"].tolist() == [0, 1, 1, 1, 1, 1, 1, 1]


@pytest.mark.parametrize("mode,target,want", [
    ("long_only", [-1, -1, 1, 1], [0, 0, 0, 1]),
    ("short_only", [-1, -1, 1, 1], [0, -1, -1, 0]),
])
def test_position_mode_clips_disallowed_side(mode, target, want):
    """Break caught: position_mode ignored (old engine never read it)."""
    p = plan(frame([100] * 4, target=target), V(execution={"position_mode": mode}))
    assert p["exec_pos"].tolist() == want


# --------------------------------------------------------------- simulation
def test_long_round_trip_applies_slippage_and_fees_on_both_sides():
    """Break caught: wrong slippage direction, fee base, or missing exit fee.
    E=10000, fee=0.001, slip=0.002, flat price 100. entry 100.2, exit 99.8.
    pnl=10000*(99.8/100.2-1)=-39.92016; fees=10+9.96008 -> 9940.11976."""
    c = V(costs={"fee_per_side": 0.001, "slippage_per_side": 0.002})
    m, t, eq = run(frame([100] * 4, target=[1, 1, 0, 0]), c)
    assert m["final_equity"] == pytest.approx(9940.11976, abs=1e-4)
    assert len(t) == 1 and t.loc[0, "side"] == "long"
    assert t.loc[0, "entry_price"] == pytest.approx(100.2)
    assert t.loc[0, "exit_price"] == pytest.approx(99.8)
    assert t.loc[0, "fees"] == pytest.approx(19.96008, abs=1e-4)
    assert t.loc[0, "pnl"] == pytest.approx(-59.88024, abs=1e-4)   # net of fees: -39.92016-19.96008


def test_short_profits_when_price_falls_and_pays_costs():
    """Break caught: short sign errors. entry 99.8 (sell low), exit 90.18 (buy high).
    pnl=+963.92786; fees=10+9.03607 -> 10944.89178."""
    c = V(costs={"fee_per_side": 0.001, "slippage_per_side": 0.002})
    m, t, _ = run(frame([100, 100, 90, 90], target=[-1, -1, 0, 0]), c)
    assert m["final_equity"] == pytest.approx(10944.89178, abs=1e-4)
    assert t.loc[0, "side"] == "short"


@pytest.mark.parametrize("sizing,want", [
    ({"equity_fraction": 0.5}, 10500.0),      # notional 5000 * +10%
    ({"equity_fraction": 1.0}, 11000.0),
])
def test_equity_fraction_scales_notional(sizing, want):
    c = V(risk=dict(BASE_CFG["risk"], **sizing))
    m, _, _ = run(frame([100, 100, 110, 110], target=[1, 1, 0, 0]), c)
    assert m["final_equity"] == pytest.approx(want)


def test_leverage_scales_notional():
    """Break caught: leverage ignored. notional 2*10000 * +10% = +2000."""
    c = V(execution={"leverage": 2})
    m, _, _ = run(frame([100, 100, 110, 110], target=[1, 1, 0, 0]), c)
    assert m["final_equity"] == pytest.approx(12000.0)


def test_equity_curve_is_marked_to_market_while_trade_is_open():
    """Break caught: realized-only equity (old engine hid open-trade drawdown).
    long from open 100; mark at 80 -> equity 8000 before any exit; max DD = -20%."""
    df = frame([100, 100, 80, 100, 100], target=[1, 1, 1, 0, 0])
    m, _, eq = run(df, V())
    assert eq["equity"].tolist() == pytest.approx([10000, 10000, 8000, 10000, 10000, 10000])
    assert m["max_drawdown"] == pytest.approx(-0.2)
    assert m["exposure"] == pytest.approx(0.6)    # 3 of 5 bars held


def test_open_position_is_closed_at_last_close():
    """Break caught: position left open / valued at a stale price at end of data."""
    df = frame([100, 100, 100], closes=[100, 100, 110], target=[1, 1, 1])
    m, t, eq = run(df, V())
    assert m["final_equity"] == pytest.approx(11000.0)
    assert t.loc[0, "exit_price"] == pytest.approx(110.0)
    assert t.loc[0, "exit_reason"] == "end_of_data"
    assert pd.Timestamp(t.loc[0, "exit_time"]) == df.index[-1] + pd.Timedelta(minutes=15)
    assert m["buy_and_hold_return"] == pytest.approx(0.10)   # first open 100 -> last close 110


FUND_START = "2026-01-01 23:00"     # 6 bars: boundary j=4 is 00:00 (funding), final boundary 00:30


def _fund(rate):
    return pd.Series([rate], index=pd.DatetimeIndex(["2026-01-02 00:00"], tz="UTC"))


@pytest.mark.parametrize("target,rate,want", [
    ([1] * 6, 0.001, 9990.0),        # long pays positive funding: 10000*0.001
    ([-1] * 6, 0.001, 10010.0),      # short receives
    ([1] * 6, -0.001, 10010.0),      # long receives negative funding
    ([1, 1, 1, 0, 0, 0], 0.001, 9990.0),   # exits AT 00:00 open: held over the bar ending at T, so it pays
    ([1, 1, 0, 0, 0, 0], 0.001, 10000.0),  # flat before the snapshot: pays nothing
    ([0, 0, 0, 1, 1, 1], 0.001, 10000.0),  # opened exactly at 00:00: pays nothing
])
def test_funding_settlement_convention_and_sign(target, rate, want):
    """Break caught: funding sign flipped, or charged to a position not held at the snapshot."""
    c = V(costs={"fee_per_side": 0.0, "slippage_per_side": 0.0,
                 "funding": {"mode": "cached_when_available", "fallback_rate_per_8h": 0.0}})
    m, _, _ = run(frame([100] * 6, target=target, start=FUND_START), c, _fund(rate))
    assert m["final_equity"] == pytest.approx(want)


@pytest.mark.parametrize("mode,series,want", [
    ("none", _fund(0.001), 10000.0),                                   # cached data ignored
    ("cached_when_available", pd.Series(dtype=float), 10000.0),        # no data -> no charge
    ("cached_with_fallback", pd.Series(dtype=float), 9999.0),          # fallback 0.0001*10000
    ("cached_with_fallback", _fund(0.001), 9990.0),                    # cached beats fallback
])
def test_funding_modes(mode, series, want):
    c = V(costs={"fee_per_side": 0.0, "slippage_per_side": 0.0,
                 "funding": {"mode": mode, "fallback_rate_per_8h": 0.0001}})
    m, _, _ = run(frame([100] * 6, target=[1] * 6, start=FUND_START), c, series)
    assert m["final_equity"] == pytest.approx(want)


@pytest.mark.parametrize("series,want", [(_fund(0.001), 1.0), (pd.Series(dtype=float), 0.0)])
def test_funding_coverage_reports_share_of_settlements_with_data(series, want):
    c = V(costs={"fee_per_side": 0.0, "slippage_per_side": 0.0,
                 "funding": {"mode": "cached_when_available", "fallback_rate_per_8h": 0.0}})
    m, _, _ = run(frame([100] * 6, target=[1] * 6, start=FUND_START), c, series)
    assert m["funding_coverage"] == want


def test_leverage_wipeout_floors_equity_at_zero_and_stops_trading():
    """Break caught: negative equity. 20x long, -10% move: 10000 + 200000*(-0.1) < 0 -> liquidated at the mark."""
    c = V(execution={"leverage": 20})
    m, t, eq = run(frame([100, 100, 90, 90], target=[1, 1, 1, 1]), c)
    assert m["final_equity"] == 0.0 and m["max_drawdown"] == pytest.approx(-1.0)
    assert t.loc[0, "exit_reason"] == "liquidated"
    assert (eq["equity"] >= 0).all()


# ------------------------------------------------------------------ metrics
def test_profit_factor_is_null_not_zero_when_there_are_no_losses():
    """Break caught: old engine reported 0.0 for a perfect record."""
    m, _, _ = run(frame([100, 100, 110, 110], target=[1, 1, 0, 0]), V())
    assert m["win_rate"] == 1.0 and m["profit_factor"] is None


def test_metrics_are_strict_json_when_there_are_no_trades():
    """Break caught: NaN sharpe/ratios leaking into metrics.json."""
    m, t, eq = run(frame([100] * 5, target=[0] * 5), V())
    json.dumps(m, allow_nan=False)
    assert m["trades"] == 0 and len(t) == 0


def test_in_and_out_of_sample_blocks_split_equity_at_configured_fraction():
    """10 bars, oos 0.3 -> split at boundary 7. eq[0]=10000, eq[7]=11000 (price 110), eq[10]=13200 (close 132).
    IS=+10%, OOS=13200/11000-1=+20%."""
    opens = [100, 100, 110, 110, 110, 110, 110, 110, 120, 120]
    closes = opens[:-1] + [132]
    m, _, _ = run(frame(opens, closes, target=[1] * 10), V())
    assert m["in_sample"]["net_return"] == pytest.approx(0.10)
    assert m["out_of_sample"]["net_return"] == pytest.approx(0.20)


# ------------------------------------------------------------ config schema
def _mut(path, value):
    c = copy.deepcopy(BASE_CFG)
    node = c
    for k in path[:-1]:
        node = node[k]
    if value is eng_DELETE:
        node.pop(path[-1], None)
    else:
        node[path[-1]] = value
    return c


eng_DELETE = object()


@pytest.mark.parametrize("path,value,needle", [
    (("execution", "force_flat_at_utc"), "23:45", "force_flat_at_utc"),
    (("execution", "force_flat_before_funding"), True, "force_flat_before_funding"),
    (("costs", "mystery"), 1, "mystery"),
    (("execution", "pyramiding"), True, "pyramiding"),
    (("costs", "funding"), "cached_when_available", "funding"),
    (("costs", "funding", "mode"), "magic", "mode"),
    (("execution", "position_mode"), "sideways", "position_mode"),
    (("costs", "fee_per_side"), -0.1, "fee_per_side"),
    (("signals", "fast_ema"), 50, "fast_ema"),          # fast >= slow(3)
])
def test_config_rejects_bad_or_unsupported_keys_loudly(path, value, needle):
    """Break caught: keys silently ignored (old engine never read 4 of them)."""
    with pytest.raises(eng.ConfigError, match=needle):
        eng.validate_config(_mut(path, value))


# --------------------------------------------------------------- data layer
def _data_cfg():
    return V(data={"start": "2027-01-01T00:00:00Z", "end": "2027-01-02T00:00:00Z"})


def _write_pair(d, ltf_rows, htf_rows):
    write_candles(d / "BTC-USDT-SWAP_15m_2027.csv.gz", ltf_rows)
    write_candles(d / "BTC-USDT-SWAP_1H_2027.csv.gz", htf_rows)


def test_load_data_finds_files_for_any_year_and_drops_unconfirmed_last_candle(tmp_path):
    """Break caught: hardcoded _2026 path; forming candle leaking into backtest."""
    l = [("2027-01-01T00:00:00Z", 1, 2, 1, 2, 1), ("2027-01-01T00:15:00Z", 2, 3, 2, 3, 1),
         ("2027-01-01T00:30:00Z", 3, 4, 3, 4, 0)]
    h = [("2027-01-01T00:00:00Z", 1, 4, 1, 3, 1)]
    _write_pair(tmp_path, l, h)
    df, htf = eng.load_data(_data_cfg(), data_dir=tmp_path)
    assert len(df) == 2 and df.index[-1] == pd.Timestamp("2027-01-01T00:15:00Z")
    assert len(htf) == 1


def test_load_data_rejects_duplicate_timestamps(tmp_path):
    l = [("2027-01-01T00:00:00Z", 1, 2, 1, 2, 1), ("2027-01-01T00:00:00Z", 2, 3, 2, 3, 1)]
    _write_pair(tmp_path, l, [("2027-01-01T00:00:00Z", 1, 4, 1, 3, 1)])
    with pytest.raises(eng.DataError, match="duplicate"):
        eng.load_data(_data_cfg(), data_dir=tmp_path)


# ------------------------------------------------------------------ signals
def _ltf_htf(ltf_closes, htf_closes):
    li = pd.date_range("2026-01-01", periods=len(ltf_closes), freq="15min", tz="UTC")
    hi = pd.date_range("2026-01-01", periods=len(htf_closes), freq="1h", tz="UTC")
    mk = lambda i, c: pd.DataFrame({"open": c, "high": c, "low": c, "close": c}, index=i, dtype=float)
    return mk(li, ltf_closes), mk(hi, htf_closes)


def test_higher_timeframe_bias_ignores_the_candle_still_forming():
    """Break caught: lookahead. Hour-5 candle closes at 06:00, so bars 05:00-05:30 (decided at 05:15-05:45)
    must not change when that candle's close changes."""
    base = [100.0] * 5
    ltf = [100.0] * 24
    out = []
    for last in (200.0, 50.0):
        df, htf = _ltf_htf(ltf, base + [last])
        s = eng.build_signals(df, htf, V())
        out.append(s[["long_bias", "short_bias"]].iloc[20:23].astype(bool).values.tolist())
    assert out[0] == out[1]


def test_require_alignment_false_lets_ltf_trend_trade_against_htf_bias():
    """Break caught: require_alignment ignored (old engine never read it)."""
    ltf = [100.0 + i for i in range(24)]            # LTF uptrend
    htf = [200.0 - 10 * i for i in range(6)]        # HTF downtrend
    df, h = _ltf_htf(ltf, htf)
    gated = eng.build_signals(df, h, V())["target"]
    c = V(signals={"higher_timeframe": {"fast_ema": 2, "slow_ema": 3, "require_alignment": False}})
    free = eng.build_signals(df, h, c)["target"]
    assert (gated == 1).sum() == 0
    assert (free.iloc[3:] == 1).all()
