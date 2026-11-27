"""Shared research engine: config schema, data layer, signals, execution plan, simulator, metrics.

Both runners consume the SAME execution plan (``build_positions``), so they cannot drift apart.

Conventions (all tested in tests/test_engine.py):
* A signal is computed from bar i's close and filled at bar i+1's open.
* ``exec_pos[j]`` is the position held over bar j (open[j] -> open[j+1]).
* Equity is marked to market at every bar open, after fills at that open.
* Funding settles at a boundary timestamp T (hours in ``funding_hours_utc``). A position pays if it was
  held over the bar that ENDS at T, so a position that exits at T's open still pays and one that enters at
  T's open does not. Long pays when the rate is positive. Notional = position value at T's open.
* Fills: entry/exit price = open * (1 +/- slippage); fee = fee_per_side * fill notional.
* Position size is fixed at entry: ``equity * equity_fraction * leverage`` (no intra-trade rebalancing).
* An open position is closed at the last bar's close ("end_of_data") with the same costs.
* Leverage > 1 is supported for sizing only; a trade whose equity hits zero is liquidated at the mark
  (maintenance margin / exchange liquidation prices are NOT modelled).
"""
from __future__ import annotations

import copy
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TF_MINUTES = {"5m": 5, "15m": 15, "1H": 60}
_TF_ALIAS = {"5m": "5m", "15m": "15m", "1H": "1H", "1h": "1H"}


class ConfigError(ValueError):
    """Invalid or unsupported strategy configuration."""


class DataError(ValueError):
    """Missing or malformed market data."""


# ------------------------------------------------------------------ config
_ALLOWED = {
    "strategy_id": None, "name": None,
    "data": {"instrument": None, "timeframe": None, "higher_timeframe": None, "start": None, "end": None},
    "signals": {"type": None, "fast_ema": None, "slow_ema": None,
                "higher_timeframe": {"fast_ema": None, "slow_ema": None, "require_alignment": None}},
    "execution": {"signal_execution": None, "position_mode": None, "leverage": None, "pyramiding": None,
                  "flat_before_funding": {"enabled": None, "bars_before": None, "funding_hours_utc": None}},
    "risk": {"position_sizing": None, "equity_fraction": None, "initial_equity": None},
    "costs": {"fee_per_side": None, "slippage_per_side": None,
              "funding": {"mode": None, "fallback_rate_per_8h": None}},
    "validation": {"data_source": None, "out_of_sample_fraction": None},
}

_DEFAULTS = {
    "name": "",
    "signals": {"type": "ema_trend", "higher_timeframe": {"require_alignment": True}},
    "execution": {"signal_execution": "next_bar_open", "position_mode": "long_short", "leverage": 1,
                  "pyramiding": False,
                  "flat_before_funding": {"enabled": False, "bars_before": 1, "funding_hours_utc": [0, 8, 16]}},
    "risk": {"position_sizing": "equity_fraction"},
    "costs": {"funding": {"mode": "none", "fallback_rate_per_8h": 0.0}},
    "validation": {"data_source": "local_okx_cache", "out_of_sample_fraction": 0.3},
}

_REQUIRED = [
    "strategy_id", "data.instrument", "data.timeframe", "data.higher_timeframe", "data.start", "data.end",
    "signals.fast_ema", "signals.slow_ema", "signals.higher_timeframe.fast_ema",
    "signals.higher_timeframe.slow_ema", "risk.equity_fraction", "risk.initial_equity",
    "costs.fee_per_side", "costs.slippage_per_side",
]

_LEGACY_HINTS = {
    "execution.force_flat_at_utc": "replaced by execution.flat_before_funding {enabled, bars_before, funding_hours_utc}",
    "execution.force_flat_before_funding": "replaced by execution.flat_before_funding {enabled, bars_before, funding_hours_utc}",
}


def _check_keys(user, allowed, path, errors):
    for k, v in user.items():
        dotted = f"{path}.{k}" if path else k
        if k not in allowed:
            hint = _LEGACY_HINTS.get(dotted)
            errors.append(f"unknown key '{dotted}'" + (f" ({hint})" if hint else ""))
        elif isinstance(allowed[k], dict):
            if not isinstance(v, dict):
                errors.append(f"'{dotted}' must be a mapping, got {type(v).__name__}")
            else:
                _check_keys(v, allowed[k], dotted, errors)


def _merge(defaults, user):
    out = copy.deepcopy(defaults)
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _get(cfg, dotted):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _ts(value, key, errors):
    try:
        t = pd.Timestamp(value)
        t = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return t.isoformat()
    except Exception:  # noqa: BLE001
        errors.append(f"'{key}' is not a valid timestamp: {value!r}")
        return value


def validate_config(raw):
    """Return a normalised deep copy of ``raw`` or raise ConfigError listing every problem."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    errors: list[str] = []
    _check_keys(raw, _ALLOWED, "", errors)
    if errors:
        raise ConfigError("; ".join(errors))
    cfg = _merge(_DEFAULTS, raw)

    for dotted in _REQUIRED:
        if _get(cfg, dotted) is None:
            errors.append(f"missing required key '{dotted}'")
    if errors:
        raise ConfigError("; ".join(errors))

    def num(dotted, lo=None, hi=None, lo_incl=True, hi_incl=True):
        v = _get(cfg, dotted)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errors.append(f"'{dotted}' must be a number, got {v!r}")
            return
        if lo is not None and (v < lo if lo_incl else v <= lo):
            errors.append(f"'{dotted}' must be {'>=' if lo_incl else '>'} {lo}, got {v}")
        if hi is not None and (v > hi if hi_incl else v >= hi):
            errors.append(f"'{dotted}' must be {'<=' if hi_incl else '<'} {hi}, got {v}")

    def one_of(dotted, options):
        v = _get(cfg, dotted)
        if v not in options:
            errors.append(f"'{dotted}' must be one of {sorted(options)}, got {v!r}")

    d = cfg["data"]
    for key in ("timeframe", "higher_timeframe"):
        if d[key] not in _TF_ALIAS:
            errors.append(f"'data.{key}' must be one of {sorted(_TF_ALIAS)}, got {d[key]!r}")
        else:
            d[key] = _TF_ALIAS[d[key]]
    d["start"], d["end"] = _ts(d["start"], "data.start", errors), _ts(d["end"], "data.end", errors)
    if not errors and pd.Timestamp(d["start"]) >= pd.Timestamp(d["end"]):
        errors.append("'data.start' must be before 'data.end'")
    if not errors and TF_MINUTES[d["higher_timeframe"]] < TF_MINUTES[d["timeframe"]]:
        errors.append("'data.higher_timeframe' must not be shorter than 'data.timeframe'")

    s = cfg["signals"]
    one_of("signals.type", {"ema_trend"})
    for fast, slow, label in ((s["fast_ema"], s["slow_ema"], "signals"),
                              (s["higher_timeframe"]["fast_ema"], s["higher_timeframe"]["slow_ema"],
                               "signals.higher_timeframe")):
        if not all(isinstance(x, int) and not isinstance(x, bool) and x >= 1 for x in (fast, slow)):
            errors.append(f"'{label}.fast_ema' and '{label}.slow_ema' must be integers >= 1")
        elif fast >= slow:
            errors.append(f"'{label}.fast_ema' ({fast}) must be smaller than slow_ema ({slow})")
    if not isinstance(s["higher_timeframe"]["require_alignment"], bool):
        errors.append("'signals.higher_timeframe.require_alignment' must be true or false")

    e = cfg["execution"]
    one_of("execution.signal_execution", {"next_bar_open"})
    one_of("execution.position_mode", {"long_short", "long_only", "short_only"})
    num("execution.leverage", 0, lo_incl=False)
    if e["pyramiding"] is not False:
        errors.append("'execution.pyramiding' must be false (adding to positions is not supported)")
    fb = e["flat_before_funding"]
    if not isinstance(fb["enabled"], bool):
        errors.append("'execution.flat_before_funding.enabled' must be true or false")
    if not (isinstance(fb["bars_before"], int) and not isinstance(fb["bars_before"], bool) and fb["bars_before"] >= 1):
        errors.append("'execution.flat_before_funding.bars_before' must be an integer >= 1")
    hrs = fb["funding_hours_utc"]
    if not (isinstance(hrs, list) and hrs and all(isinstance(h, int) and 0 <= h <= 23 for h in hrs)):
        errors.append("'execution.flat_before_funding.funding_hours_utc' must be a non-empty list of hours 0-23")

    one_of("risk.position_sizing", {"equity_fraction"})
    num("risk.equity_fraction", 0, 1, lo_incl=False)
    num("risk.initial_equity", 0, lo_incl=False)
    num("costs.fee_per_side", 0)
    num("costs.slippage_per_side", 0)
    one_of("costs.funding.mode", {"none", "cached_when_available", "cached_with_fallback"})
    num("costs.funding.fallback_rate_per_8h")
    num("validation.out_of_sample_fraction", 0, 1, hi_incl=False)

    if errors:
        raise ConfigError("; ".join(errors))
    return cfg


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return validate_config(yaml.safe_load(f))


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -------------------------------------------------------------------- data
def _ns_utc_index(idx):
    return idx.as_unit("ns") if hasattr(idx, "as_unit") else idx


def _read_candles(files, minutes):
    parts = []
    for f in files:
        x = pd.read_csv(f, compression="gzip")
        col = "timestamp_utc" if "timestamp_utc" in x.columns else "timestamp"
        x[col] = pd.to_datetime(x[col], utc=True)
        x = x.set_index(col)
        x.index.name = "timestamp"
        if "confirm" in x.columns:
            x = x[x["confirm"] == 1]          # never trade on a still-forming candle
        parts.append(x)
    df = pd.concat(parts).sort_index()
    df.index = _ns_utc_index(df.index)
    if df.index.has_duplicates:
        raise DataError(f"duplicate timestamps in {[Path(f).name for f in files]}")
    gaps = int((df.index.to_series().diff().dropna() != pd.Timedelta(minutes=minutes)).sum())
    if gaps:
        warnings.warn(f"{gaps} timestamp gap(s) in {[Path(f).name for f in files]}", stacklevel=3)
    return df


def _candle_files(data_dir, instrument, tf):
    files = sorted(Path(data_dir).glob(f"{instrument}_{tf}_*.csv.gz"))
    if not files:
        raise DataError(f"no candle file matching {instrument}_{tf}_*.csv.gz in {data_dir}")
    return files


def load_data(cfg, data_dir=DATA_DIR):
    d = cfg["data"]
    df = _read_candles(_candle_files(data_dir, d["instrument"], d["timeframe"]), TF_MINUTES[d["timeframe"]])
    h = _read_candles(_candle_files(data_dir, d["instrument"], d["higher_timeframe"]),
                      TF_MINUTES[d["higher_timeframe"]])
    start, end = pd.Timestamp(d["start"]), pd.Timestamp(d["end"])
    df = df.loc[start:end].copy()
    if df.empty:
        raise DataError(f"no {d['timeframe']} candles between {start} and {end}")
    return df, h.loc[:end].copy()


def load_funding(cfg, data_dir=DATA_DIR):
    """Realised funding rate per settlement, indexed by UTC timestamp. Empty if mode is 'none' or no file."""
    empty = pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    if cfg["costs"]["funding"]["mode"] == "none":
        return empty
    files = sorted(Path(data_dir).glob(f"{cfg['data']['instrument']}_funding_*.csv.gz"))
    if not files:
        return empty
    x = pd.concat([pd.read_csv(f, compression="gzip") for f in files])
    x["funding_time_utc"] = pd.to_datetime(x["funding_time_utc"], utc=True)
    col = "realized_rate" if "realized_rate" in x.columns else "funding_rate"
    s = x.set_index("funding_time_utc")[col].astype(float).sort_index()
    s.index = _ns_utc_index(s.index)
    if s.index.has_duplicates:
        raise DataError("duplicate funding timestamps")
    return s


# ----------------------------------------------------------------- signals
def build_signals(df, htf, cfg):
    s = cfg["signals"]
    out = df.copy()
    out["ema_fast"] = out["close"].ewm(span=s["fast_ema"], adjust=False, min_periods=s["fast_ema"]).mean()
    out["ema_slow"] = out["close"].ewm(span=s["slow_ema"], adjust=False, min_periods=s["slow_ema"]).mean()
    hs = s["higher_timeframe"]
    h = htf[["close"]].copy()
    h["htf_fast"] = h["close"].ewm(span=hs["fast_ema"], adjust=False, min_periods=hs["fast_ema"]).mean()
    h["htf_slow"] = h["close"].ewm(span=hs["slow_ema"], adjust=False, min_periods=hs["slow_ema"]).mean()
    h["long_bias"] = h["htf_fast"] > h["htf_slow"]
    h["short_bias"] = h["htf_fast"] < h["htf_slow"]
    # A higher-timeframe candle labelled T closes at T+tf; shifting by one label means only completed
    # candles are visible to a lower-timeframe bar.
    h = h[["long_bias", "short_bias"]].astype(float).shift(1)
    out = pd.merge_asof(out.sort_index(), h.sort_index(), left_index=True, right_index=True, direction="backward")
    out["long_bias"] = out["long_bias"].fillna(0.0).astype(bool)
    out["short_bias"] = out["short_bias"].fillna(0.0).astype(bool)
    up = out["ema_fast"] > out["ema_slow"]
    down = out["ema_fast"] < out["ema_slow"]
    if hs["require_alignment"]:
        up, down = up & out["long_bias"], down & out["short_bias"]
    out["target"] = 0
    out.loc[up, "target"] = 1
    out.loc[down, "target"] = -1
    return out


def build_positions(sig, cfg):
    """Turn per-bar targets into the position actually held over each bar (the shared execution plan)."""
    out = sig.copy()
    t = out["target"].astype(int)
    mode = cfg["execution"]["position_mode"]
    if mode == "long_only":
        t = t.clip(lower=0)
    elif mode == "short_only":
        t = t.clip(upper=0)
    pos = t.shift(1).fillna(0).astype(int)
    flat = np.zeros(len(out), dtype=bool)
    fb = cfg["execution"]["flat_before_funding"]
    if fb["enabled"]:
        bar = pd.Timedelta(minutes=TF_MINUTES[cfg["data"]["timeframe"]])
        end = out.index.tz_convert("UTC") + bar
        for m in range(fb["bars_before"]):
            ts = end + m * bar
            flat |= np.asarray(ts.hour.isin(fb["funding_hours_utc"]) & (ts.minute == 0))
    pos = pos.where(~flat, 0)
    out["exec_pos"] = pos.astype(int)
    out["flat_window"] = flat
    return out


def prepare(cfg, data_dir=DATA_DIR):
    """Single entry point both runners use: data -> signals -> execution plan, plus funding series."""
    df, htf = load_data(cfg, data_dir)
    return build_positions(build_signals(df, htf, cfg), cfg), load_funding(cfg, data_dir)


# --------------------------------------------------------------- simulator
TRADE_COLS = ["entry_time", "exit_time", "side", "entry_price", "exit_price", "notional", "fees", "funding",
              "pnl", "return", "bars_held", "exit_reason"]


def _funding_by_boundary(bidx, cfg, funding):
    """Rate applied at each boundary (0 where nothing settles) and the share of settlements with data."""
    mode = cfg["costs"]["funding"]["mode"]
    n = len(bidx)
    rates = np.zeros(n)
    if mode == "none":
        return rates, None
    hours = cfg["execution"]["flat_before_funding"]["funding_hours_utc"]
    settle = np.asarray(bidx.hour.isin(hours) & (bidx.minute == 0))
    if not settle.any():
        return rates, None
    have = funding if funding is not None and len(funding) else pd.Series(dtype=float, index=bidx[:0])
    vals = have.reindex(bidx[settle])
    covered = vals.notna().to_numpy()
    fallback = cfg["costs"]["funding"]["fallback_rate_per_8h"] if mode == "cached_with_fallback" else 0.0
    rates[settle] = np.where(covered, vals.fillna(0.0).to_numpy(), fallback)
    return rates, float(covered.mean())


def _r(x, nd=8):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def _block(eq, P, a, b, ppy):
    seg = eq[a:b + 1]
    peak = np.maximum.accumulate(seg)
    prev = seg[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(prev > 0, seg[1:] / prev - 1, 0.0)
    sd = r.std(ddof=1) if len(r) > 2 else 0.0
    sharpe = float(r.mean() / sd * np.sqrt(ppy)) if np.isfinite(sd) and sd > 1e-12 else None
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = np.where(peak > 0, seg / peak - 1, 0.0)
        net = seg[-1] / seg[0] - 1 if seg[0] > 0 else None     # undefined once equity is already zero
    return {"bars": int(b - a),
            "net_return": _r(net) if net is not None else None,
            "max_drawdown": _r(dd.min()),
            "sharpe": _r(sharpe),
            "buy_and_hold_return": _r(P[b] / P[a] - 1)}


def summarize(cfg, bidx, P, eq, pos, tdf, coverage=None):
    """Metrics from a marked-to-market equity path. Shared by every engine so definitions cannot diverge.

    bidx/P/eq have n+1 entries (one per bar open plus the final close); pos has n+1 (last is 0)."""
    n = len(bidx) - 1
    minutes = TF_MINUTES[cfg["data"]["timeframe"]]
    E0 = float(cfg["risk"]["initial_equity"])
    pnl = tdf["pnl"].astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    ppy = 525600.0 / minutes
    whole = _block(eq, P, 0, n, ppy)
    oos = cfg["validation"]["out_of_sample_fraction"]
    split = int(round(n * (1 - oos)))
    has_split = oos > 0 and 1 <= split < n
    return {
        "strategy_id": cfg["strategy_id"],
        "start": bidx[0].isoformat(), "end": bidx[-1].isoformat(), "bars": n,
        "initial_equity": E0,
        "final_equity": _r(eq[-1], 6),
        "net_return": whole["net_return"], "max_drawdown": whole["max_drawdown"], "sharpe": whole["sharpe"],
        "buy_and_hold_return": whole["buy_and_hold_return"],
        "trades": int(len(tdf)),
        "win_rate": _r((pnl > 0).mean()) if len(tdf) else None,
        "profit_factor": _r(wins.sum() / abs(losses.sum())) if len(losses) and len(wins) else None,
        "avg_trade_return": _r(tdf["return"].astype(float).mean()) if len(tdf) else None,
        "avg_bars_held": _r(tdf["bars_held"].astype(float).mean(), 2) if len(tdf) else None,
        "exposure": _r((pos[:n] != 0).mean()),
        "total_fees": _r(tdf["fees"].sum(), 6) if len(tdf) else 0.0,
        "total_funding": _r(tdf["funding"].sum(), 6) if len(tdf) else 0.0,
        "funding_mode": cfg["costs"]["funding"]["mode"],
        "funding_coverage": _r(coverage),
        "in_sample": _block(eq, P, 0, split, ppy) if has_split else None,
        "out_of_sample": _block(eq, P, split, n, ppy) if has_split else None,
        "engine_assumptions": ("next-bar-open fills; fee+slippage per side on fill notional; marked-to-market "
                               "equity; fixed notional per trade; open position closed at last close"),
    }


def simulate(plan, cfg, funding=None):
    n = len(plan)
    idx = plan.index
    minutes = TF_MINUTES[cfg["data"]["timeframe"]]
    bar = pd.Timedelta(minutes=minutes)
    op = plan["open"].to_numpy(float)
    cl = plan["close"].to_numpy(float)
    P = np.append(op, cl[-1])                              # boundary prices; last boundary = final close
    bidx = idx.append(pd.DatetimeIndex([idx[-1] + bar]))
    pos = np.append(plan["exec_pos"].to_numpy(int), 0)     # final boundary: no position
    flat = np.append(plan["flat_window"].to_numpy(bool), False)
    fee = cfg["costs"]["fee_per_side"]
    slip = cfg["costs"]["slippage_per_side"]
    E0 = float(cfg["risk"]["initial_equity"])
    expo = cfg["risk"]["equity_fraction"] * cfg["execution"]["leverage"]
    f_rate, coverage = _funding_by_boundary(bidx, cfg, funding)

    cash, cur, N0, epx = E0, 0, 0.0, 0.0
    eq = np.empty(n + 1)
    held = np.zeros(n + 1, dtype=int)
    trades: list[dict] = []
    tr = None
    dead = False

    def close_trade(j, xpx, reason):
        nonlocal tr
        tr.update(exit_time=bidx[j].isoformat(), exit_price=xpx, exit_reason=reason, bars_held=j - tr.pop("_j0"))
        tr["pnl"] = cash - tr["_eq0"]
        tr["return"] = tr["pnl"] / tr.pop("_eq0")
        trades.append(tr)
        tr = None

    for j in range(n + 1):
        px = P[j]
        if cur != 0 and f_rate[j] != 0.0:
            pay = cur * N0 * (px / epx) * f_rate[j]
            cash -= pay
            tr["funding"] -= pay
        new = int(pos[j])
        if new != cur:
            if cur != 0:
                xpx = px * (1 - slip * cur)
                nx = N0 * xpx / epx
                cash += cur * N0 * (xpx / epx - 1) - fee * nx
                tr["fees"] += fee * nx
                close_trade(j, xpx, "end_of_data" if j == n else ("flat_window" if (new == 0 and flat[j]) else "signal"))
                cur = 0
            if new != 0:
                eq0 = cash
                epx = px * (1 + slip * new)
                N0 = expo * cash
                cash -= fee * N0
                cur = new
                tr = {"entry_time": bidx[j].isoformat(), "side": "long" if new > 0 else "short",
                      "entry_price": epx, "notional": N0, "fees": fee * N0, "funding": 0.0,
                      "_eq0": eq0, "_j0": j}
        unreal = cur * N0 * (px / epx - 1) if cur else 0.0
        if cur and cash + unreal <= 0:                      # wiped out: floor at zero, stop trading
            cash = 0.0
            close_trade(j, px, "liquidated")
            cur, unreal, dead = 0, 0.0, True
        eq[j] = cash + unreal
        held[j] = cur
        if dead:
            eq[j + 1:] = 0.0
            break

    tdf = pd.DataFrame(trades, columns=TRADE_COLS)
    metrics = summarize(cfg, bidx, P, eq, pos, tdf, coverage)
    equity = pd.DataFrame({"timestamp": [t.isoformat() for t in bidx], "equity": eq, "position": held})
    return metrics, tdf, equity


def save_outputs(out_dir, metrics, trades, equity):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    trades.to_csv(out_dir / "trades.csv", index=False)
    equity.to_csv(out_dir / "equity_curve.csv", index=False)
