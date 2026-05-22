#!/usr/bin/env python3
"""
run_sessions.py — Multi-window, multi-session backtest runner.

For each of W1–W4, runs Ungated (none gate) and V1 (drift_4h gate) across
three independently-windowed sub-sessions:
  European   07:00 → RTH_open
  US_RTH     RTH_open → RTH_close
  Post_close RTH_close → 21:00 (W1/W3/W4) or 22:00 (W2)

Each sub-session computes its OWN rolling z-score from its own bars only.
No cross-session concatenation.

Usage:
    python notebooks/run_sessions.py [--windows W1 W2 W3 W4] [--force]
"""

import argparse
import json
import sys
import numpy as np
import pandas as pd
import scipy.stats as spstats
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from strategy import (
    WINDOWS, WindowConfig, StrategyParams,
    load_sofr, load_dt_years, load_volume_gate,
    load_rth_bars, compute_z, build_entry_mask, simulate, compute_stats,
)

DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
RESULTS  = Path(__file__).parent.parent / 'results'

# ─── Session definitions per window ──────────────────────────────────────────
# (name, t_open, t_close)  — UTC time strings for between_time()
SESSION_MAP = {
    'W1': [
        ('European',   '07:00', '12:29'),
        ('US_RTH',     '12:30', '19:14'),
        ('Post_close', '19:15', '20:59'),
    ],
    'W2': [
        ('European',   '07:00', '13:29'),
        ('US_RTH',     '13:30', '20:14'),
        ('Post_close', '20:15', '21:59'),
    ],
    'W3': [
        ('European',   '07:00', '12:29'),
        ('US_RTH',     '12:30', '19:14'),
        ('Post_close', '19:15', '20:59'),
    ],
    'W4': [
        ('European',   '07:00', '12:29'),
        ('US_RTH',     '12:30', '19:14'),
        ('Post_close', '19:15', '20:59'),
    ],
}

GATES = {
    'Ungated': 'none',
    'V1':       'drift_4h',
}

N_LOTS = 10   # all runs use 10 lots (avoids banker's rounding bug, n_lots>=3)


# ─── Sharpe-per-trade ────────────────────────────────────────────────────────

def sharpe_per_trade(gross_pts: np.ndarray, n_boot: int = 5000, ci: float = 0.90):
    """
    Sharpe-per-trade = mean(gross_pts) / std(gross_pts, ddof=1).
    Returns (point_estimate, (lo, hi)) 90% CI via percentile bootstrap.
    """
    arr = np.array(gross_pts)
    n   = len(arr)
    if n < 3:
        return np.nan, (np.nan, np.nan)
    std = arr.std(ddof=1)
    if std == 0:
        return np.nan, (np.nan, np.nan)
    s = arr.mean() / std
    rng  = np.random.default_rng(42)
    boot = []
    for _ in range(n_boot):
        samp = rng.choice(arr, size=n, replace=True)
        sb   = samp.std(ddof=1)
        if sb > 0:
            boot.append(samp.mean() / sb)
    boot = np.array(boot)
    lo   = np.percentile(boot, 100 * (1 - ci) / 2)
    hi   = np.percentile(boot, 100 * (1 + ci) / 2)
    return s, (lo, hi)


# ─── Stats summary dict (extended with Sharpe-per-trade) ─────────────────────

def extended_stats(df: pd.DataFrame, params: StrategyParams) -> dict:
    stats = compute_stats(df, params)
    s, (lo, hi) = sharpe_per_trade(df['gross_pts'].values)
    stats['sharpe_pt']    = s
    stats['sharpe_pt_lo'] = lo
    stats['sharpe_pt_hi'] = hi
    return stats


# ─── Per-session runner ───────────────────────────────────────────────────────

def run_one_session(
    wkey:      str,
    sess_name: str,
    t_open:    str,
    t_close:   str,
    gate_name: str,
    gate:      str,
    sofr_utc:  pd.Series,
    dt_yr:     float,
    vol_gate:  dict,
    force:     bool,
) -> dict | None:
    """
    Returns stats dict or None if no trades / already exists.
    """
    cfg_base  = WINDOWS[wkey]
    label     = f'{sess_name}_{gate_name}'
    out       = RESULTS / f'{cfg_base.front}_{cfg_base.back}_{cfg_base.roll_start.replace("-", "")}' / label

    if out.exists() and not force:
        # Load existing trades.parquet and recompute stats
        trades_path = out / 'trades.parquet'
        if trades_path.exists():
            df = pd.read_parquet(trades_path)
            if df.empty:
                return None
            params = StrategyParams(n_lots=N_LOTS, regime_gate=gate)
            return extended_stats(df, params)
        return None

    out.mkdir(parents=True, exist_ok=True)

    # Build session-scoped WindowConfig (preserves fomc_utc, fomc_cut)
    cfg = WindowConfig(
        front=cfg_base.front,
        back=cfg_base.back,
        roll_start=cfg_base.roll_start,
        fomc_utc=cfg_base.fomc_utc,
        fomc_cut=cfg_base.fomc_cut,
        rth_open=t_open,
        rth_close=t_close,
    )

    params = StrategyParams(n_lots=N_LOTS, regime_gate=gate)

    try:
        spread, fv, dev = load_rth_bars(cfg, params, sofr_utc, dt_yr, DATA_DIR)
    except Exception as e:
        print(f'    !! load_rth_bars failed for {label}: {e}')
        return None

    if len(spread) == 0:
        print(f'    !! No bars for {label}')
        return None

    z     = compute_z(dev, params)
    emask = build_entry_mask(spread, vol_gate, cfg, params)
    df    = simulate(spread, dev, z, emask, cfg, params)

    if df.empty:
        print(f'    !! No trades for {label}')
        return None

    df.to_parquet(out / 'trades.parquet')
    ts = pd.DataFrame({'spread': spread, 'fv': fv, 'dev': dev, 'zscore': z},
                      index=spread.index)
    ts.to_parquet(out / 'timeseries.parquet')

    stats = extended_stats(df, params)

    def _serial(v):
        if isinstance(v, (np.integer,)):   return int(v)
        if isinstance(v, (np.floating,)):  return float(v)
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)): return None
        return v

    with open(out / 'stats.json', 'w') as f:
        json.dump({k: _serial(v) for k, v in stats.items()}, f, indent=2)

    return stats


# ─── Print helpers ────────────────────────────────────────────────────────────

def _fmt(stats: dict, label: str) -> str:
    n       = stats['n']
    wr      = stats['wr']
    pf      = stats['pf']
    avg_g   = stats['avg_gross']
    tot_g10 = stats['tot_gross_scaled']
    tot_n10 = stats['tot_net_tight_scaled']
    sl_pct  = stats.get('sl_pct', 0.0)
    t_g     = stats['t_gross']
    p_g     = stats['p_gross']
    sp      = stats.get('sharpe_pt', float('nan'))
    sp_lo   = stats.get('sharpe_pt_lo', float('nan'))
    sp_hi   = stats.get('sharpe_pt_hi', float('nan'))
    longs   = stats.get('long_n', 0)
    shorts  = stats.get('short_n', 0)

    return (
        f'  {label:<30}  '
        f'n={n:>3}  WR={wr:5.1f}%  PF={pf:5.2f}  '
        f'avg/lot=${avg_g:+6.2f}  '
        f'gross×{N_LOTS}=${tot_g10:+7.0f}  '
        f'net×{N_LOTS}=${tot_n10:+7.0f}  '
        f'SL={sl_pct:4.1f}%  '
        f't={t_g:+5.3f} p={p_g:.3f}  '
        f'Sharpe/tr={sp:+.3f} [90%CI {sp_lo:+.3f},{sp_hi:+.3f}]  '
        f'L={longs} S={shorts}'
    )


def print_window_results(wkey: str, all_results: dict):
    cfg = WINDOWS[wkey]
    print(f'\n{"═"*140}')
    print(f'  {wkey}: {cfg.front} → {cfg.back}  (roll start {cfg.roll_start})')
    print(f'{"═"*140}')
    for (sess, gate_name), stats in all_results.items():
        if stats is None:
            print(f'  {sess+"_"+gate_name:<30}  ── NO TRADES ──')
        else:
            print(_fmt(stats, f'{sess}_{gate_name}'))
    print()


# ─── Per-day and L/S decomposition for detailed view ─────────────────────────

def per_day_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades by trade_date × dir_label."""
    if df.empty:
        return pd.DataFrame()
    gb = df.groupby(['trade_date', 'dir_label'])
    return gb.agg(
        n        =('gross_usd', 'count'),
        WR_pct   =('gross_usd', lambda x: (x > 0).mean() * 100),
        gross_sum=('gross_usd', 'sum'),
        net_sum  =('net_Tight', 'sum'),
    ).round(2)


def print_day_breakdown(wkey: str, sess: str, gate_name: str, df: pd.DataFrame):
    if df.empty:
        return
    print(f'  ── {wkey} {sess}_{gate_name} per-day breakdown ──')
    breakdown = df.groupby('trade_date').agg(
        n       =('gross_usd', 'count'),
        WR_pct  =('gross_usd', lambda x: (x > 0).mean() * 100),
        gross   =('gross_usd', 'sum'),
        net     =('net_Tight', 'sum'),
        longs   =('direction', lambda x: (x > 0).sum()),
        shorts  =('direction', lambda x: (x < 0).sum()),
        day_lbl =('day_label', 'first'),
    ).round(2)
    for date, row in breakdown.iterrows():
        print(f'    {date} ({row["day_lbl"]:<6})  n={int(row["n"]):>2}  '
              f'WR={row["WR_pct"]:5.1f}%  gross=${row["gross"]:+7.2f}  '
              f'net=${row["net"]:+7.2f}  L={int(row["longs"])} S={int(row["shorts"])}')


# ─── Lot-scaling analysis ─────────────────────────────────────────────────────

def print_lot_scaling(stats_base: dict):
    """
    Show P&L at varying lot sizes given fixed per-lot gross and cost structure.
    Cost model: slip=$6.25/lot + inst_comm=$4.00/lot → $10.25/lot round-trip.
    Gross scales linearly; net = gross×lots - $10.25×lots = (gross_avg - $10.25)×lots.
    Break-even when avg_gross_per_lot > $10.25 → requires PF > 1 and WR > 50%.
    """
    n         = stats_base['n']
    avg_g_lot = stats_base['avg_gross']          # $/lot
    tc        = 8.04                              # $/lot round-trip: exch $4.60 + NFA $0.04 + broker $3.40
    avg_net   = avg_g_lot - tc
    print(f'\n  ── Lot-scaling analysis (per-trade avg gross = ${avg_g_lot:.2f}/lot) ──')
    print(f'  TC (tight slip + inst comm) = ${tc:.2f}/lot')
    print(f'  Net per trade per lot       = ${avg_net:+.2f}')
    print(f'  {"Lots":>5}  {"Tot gross":>12}  {"Tot TC":>10}  {"Tot net":>10}  {"Net/trade":>12}')
    for lots in [1, 5, 10, 20, 40, 100]:
        tot_g  = avg_g_lot * lots * n
        tot_tc = tc        * lots * n
        tot_n  = tot_g - tot_tc
        print(f'  {lots:>5}  ${tot_g:>10.0f}  ${tot_tc:>9.0f}  ${tot_n:>+9.0f}  ${avg_net*lots:>+10.2f}/tr')
    print()
    be_avg_needed = tc
    print(f'  Break-even requires avg_gross/lot > ${be_avg_needed:.2f}')
    print(f'  Current avg_gross/lot = ${avg_g_lot:.2f} → {"ABOVE" if avg_g_lot > be_avg_needed else "BELOW"} break-even threshold')
    print(f'  Lot size has NO impact on profitability per-lot;')
    print(f'  scaling only multiplies existing edge (positive or negative).')


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--windows', nargs='+', default=['W1','W2','W3','W4'],
                        choices=['W1','W2','W3','W4'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    for wkey in args.windows:
        cfg_base = WINDOWS[wkey]
        sessions = SESSION_MAP[wkey]

        print(f'\n{"─"*60}')
        print(f'  Loading shared data for {wkey}...')
        sofr_utc = load_sofr(DATA_DIR)
        dt_yr    = load_dt_years(cfg_base, DATA_DIR)
        # Use full-day window for vol_gate (volume gate is day-level, not sub-session)
        vol_gate = load_volume_gate(cfg_base, StrategyParams(), DATA_DIR)

        window_results = {}
        trades_cache   = {}   # (sess, gate_name) → df

        for (sess_name, t_open, t_close) in sessions:
            for gate_name, gate in GATES.items():
                label = f'{sess_name}_{gate_name}'
                print(f'  Running {wkey} {label}  [{t_open}–{t_close}]...')
                stats = run_one_session(
                    wkey, sess_name, t_open, t_close,
                    gate_name, gate,
                    sofr_utc, dt_yr, vol_gate,
                    force=args.force,
                )
                window_results[(sess_name, gate_name)] = stats

                # Cache trades for day breakdown
                key = (sess_name, gate_name)
                out = RESULTS / f'{cfg_base.front}_{cfg_base.back}_{cfg_base.roll_start.replace("-", "")}' / label
                tp  = out / 'trades.parquet'
                if tp.exists():
                    try:
                        trades_cache[key] = pd.read_parquet(tp)
                    except Exception:
                        trades_cache[key] = pd.DataFrame()

        # ── Summary table ──────────────────────────────────────────────────
        print_window_results(wkey, window_results)

        # ── Lot-scaling analysis on US_RTH Ungated ────────────────────────
        rth_stats = window_results.get(('US_RTH', 'Ungated'))
        if rth_stats and rth_stats['n'] > 0:
            print_lot_scaling(rth_stats)

        # ── Per-day breakdown for US_RTH (both gates) ─────────────────────
        for gate_name in GATES:
            key = ('US_RTH', gate_name)
            df  = trades_cache.get(key, pd.DataFrame())
            if not df.empty:
                print_day_breakdown(wkey, 'US_RTH', gate_name, df)

        # ── Long / short win-rate decomposition ───────────────────────────
        print(f'\n  ── {wkey} Long/Short breakdown by session ──')
        for (sess_name, t_open, t_close) in sessions:
            for gate_name in GATES:
                key = (sess_name, gate_name)
                df  = trades_cache.get(key, pd.DataFrame())
                if df.empty:
                    continue
                for d, dlabel in [(1, 'Long'), (-1, 'Short')]:
                    sub = df[df['direction'] == d]
                    if sub.empty:
                        continue
                    n   = len(sub)
                    wr  = (sub['gross_usd'] > 0).mean() * 100
                    avg = sub['gross_usd'].mean()
                    print(f'    {sess_name}_{gate_name:<20}  {dlabel:>5}  '
                          f'n={n:>2}  WR={wr:5.1f}%  avg=${avg:+.2f}/lot')


if __name__ == '__main__':
    main()
