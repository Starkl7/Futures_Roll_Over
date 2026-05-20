#!/usr/bin/env python3
"""
run_backtest.py — CLI runner for the ES calendar spread backtest.

Runs one or all roll windows and saves results to results/<FRONT>_<BACK>_<YYYYMMDD>/:
  trades.parquet      — per-trade DataFrame (1-lot P&L)
  timeseries.parquet  — spread / fv / dev / zscore at 1s resolution
  volume_arc.parquet  — daily back-share arc (if ohlcv1d files present)
  config.json         — window config snapshot
  stats.json          — full stats dict

Usage:
    python notebooks/run_backtest.py --window W1
    python notebooks/run_backtest.py --window W2 --lots 10
    python notebooks/run_backtest.py --window all --gate fv_dev --force
    python notebooks/run_backtest.py --window W1 --gate slope
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from strategy import (
    WINDOWS, StrategyParams,
    load_sofr, load_dt_years, load_volume_gate, load_volume_arc,
    load_rth_bars, compute_z, build_entry_mask, simulate, compute_stats,
)

DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
RESULTS  = Path(__file__).parent.parent / 'results'


def _results_path(cfg, gate: str = 'none') -> Path:
    gate_dir = gate.replace('+', '_')
    return RESULTS / f'{cfg.front}_{cfg.back}_{cfg.roll_start.replace("-", "")}' / gate_dir


def run_window(window_key: str, params: StrategyParams, force: bool = False) -> Path:
    cfg = WINDOWS[window_key]
    out = _results_path(cfg, params.regime_gate)

    if out.exists() and not force:
        print(f'  Results already exist at {out}')
        print(f'  Use --force to overwrite.')
        return out

    out.mkdir(parents=True, exist_ok=True)
    print(f'\n{"=" * 60}')
    print(f'  Window {window_key}: {cfg.front} → {cfg.back}  (roll start {cfg.roll_start})')
    print(f'{"=" * 60}')

    print('Loading SOFR...')
    sofr_utc = load_sofr(DATA_DIR)

    print('Loading ΔT from definitions...')
    dt_yr = load_dt_years(cfg, DATA_DIR)

    print('Loading RTH bars...')
    spread, fv, dev = load_rth_bars(cfg, params, sofr_utc, dt_yr, DATA_DIR)
    print(f'  {len(spread):,} RTH 1s bars  ({spread.index[0].date()} → {spread.index[-1].date()})')

    print('Loading volume gate...')
    vol_gate  = load_volume_gate(cfg, params, DATA_DIR)
    open_days = sum(1 for v in vol_gate.values() if v)
    print(f'  Gate ({params.vol_gate_low:.0%}–{params.vol_gate_high:.0%}): {open_days} active days')

    print('Running simulation...')
    z     = compute_z(dev, params)
    emask = build_entry_mask(spread, vol_gate, cfg, params)
    df    = simulate(spread, dev, z, emask, cfg, params)
    print(f'  {len(df)} trades generated')

    if df.empty:
        print('  ERROR: no trades generated. Check data and gate settings.')
        return out

    stats = compute_stats(df, params)

    # ── Save outputs ──────────────────────────────────────────────────────────
    df.to_parquet(out / 'trades.parquet')

    ts = pd.DataFrame({'spread': spread, 'fv': fv, 'dev': dev, 'zscore': z},
                      index=spread.index)
    ts.to_parquet(out / 'timeseries.parquet')

    arc = load_volume_arc(cfg, DATA_DIR)
    if not arc.empty:
        arc.to_parquet(out / 'volume_arc.parquet')

    with open(out / 'config.json', 'w') as f:
        json.dump({
            'front':      cfg.front,
            'back':       cfg.back,
            'roll_start': cfg.roll_start,
            'fomc_utc':   str(cfg.fomc_utc),
            'fomc_cut':   cfg.fomc_cut,
            'rth_open':   cfg.rth_open,
            'rth_close':  cfg.rth_close,
            'window':     params.window,
            'threshold':  params.threshold,
            'tp':         params.tp,
            'sl':         params.sl,
            'n_lots':     params.n_lots,
        }, f, indent=2)

    stats_serialisable = {}
    for k, v in stats.items():
        if isinstance(v, np.integer):
            stats_serialisable[k] = int(v)
        elif isinstance(v, np.floating):
            stats_serialisable[k] = float(v)
        elif isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            stats_serialisable[k] = None
        else:
            stats_serialisable[k] = v
    with open(out / 'stats.json', 'w') as f:
        json.dump(stats_serialisable, f, indent=2)

    _print_summary(stats, out)
    return out


def _print_summary(stats: dict, out: Path):
    lots = stats['n_lots']
    print(f'\n  Saved to {out}/')
    print(f'  {"─" * 40}')
    print(f'  n={stats["n"]}  WR={stats["wr"]:.1f}%  PF={stats["pf"]:.2f}')
    print(f'  Avg gross (1 lot): ${stats["avg_gross"]:.2f}')
    print(f'  Tot gross (1 lot): ${stats["tot_gross"]:.0f}')
    print(f'  Avg net inst-tight (1 lot): ${stats["avg_net_tight"]:.2f}')
    if lots > 1:
        print(f'  ── Scaled to {lots} lots ──')
        print(f'  Tot gross: ${stats["tot_gross_scaled"]:.0f}')
        print(f'  Tot net (inst tight): ${stats["tot_net_tight_scaled"]:.0f}')
        print(f'  MDD: ${stats["mdd_scaled"]:.0f}')
    print(f'  t={stats["t_gross"]:.3f}  p={stats["p_gross"]:.4f}')
    print(f'  {"─" * 40}')


def main():
    parser = argparse.ArgumentParser(description='ES calendar spread backtest runner')
    parser.add_argument('--window', default='W1', choices=['W1', 'W2', 'W3', 'W4', 'all'],
                        help='Which window to run (default: W1)')
    parser.add_argument('--lots', type=int, default=1,
                        help='Number of lots per trade (default: 1)')
    parser.add_argument('--gate', default='none',
                        choices=['none', 'fv_dev', 'slope', 'fv_dev+slope', 'return',
                                 'half_life', 'kalman', 'session', 'ofi', 'drift_4h',
                                 'half_life+kalman', 'kalman+session'],
                        help='Regime gate to apply (default: none)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing results')
    args = parser.parse_args()

    params = StrategyParams(n_lots=args.lots, regime_gate=args.gate)
    keys   = list(WINDOWS.keys()) if args.window == 'all' else [args.window]
    for key in keys:
        run_window(key, params, force=args.force)


if __name__ == '__main__':
    main()
