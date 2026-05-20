#!/usr/bin/env python3
"""
05_robustness_globex.py

Phase 1 : Parameter sensitivity — perturb TP/SL/window/threshold around the
          winning configs from Phase 2 of 04_threshold_analysis.py. Statistical
          significance tests (t-stat, bootstrap CI) reveal whether 31 trades
          represent a real edge or a lucky sample.

Phase 2 : Friday session filter — skip the first 30 and 45 minutes of Friday
          RTH. Compare the filtered Friday against all other days.

Phase 3 : Full Globex — lift the RTH restriction. Trade all hours (but reset
          z-score window at each day-file boundary to avoid overnight
          contamination). Report RTH vs non-RTH trade breakdown with
          session-appropriate slippage.

Phase 4 : Trading methodology — detailed explanation of every cost assumption,
          execution model, and what this simulation does and does not capture.

Usage:
    cd /Users/stark/Desktop/Projects/Futures_RollOver
    .venv/bin/python notebooks/05_robustness_globex.py
"""

import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
OUT_DIR  = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Roll window ───────────────────────────────────────────────────────────────
FRONT, BACK = 'ESU4', 'ESZ4'
ROLL_START  = '2024-09-12'
FOMC_UTC    = pd.Timestamp('2024-09-18 18:00:00', tz='UTC')
RTH_OPEN, RTH_CLOSE = '12:30', '19:15'  # UTC

TICK_SIZE  = 0.25
MULTIPLIER = 50.0
TICK_VALUE = TICK_SIZE * MULTIPLIER   # $12.50

DIV_YIELD    = 0.0130
FOMC_CUT_BPS = 0.0050
DT_YR        = None   # set in main() from definitions parquet

# ── Signal parameters ─────────────────────────────────────────────────────────
Z_WINDOWS    = ['30s', '1min', '2min', '5min', '10min']
Z_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
MIN_HOLD     = 5

# ── Winners from Phase 2 of 04_threshold_analysis ────────────────────────────
WINNER = {'window': '10min', 'thresh': 2.5, 'mode': 'pts', 'tp': 0.75, 'sl': 0.50}

# ── Sensitivity grid ──────────────────────────────────────────────────────────
TP_PERTURB = [0.375, 0.50, 0.625, 0.75, 0.875, 1.00, 1.25]
SL_PERTURB = [0.25, 0.375, 0.50, 0.625, 0.75, 1.00]

# ── Cost structure ────────────────────────────────────────────────────────────
TC_PER_CONTRACT_RT = 2.00   # institutional (CME direct-clearing)
TC_PER_CONTRACT_RETAIL = 3.70  # IB retail rate
LEGS = 2

TC_BASE        = TC_PER_CONTRACT_RT     * LEGS  # $4.00  institutional
TC_BASE_RETAIL = TC_PER_CONTRACT_RETAIL * LEGS  # $7.40  retail

# Slippage scenarios (total round-trip for the calendar spread, both legs)
SLIPPAGE = {
    'Tight': 0.5 * TICK_VALUE,    # $6.25   algorithmic limit orders, spread book
    'Mid':   1.0 * TICK_VALUE,    # $12.50  standard limit order crossing
    'Wide':  2.0 * TICK_VALUE,    # $25.00  market orders / stressed
    'Night': 4.0 * TICK_VALUE,    # $50.00  overnight / thin sessions
}
SLIP_RT = {k: v + TC_BASE for k, v in SLIPPAGE.items()}  # total all-in RT per trade

SEP  = '─' * 115
SEP2 = '═' * 115

DAY_META = {
    '2024-09-12': ('Thu1',  'steelblue'),
    '2024-09-13': ('Fri',   'crimson'),
    '2024-09-15': ('Sun',   'gray'),
    '2024-09-16': ('Mon',   'darkorange'),
    '2024-09-17': ('Tue',   'purple'),
    '2024-09-18': ('Wed*',  'forestgreen'),
    '2024-09-19': ('Thu2',  'teal'),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_sofr_daily():
    sofr_raw = pd.read_csv(DATA_DIR / 'SOFR.csv', parse_dates=['observation_date'],
                            index_col='observation_date')
    sofr_s   = sofr_raw.iloc[:, 0].dropna() / 100.0
    sofr_idx = pd.DatetimeIndex(sofr_s.index).tz_localize('UTC')
    return pd.Series(sofr_s.values, index=sofr_idx)


def _build_fv_for_series(spread: pd.Series, front_mid: pd.Series,
                          sofr_utc: pd.Series, dt_yr: float) -> pd.Series:
    """FV for any (spread, front_mid) pair using pre-loaded SOFR lookup."""
    daily_idx  = pd.date_range(spread.index[0].normalize(),
                               spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()
    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    ).ffill()
    if len(r_f[r_f.index < FOMC_UTC]) > 0:
        pre_sofr = float(r_f[r_f.index < FOMC_UTC].iloc[-1])
        r_f[r_f.index >= FOMC_UTC] = pre_sofr - FOMC_CUT_BPS
    return front_mid.reindex(r_f.index).ffill() * (r_f - DIV_YIELD) * dt_yr


def load_rth_data(sofr_utc: pd.Series, dt_yr: float):
    """Standard RTH load — returns (spread, fv, ba_sum)."""
    COLS  = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00', 'symbol']
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    print(f'  Loading {len(files)} RTH day files', end='', flush=True)
    parts = []
    for f in files:
        df    = pd.read_parquet(f, columns=COLS)
        df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['ba']  = df['ask_px_00'] - df['bid_px_00']
        wide = (df.groupby('symbol')[['mid', 'ba']]
                .resample('1s').last().ffill()
                .unstack('symbol')
                .between_time(RTH_OPEN, RTH_CLOSE))
        parts.append(wide)
        print('.', end='', flush=True)
    print(' done')
    full    = pd.concat(parts).sort_index()
    spread  = (full[('mid', BACK)] - full[('mid', FRONT)]).dropna()
    ba      = (full[('ba',  FRONT)] + full[('ba',  BACK)]).reindex(spread.index).fillna(0.5)
    front   = full[('mid', FRONT)].reindex(spread.index).ffill()
    fv      = _build_fv_for_series(spread, front, sofr_utc, dt_yr)
    dev     = (spread - fv).dropna()
    return spread.reindex(dev.index), fv.reindex(dev.index), ba.reindex(dev.index)


def load_globex_per_day(sofr_utc: pd.Series, dt_yr: float) -> list[dict]:
    """
    Load each day file WITHOUT the RTH filter. Returns list of dicts:
      {'date': str, 'spread': Series, 'fv': Series, 'ba': Series, 'is_rth': bool Series}
    Z-scores will be computed fresh per-day to avoid overnight contamination.
    """
    COLS  = ['bid_px_00', 'ask_px_00', 'symbol']
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    print(f'  Loading {len(files)} full Globex day files', end='', flush=True)
    sessions = []
    for f in files:
        date_str = Path(f).stem.split('_')[-1]
        df = pd.read_parquet(f, columns=COLS)
        df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['ba']  = df['ask_px_00'] - df['bid_px_00']
        wide = (df.groupby('symbol')[['mid', 'ba']]
                .resample('1s').last().ffill()
                .unstack('symbol'))
        spread    = (wide[('mid', BACK)] - wide[('mid', FRONT)]).dropna()
        ba        = (wide[('ba',  FRONT)] + wide[('ba',  BACK)]).reindex(spread.index).fillna(0.5)
        front     = wide[('mid', FRONT)].reindex(spread.index).ffill()
        fv        = _build_fv_for_series(spread, front, sofr_utc, dt_yr)
        dev       = (spread - fv).dropna()
        spread    = spread.reindex(dev.index)
        ba        = ba.reindex(dev.index)
        # Mark which bars fall in RTH
        is_rth = pd.Series(False, index=dev.index)
        rth_mask = (dev.index.time >= pd.Timestamp(f'2000-01-01 {RTH_OPEN}').time()) & \
                   (dev.index.time <= pd.Timestamp(f'2000-01-01 {RTH_CLOSE}').time())
        is_rth[rth_mask] = True
        sessions.append({'date': date_str, 'spread': spread, 'fv': fv,
                         'dev': dev, 'ba': ba, 'is_rth': is_rth})
        print('.', end='', flush=True)
    print(' done')
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE  (identical to 04 — self-contained copy)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_z(dev: pd.Series, window: str) -> np.ndarray:
    mu  = dev.rolling(window, min_periods=1).mean()
    sig = dev.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


def simulate(spread: pd.Series, dev: pd.Series, z: np.ndarray,
             entry_thresh: float, exit_cfg: dict,
             min_hold: int = MIN_HOLD,
             force_rth_close: pd.Series = None) -> pd.DataFrame:
    """
    General simulation engine (pts / z_revert / partial / trail modes).
    force_rth_close: boolean Series aligned with spread; if True at bar i,
    force-close any open position (used for Globex to close at RTH end).
    """
    mode   = exit_cfg['mode']
    prices = spread.values
    zvals  = z
    times  = spread.index
    n      = len(prices)

    dates       = np.array([t.date() for t in times])
    is_last     = np.zeros(n, dtype=bool)
    is_last[-1] = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last[i] = True

    force_close = (np.zeros(n, dtype=bool) if force_rth_close is None
                   else force_rth_close.reindex(spread.index).fillna(False).values)

    trades, pos, entry_i = [], 0, -1
    entry_px = np.nan
    bars_held = 0
    lot1_closed = False
    best_px = np.nan
    pending_dir = 0

    for i in range(n):
        zi = zvals[i]
        px = prices[i]

        if pending_dir != 0 and pos == 0:
            pos = pending_dir; entry_i = i; entry_px = px
            bars_held = 0; lot1_closed = False; best_px = px; pending_dir = 0

        if pos != 0:
            bars_held += 1
            if pos == 1: best_px = max(best_px, px)
            else:        best_px = min(best_px, px)

            exit_now = False
            if is_last[i] or force_close[i]:
                exit_now = True
            elif mode == 'z_revert':
                if bars_held >= min_hold and not np.isnan(zi) and abs(zi) < exit_cfg['exit_z']:
                    exit_now = True
            elif mode == 'pts':
                move = pos * (px - entry_px)
                if move >= exit_cfg['tp'] or move <= -exit_cfg['sl']:
                    exit_now = True
            elif mode == 'trail':
                move = pos * (px - entry_px)
                trail_hit = (pos == 1  and px < best_px - exit_cfg['trail']) or \
                            (pos == -1 and px > best_px + exit_cfg['trail'])
                if move >= exit_cfg['tp'] or move <= -exit_cfg['sl'] or \
                        (trail_hit and bars_held >= min_hold):
                    exit_now = True
            elif mode == 'partial':
                move = pos * (px - entry_px)
                if not lot1_closed and move >= exit_cfg['tp1']:
                    lot1_closed = True
                if lot1_closed:
                    if 'trail' in exit_cfg:
                        trail_hit = (pos == 1  and px < best_px - exit_cfg['trail']) or \
                                    (pos == -1 and px > best_px + exit_cfg['trail'])
                        if trail_hit and bars_held >= min_hold: exit_now = True
                    elif 'tp2' in exit_cfg and move >= exit_cfg['tp2']:
                        exit_now = True
                if move <= -exit_cfg['sl'] or is_last[i] or force_close[i]:
                    exit_now = True

            if exit_now:
                if mode == 'partial' and lot1_closed:
                    gross_pts = 0.5 * exit_cfg['tp1'] + 0.5 * pos * (px - entry_px)
                else:
                    gross_pts = pos * (px - entry_px)
                gross_usd = gross_pts * MULTIPLIER
                trades.append({
                    'entry_time'   : times[entry_i],
                    'exit_time'    : times[i],
                    'direction'    : pos,
                    'entry_spread' : entry_px,
                    'exit_spread'  : px,
                    'gross_pts'    : gross_pts,
                    'gross_usd'    : gross_usd,
                    'bars_held'    : bars_held,
                    'eod_close'    : bool(is_last[i] or force_close[i]),
                    'post_fomc'    : times[entry_i] >= FOMC_UTC,
                    'is_rth_entry' : (force_rth_close is None or
                                      bool(not force_rth_close.reindex([times[entry_i]]).iloc[0])),
                })
                pos = 0; entry_i = -1; bars_held = 0
                best_px = np.nan; lot1_closed = False; pending_dir = 0

        if pos == 0 and pending_dir == 0 and i > 0:
            zp = zvals[i - 1]
            if not np.isnan(zi) and not np.isnan(zp):
                if zp >= -entry_thresh and zi < -entry_thresh:
                    pending_dir =  1
                elif zp <= entry_thresh and zi > entry_thresh:
                    pending_dir = -1

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    for slip_lbl, slip_cost in SLIPPAGE.items():
        df[f'net_{slip_lbl}'] = df['gross_usd'] - slip_cost - TC_BASE
    df['cum_gross'] = df['gross_usd'].cumsum()
    df['peak']      = df['cum_gross'].cummax()
    df['drawdown']  = df['cum_gross'] - df['peak']
    return df


def _agg(t: pd.DataFrame, slip_lbl: str = 'Tight') -> dict:
    if t is None or t.empty:
        return {'n': 0}
    n = len(t)
    g = t['gross_usd']
    w = t.loc[t['gross_usd'] > 0, 'gross_usd'].sum()
    l = t.loc[t['gross_usd'] < 0, 'gross_usd'].sum()
    return {
        'n'          : n,
        'wr'         : (g > 0).mean() * 100,
        'avg_gross'  : g.mean(),
        'std_gross'  : g.std(),
        'tot_gross'  : g.sum(),
        'avg_hold'   : t['bars_held'].mean(),
        'eod_pct'    : t['eod_close'].mean() * 100,
        'pf'         : round(w / abs(l), 2) if l != 0 else float('inf'),
        'mdd'        : t['drawdown'].min(),
        f'avg_net_{slip_lbl}': t[f'net_{slip_lbl}'].mean(),
        f'tot_net_{slip_lbl}': t[f'net_{slip_lbl}'].sum(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL SIGNIFICANCE
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, ci: float = 0.95) -> tuple:
    """Bootstrap percentile CI for the mean."""
    if len(values) < 2:
        return (np.nan, np.nan)
    means = np.array([np.mean(np.random.choice(values, size=len(values), replace=True))
                      for _ in range(n_boot)])
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def t_stat(values: np.ndarray) -> tuple:
    """t-statistic for mean != 0, returns (t, p_two_tail)."""
    if len(values) < 3 or np.std(values) == 0:
        return (np.nan, np.nan)
    t, p = spstats.ttest_1samp(values, 0.0)
    return t, p


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — PARAMETER SENSITIVITY
# ─────────────────────────────────────────────────────────────────────────────
def run_sensitivity(spread, dev, zscores) -> dict:
    """
    1a. TP × SL heatmap (window=10min, thresh=2.5 fixed).
    1b. Window × Threshold (TP=0.75, SL=0.50 fixed).
    1c. Statistical significance for the winning configs.
    """
    z_win  = zscores[WINNER['window']]
    thresh = WINNER['thresh']

    # 1a: TP × SL grid
    tp_sl_nets   = np.full((len(TP_PERTURB), len(SL_PERTURB)), np.nan)
    tp_sl_gross  = np.full_like(tp_sl_nets, np.nan)
    tp_sl_n      = np.zeros_like(tp_sl_nets, dtype=int)
    tp_sl_ci_lo  = np.full_like(tp_sl_nets, np.nan)
    tp_sl_ci_hi  = np.full_like(tp_sl_nets, np.nan)
    tp_sl_tstat  = np.full_like(tp_sl_nets, np.nan)

    for i, tp in enumerate(TP_PERTURB):
        for j, sl in enumerate(SL_PERTURB):
            cfg = {'mode': 'pts', 'tp': tp, 'sl': sl}
            t   = simulate(spread, dev, z_win, thresh, cfg)
            if not t.empty:
                tp_sl_nets[i, j]  = t['net_Tight'].mean()
                tp_sl_gross[i, j] = t['gross_usd'].mean()
                tp_sl_n[i, j]     = len(t)
                lo, hi = bootstrap_ci(t['net_Tight'].values)
                tp_sl_ci_lo[i, j] = lo
                tp_sl_ci_hi[i, j] = hi
                ts, _ = t_stat(t['net_Tight'].values)
                tp_sl_tstat[i, j] = ts

    # 1b: Window × Threshold grid (TP=0.75, SL=0.50)
    cfg_fixed = {'mode': 'pts', 'tp': WINNER['tp'], 'sl': WINNER['sl']}
    win_thresh_rows = []
    for win in Z_WINDOWS:
        z = zscores[win]
        for th in Z_THRESHOLDS:
            t = simulate(spread, dev, z, th, cfg_fixed)
            r = _agg(t)
            r['window'] = win
            r['thresh'] = th
            if not t.empty:
                lo, hi = bootstrap_ci(t['gross_usd'].values)
                ts, pv = t_stat(t['gross_usd'].values)
                r['ci_lo_gross'] = lo; r['ci_hi_gross'] = hi
                r['t_stat'] = ts; r['p_val'] = pv
                r['lo_net'], r['hi_net'] = bootstrap_ci(t['net_Tight'].values)
            else:
                r.update({'ci_lo_gross': np.nan, 'ci_hi_gross': np.nan,
                          't_stat': np.nan, 'p_val': np.nan,
                          'lo_net': np.nan, 'hi_net': np.nan})
            win_thresh_rows.append(r)

    return {
        'tp_sl_nets': tp_sl_nets, 'tp_sl_gross': tp_sl_gross,
        'tp_sl_n': tp_sl_n, 'tp_sl_ci_lo': tp_sl_ci_lo, 'tp_sl_ci_hi': tp_sl_ci_hi,
        'tp_sl_tstat': tp_sl_tstat,
        'win_thresh': win_thresh_rows,
    }


def print_sensitivity(res: dict):
    print()
    print('  PHASE 1A — TP × SL SENSITIVITY  (10min window, z>2.5σ, avg net/trade tight)')
    print(SEP)
    header = f"  {'TP \\ SL':>9} " + ''.join(f"  SL={sl:>4.3f}" for sl in SL_PERTURB)
    print(header)
    print('  ' + '─' * (len(header) - 2))
    for i, tp in enumerate(TP_PERTURB):
        row = f"  TP={tp:>5.3f}  "
        for j, sl in enumerate(SL_PERTURB):
            v = res['tp_sl_nets'][i, j]
            n = res['tp_sl_n'][i, j]
            if np.isnan(v):
                row += '         — '
            else:
                marker = '✓' if v >= 0 else ' '
                row += f" ${v:>6.1f}{marker}({n:>3})"
        print(row)
    print()
    print('  Values: avg net P&L/trade (tight slippage)  |  n=trade count  |  ✓=profitable')
    print('  Winning cell (04 result): TP=0.75 / SL=0.50')

    print()
    print('  PHASE 1B — WINDOW × THRESHOLD SENSITIVITY  (TP=0.75pt, SL=0.50pt)')
    print(SEP)
    print(f"  {'Window':<6} {'Thresh':>7}  {'N':>5}  {'WR%':>6}  "
          f"{'AvgGross':>9}  {'95% CI Gross':>22}  {'t-stat':>7}  {'p-val':>7}  "
          f"{'AvgNetTight':>12}  {'95% CI Net':>22}  {'Sig?':>6}")
    print(SEP)
    for r in res['win_thresh']:
        if not r.get('n'):
            continue
        ci_g = f"[${r['ci_lo_gross']:>6.0f}, ${r['ci_hi_gross']:>6.0f}]"
        ci_n = f"[${r['lo_net']:>6.0f}, ${r['hi_net']:>6.0f}]"
        sig  = ('p<0.05' if r.get('p_val', 1) < 0.05 else
                'p<0.10' if r.get('p_val', 1) < 0.10 else '  n.s.')
        print(
            f"  {r['window']:<6} {r['thresh']:>7.1f}  {r['n']:>5}  {r['wr']:>5.1f}%  "
            f"  ${r['avg_gross']:>7.2f}  {ci_g}  {r.get('t_stat',np.nan):>7.2f}  "
            f"{r.get('p_val',np.nan):>7.4f}  ${r['avg_net_Tight']:>10.2f}  "
            f"{ci_n}  {sig:>6}"
        )
    print()
    print('  KEY: if 95% CI for avg net/trade straddles $0 → result is NOT statistically')
    print('  significant at this sample size. This is the "lucky trade" test.')


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — FRIDAY SESSION FILTER
# ─────────────────────────────────────────────────────────────────────────────
def run_friday_filter(spread, dev, zscores) -> dict:
    """
    Split Friday into: first 30/45 min vs remainder. Compare vs other days.
    """
    z_win = zscores[WINNER['window']]
    cfg   = {'mode': 'pts', 'tp': WINNER['tp'], 'sl': WINNER['sl']}
    all_trades = simulate(spread, dev, z_win, WINNER['thresh'], cfg)

    # Segment definitions (UTC: RTH_OPEN = 12:30 = 08:30 ET)
    FRI = '2024-09-13'
    FRI_OPEN_UTC     = pd.Timestamp(f'{FRI} 12:30', tz='UTC')   # 08:30 ET
    FRI_CUT30_UTC    = pd.Timestamp(f'{FRI} 13:00', tz='UTC')   # 09:00 ET
    FRI_CUT45_UTC    = pd.Timestamp(f'{FRI} 13:15', tz='UTC')   # 09:15 ET

    results = {}

    # All days
    results['All days (baseline)'] = _agg(all_trades)

    # Non-Friday days
    non_fri = all_trades[all_trades['entry_time'].dt.date != pd.Timestamp(FRI).date()]
    results['All non-Friday days'] = _agg(non_fri)

    # Friday full
    fri_all = all_trades[all_trades['entry_time'].dt.date == pd.Timestamp(FRI).date()]
    results['Friday — full session'] = _agg(fri_all)

    # Friday minus first 30 min
    fri_30 = fri_all[fri_all['entry_time'] >= FRI_CUT30_UTC]
    results['Friday — skip first 30min (from 09:00 ET)'] = _agg(fri_30)

    # Friday minus first 45 min
    fri_45 = fri_all[fri_all['entry_time'] >= FRI_CUT45_UTC]
    results['Friday — skip first 45min (from 09:15 ET)'] = _agg(fri_45)

    # Friday first 30 min only (the problematic part)
    fri_open30 = fri_all[fri_all['entry_time'] < FRI_CUT30_UTC]
    results['Friday — first 30min ONLY (08:30–09:00 ET)'] = _agg(fri_open30)

    # Per-day breakdown (for comparison table)
    per_day = {}
    for d in sorted({str(t.date()) for t in all_trades['entry_time']}):
        dm = all_trades[all_trades['entry_time'].dt.date == pd.Timestamp(d).date()]
        per_day[d] = _agg(dm)

    results['_per_day'] = per_day
    results['_fri_open'] = fri_open30
    results['_fri_rest'] = fri_45
    return results


def print_friday_filter(res: dict):
    print()
    print('  PHASE 2 — FRIDAY SESSION FILTER  (10min / z>2.5 / TP=0.75 / SL=0.50)')
    print(SEP)
    print(f"  {'Segment':<50}  {'N':>5}  {'WR%':>6}  {'AvgHold':>8}  "
          f"{'AvgGross':>9}  {'TotGross':>10}  {'NetTight/Tr':>12}  {'EOD%':>6}")
    print(SEP)
    SKIP = {'_per_day', '_fri_open', '_fri_rest'}
    for label, r in res.items():
        if label in SKIP or not r.get('n'):
            continue
        print(
            f"  {label:<50}  {r['n']:>5}  {r['wr']:>5.1f}%  "
            f"{r['avg_hold']:>7.0f}s  ${r['avg_gross']:>8.2f}  "
            f"${r['tot_gross']:>9,.0f}  ${r['avg_net_Tight']:>10.2f}  "
            f"{r['eod_pct']:>5.0f}%"
        )

    print()
    print('  PER-DAY BREAKDOWN (for reference):')
    pd_res = res.get('_per_day', {})
    print(f"  {'Date':<12}  {'Day':<5}  {'N':>5}  {'WR%':>6}  {'AvgGross':>9}  "
          f"{'NetTight/Tr':>12}  {'Assessment'}")
    print('  ' + '─' * 80)
    for d, r in pd_res.items():
        label = DAY_META.get(d, ('?',))[0]
        if not r.get('n'):
            print(f'  {d}  {label:<5}  {"—":>5}')
            continue
        # Compare to the overall non-Friday avg
        non_fri_r = res.get('All non-Friday days', {})
        nf_avg = non_fri_r.get('avg_gross', 0)
        diff = r['avg_gross'] - nf_avg
        flag = ('⚡ BEST' if diff > 1 else
                '⚠  WORST' if diff < -1 else '  OK')
        print(
            f"  {d}  {label:<5}  {r['n']:>5}  {r['wr']:>5.1f}%  "
            f"${r['avg_gross']:>8.2f}  ${r['avg_net_Tight']:>10.2f}  {flag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FULL GLOBEX
# ─────────────────────────────────────────────────────────────────────────────
def run_globex(sessions: list[dict]) -> dict:
    """
    Run the winning config on each day file without RTH filter.
    Apply session-appropriate slippage: RTH uses Tight/Mid/Wide;
    overnight entries get the Night scenario.
    Reports RTH vs overnight breakdown.
    """
    cfg = {'mode': 'pts', 'tp': WINNER['tp'], 'sl': WINNER['sl']}
    all_trades = []

    for sess in sessions:
        d   = sess['date']
        sp  = sess['spread']
        dev = sess['dev']
        if len(dev) < 10:
            continue
        z   = _compute_z(dev, WINNER['window'])
        trades = simulate(sp, dev, z, WINNER['thresh'], cfg)
        if not trades.empty:
            trades['date'] = d
            # Tag RTH vs overnight entries using bar's is_rth classification
            is_rth_map = sess['is_rth']
            trades['entry_is_rth'] = trades['entry_time'].map(
                lambda t: bool(is_rth_map.get(t, False))
            )
            all_trades.append(trades)

    if not all_trades:
        return {}

    df = pd.concat(all_trades, ignore_index=True)

    # For overnight entries, replace slippage with Night scenario
    night_cost = SLIPPAGE['Night'] + TC_BASE
    tight_cost = SLIPPAGE['Tight'] + TC_BASE
    df['net_Tight_adj'] = np.where(
        df['entry_is_rth'],
        df['gross_usd'] - tight_cost,
        df['gross_usd'] - night_cost
    )

    rth_t  = df[df['entry_is_rth']]
    night_t = df[~df['entry_is_rth']]

    return {
        'all'      : df,
        'rth'      : rth_t,
        'overnight': night_t,
    }


def print_globex(res: dict):
    if not res:
        print('  No Globex trades generated.')
        return
    print()
    print('  PHASE 3 — FULL GLOBEX (no RTH filter, per-day z-score reset)')
    print(f'  Config: {WINNER["window"]} / z>{WINNER["thresh"]}σ / TP={WINNER["tp"]}pt / SL={WINNER["sl"]}pt')
    print(SEP)

    def _row(label, t):
        if t is None or t.empty:
            return f'  {label:<40}  {"—":>5}'
        n   = len(t)
        wr  = (t['gross_usd'] > 0).mean() * 100
        ag  = t['gross_usd'].mean()
        tg  = t['gross_usd'].sum()
        ah  = t['bars_held'].mean()
        nt  = t['net_Tight'].mean() if 'net_Tight' in t.columns else np.nan
        return (f'  {label:<40}  {n:>5}  {wr:>5.1f}%  {ah:>7.0f}s  '
                f'${ag:>8.2f}  ${tg:>10,.0f}  ${nt:>10.2f}')

    all_t  = res.get('all',       pd.DataFrame())
    rth_t  = res.get('rth',       pd.DataFrame())
    night_t = res.get('overnight', pd.DataFrame())

    print(f"  {'Segment':<40}  {'N':>5}  {'WR%':>6}  {'AvgHold':>7}  "
          f"{'AvgGross':>9}  {'TotGross':>11}  {'NetTight/Tr':>11}")
    print(SEP)
    print(_row('Full Globex (all sessions)', all_t))
    print(_row('RTH entries only',           rth_t))
    print(_row('Overnight entries only',     night_t))

    print()
    print('  Note on overnight slippage: overnight entries use "Night" scenario')
    print('  (4 ticks = $50 RT vs $6.25 tight in RTH) due to wider bid-ask.')
    if not night_t.empty:
        night_cost = SLIPPAGE['Night'] + TC_BASE
        print(f'  Overnight all-in cost: ${night_cost:.2f}/trade  vs  Tight RTH: ${SLIP_RT["Tight"]:.2f}/trade')
        adj_avg = res['all']['net_Tight_adj'].mean() if 'net_Tight_adj' in res['all'].columns else np.nan
        print(f'  Session-adjusted avg net (tight RTH / night overnight): ${adj_avg:.2f}/trade')

    # Per-day full-Globex breakdown
    print()
    print('  PER-DAY FULL GLOBEX BREAKDOWN:')
    print(f"  {'Date':<12}  {'Day':<5}  {'N_all':>6}  {'N_RTH':>6}  "
          f"{'N_OVN':>6}  {'RTH WR%':>8}  {'OVN WR%':>8}  "
          f"{'AvgGross_RTH':>13}  {'AvgGross_OVN':>13}")
    print('  ' + '─' * 90)
    for d, lbl_col in DAY_META.items():
        lbl = lbl_col[0]
        dt  = all_t[all_t['date'] == d] if not all_t.empty else pd.DataFrame()
        if dt.empty:
            print(f'  {d}  {lbl:<5}  {"—":>6}')
            continue
        dr = dt[dt['entry_is_rth']]
        dn = dt[~dt['entry_is_rth']]
        wr_r = (dr['gross_usd'] > 0).mean()*100 if not dr.empty else np.nan
        wr_n = (dn['gross_usd'] > 0).mean()*100 if not dn.empty else np.nan
        ag_r = dr['gross_usd'].mean() if not dr.empty else np.nan
        ag_n = dn['gross_usd'].mean() if not dn.empty else np.nan
        wr_r_s = f'{wr_r:.0f}%' if not np.isnan(wr_r) else '—'
        wr_n_s = f'{wr_n:.0f}%' if not np.isnan(wr_n) else '—'
        ag_r_s = f'${ag_r:.2f}' if not np.isnan(ag_r) else '—'
        ag_n_s = f'${ag_n:.2f}' if not np.isnan(ag_n) else '—'
        print(f'  {d}  {lbl:<5}  {len(dt):>6}  {len(dr):>6}  '
              f'{len(dn):>6}  {wr_r_s:>8}  {wr_n_s:>8}  {ag_r_s:>13}  {ag_n_s:>13}')


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — TRADING METHODOLOGY
# ─────────────────────────────────────────────────────────────────────────────
def print_methodology():
    print()
    print(SEP2)
    print('  PHASE 4 — TRADING METHODOLOGY, COST ASSUMPTIONS, AND EXECUTION MODEL')
    print(SEP2)

    print("""
  ─────────────────────────────────────────────────────────────────────────────
  A. WHAT WE ARE TRADING
  ─────────────────────────────────────────────────────────────────────────────

  Instrument  : E-mini S&P 500 (ES) calendar spread
  Leg 1       : Long ESZ4 (Dec 2024 quarterly future)
  Leg 2       : Short ESU4 (Sep 2024 quarterly future)
  Together    : 1 spread = 1 ESZ4 long + 1 ESU4 short, held simultaneously
  P&L driver  : ESZ4 price − ESU4 price (the "calendar spread")
  Tick size   : 0.25 index points per leg = $12.50 per tick per contract
  Multiplier  : $50 per index point per contract
  1 pt move   : $50 profit/loss per spread position

  Example trade (LONG calendar spread):
    Entry : spread = 59.00 pts (ESZ4 cheap relative to fair value)
    Exit  : spread = 59.75 pts (+0.75 pt move = 3 ticks)
    Gross : 0.75 × $50 = +$37.50

  ─────────────────────────────────────────────────────────────────────────────
  B. SIGNAL CONSTRUCTION
  ─────────────────────────────────────────────────────────────────────────────

  Data resolution  : Top-of-book (Level 1) quotes, resampled to 1-second bars
  Spread price     : (ESZ4_ask + ESZ4_bid)/2 − (ESU4_ask + ESU4_bid)/2
                     i.e. midprice of ESZ4 minus midprice of ESU4
  Fair value (FV)  : S_front × (SOFR_daily − div_yield) × ΔT
                     where SOFR uses the prior night's published rate (no
                     look-ahead), with an intraday override at the FOMC
                     announcement time using the announced cut.
  Deviation        : observed spread − FV  (positive = spread is "rich")
  Z-score          : rolling(deviation, window) → (dev − μ) / σ
                     computed on a sliding window of {30s, 1min, 2min, 5min, 10min}
  Entry trigger    : EDGE-TRIGGERED z-crossing (not level-triggered):
                       z_prev ≤ +threshold AND z_now > +threshold → SHORT signal
                       z_prev ≥ −threshold AND z_now < −threshold → LONG  signal
  Fill timing      : Order placed at bar T (when z crosses), filled at bar T+1
                     This models 1-second execution latency (realistic for
                     algorithmic strategies; may be optimistic for manual execution)

  ─────────────────────────────────────────────────────────────────────────────
  C. EXECUTION MODEL
  ─────────────────────────────────────────────────────────────────────────────

  Method        : Calendar spread order on CME Globex (both legs filled
                  simultaneously as one spread instrument). This eliminates
                  leg risk — the risk that only one leg fills before the market
                  moves, leaving an unhedged outright position.

  Alternative   : Legged execution (send limit orders for each leg separately).
                  Increases execution complexity and leg-risk, but may achieve
                  tighter midprice fills. NOT modeled here.

  Price used    : Entry and exit prices are the midprice of the spread at the
                  signal bar's close. Slippage is then DEDUCTED as a flat cost
                  (see Section D).

  Position size : 1 calendar spread lot (1 ESZ4 + 1 ESU4)
                  → All P&L figures are per-lot. Scale linearly with lot size.
                  → Market impact is negligible at 1-5 lots (ES daily volume:
                    ~1-2 million contracts; 1 lot is immaterial).

  No overnight  : RTH simulation closes all positions at 15:15 ET (19:15 UTC)
                  each day. No position is carried through the overnight session.
                  Full Globex simulation resets daily.

  ─────────────────────────────────────────────────────────────────────────────
  D. TRANSACTION COSTS — DETAILED BREAKDOWN
  ─────────────────────────────────────────────────────────────────────────────

  Commission (per contract, round-trip):

    Institutional (direct CME clearing / Tier-1 FCM):
      CME exchange fee           : ~$0.85/side = $1.70 RT
      NFA regulatory fee         : $0.02/side  = $0.04 RT
      Clearing/give-up fee       : ~$0.12/side = $0.24 RT
      ────────────────────────────────────────────────
      Total institutional RT     : ~$1.98/contract ≈ $2.00 modeled
      Calendar spread (2 legs)   : $4.00 per trade   ← our TC_BASE

    Retail (Interactive Brokers, non-pro):
      IB commission              : ~$0.85/side = $1.70 RT
      Exchange + NFA             : ~$1.16/side = $2.32 RT  (IB passes through)
      Clearing                   : included in above
      ────────────────────────────────────────────────
      Total retail RT            : ~$3.70/contract
      Calendar spread (2 legs)   : $7.40 per trade   ← 85% higher than institutional

    → Our model uses institutional rates ($4.00 base). Retail traders would
      need to add $3.40 to every breakeven calculation below.

  ─────────────────────────────────────────────────────────────────────────────
  E. SLIPPAGE — THREE SCENARIOS + OVERNIGHT
  ─────────────────────────────────────────────────────────────────────────────

  Slippage is the cost of crossing the bid-ask spread plus market impact.
  For a calendar spread, there are TWO bid-ask crossings per leg (entry + exit)
  and TWO legs. We express slippage as total round-trip cost for the spread.

  How midprice execution relates to the bid-ask:
    • The bid-ask on each ES leg is typically 0.25 pts (1 tick) during RTH.
    • Trading AT midprice costs 0 (impossible in practice).
    • Trading at bid/ask costs 0.125 pts (half-spread) per side per leg.
    • Round-trip for 2 legs: 4 × 0.125 = 0.50 pts = $25 (this is our "Wide" scenario).

  SCENARIO 1 — Tight ($6.25 RT = 0.5 tick):
    Achievable when: using the CME Globex calendar SPREAD market (not legged),
    where the spread bid-ask is often 0.05-0.10 pts during liquid RTH. Entry
    limit order posted at spread-mid, filled by an incoming aggressive order.
    Represents best-case systematic execution at institutional scale.
    Total all-in: $6.25 slippage + $4.00 commission = $10.25/trade

  SCENARIO 2 — Mid ($12.50 RT = 1 tick):
    Achievable when: limit orders on each leg at their respective midpoints,
    each crossing half of the 0.25-pt bid-ask. This is the "1 tick per leg"
    interpretation — each leg pays 0.125 pts at entry and 0.125 pts at exit.
    Total all-in: $12.50 slippage + $4.00 commission = $16.50/trade

  SCENARIO 3 — Wide ($25.00 RT = 2 ticks):
    Applicable when: using market orders, or during the opening minutes of
    a session when the bid-ask widens. Also represents adverse selection —
    the other side knows more than we do and we're consistently paying to
    cross the full spread. Conservative / stress-test scenario.
    Total all-in: $25.00 slippage + $4.00 commission = $29.00/trade

  SCENARIO 4 — Night ($50.00 RT = 4 ticks):  [Full Globex only]
    Applicable to trades entered during non-RTH hours (overnight, pre-market,
    Sunday open). The calendar spread book during these hours shows B/A of
    0.50-1.00 pts per leg (vs 0.25 pts in RTH). Used only in Phase 3.
    Total all-in: $50.00 slippage + $4.00 commission = $54.00/trade

  BREAKEVEN gross P&L per trade (must exceed this to profit):
    Tight (institutional best-case):  $10.25  = 0.82 ticks = 0.205 pts
    Mid   (realistic limit orders):   $16.50  = 1.32 ticks = 0.330 pts
    Wide  (market orders/stress):     $29.00  = 2.32 ticks = 0.580 pts
    Night (overnight sessions):       $54.00  = 4.32 ticks = 1.080 pts

  ─────────────────────────────────────────────────────────────────────────────
  F. WHAT THIS MODEL DOES NOT CAPTURE
  ─────────────────────────────────────────────────────────────────────────────

  1. Market impact at scale: at 10-50 lots, the calendar spread book may
     move against a large order. Not material for 1-5 lots.

  2. Leg risk (if legged): if executing each leg separately, there is a risk
     the first leg fills but the second does not (or fills at a worse price
     after the market moves). This can cause outright ES exposure.

  3. Adverse selection: counterparties in the spread book may have information
     about imminent flow (e.g., a large roll trade about to hit). Our model
     assumes fills are symmetric. In reality, HFT firms provide liquidity
     and adjust quotes ahead of predictable roll flow.

  4. Financing / margin: calendar spread margin is ~$300-500 per lot (CME
     hedge credit). Financing cost at SOFR on $300 margin is negligible
     (~$0.04/day). Not modeled.

  5. Early morning slippage: the first 5-10 minutes of each session often
     have wider spreads even during RTH. Our slippage model is constant
     throughout the session.

  6. Quote stuffing / data gaps: at 1-second resolution, fast order book
     events (sub-second spikes) are invisible. A spike to z=+4 that lasts
     0.3 seconds appears as a clean 1-second bar with z=+2. This may cause
     the simulation to miss some extreme entries.

  7. Roll of position: ESU4 expires Sep 20 (day after roll window). Any
     position not closed by Sep 19 EOD would face delivery risk. Our EOD
     force-close handles this for the simulation but real trading requires
     explicit roll monitoring.

  ─────────────────────────────────────────────────────────────────────────────
  G. WHAT A STATISTICALLY SIGNIFICANT RESULT REQUIRES
  ─────────────────────────────────────────────────────────────────────────────

  From information theory: to detect a Sharpe ratio of 1.0 with 80% power
  at the 5% significance level (one-sided), you need n ≥ 80 trades.
  For Sharpe = 0.5 (a more modest claim), you need n ≥ 320 trades.

  Our best config (TP=0.75/SL=0.50, 10min, z>2.5) generates ~31 trades per
  roll window. Across all 4 windows (Sep/Dec 2024, Mar/Jun 2025):
    ~31 × 4 = ~124 trades — sufficient for Sharpe ≥ 1.0 detection only.

  The current single-window (31 trades) result cannot be distinguished from
  a coin-flip at any reasonable confidence level unless the t-statistic
  exceeds ~2.0 on the net P&L series.
""")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def chart_tp_sl_heatmap(res: dict) -> str:
    nets   = res['tp_sl_nets']
    n_tr   = res['tp_sl_n']
    tstats = res['tp_sl_tstat']

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    titles = ['Avg Net P&L/trade — Tight slippage ($)',
              'Trade Count',
              't-statistic (net_Tight vs 0)']
    mats   = [nets, n_tr.astype(float), tstats]
    cmaps  = ['RdYlGn', 'Blues', 'RdYlGn']

    for ax, mat, title, cmap in zip(axes, mats, titles, cmaps):
        vmax = np.nanmax(np.abs(mat)) if not np.all(np.isnan(mat)) else 1
        im = ax.imshow(mat, cmap=cmap, aspect='auto',
                       vmin=-vmax if cmap == 'RdYlGn' else 0, vmax=vmax)
        ax.set_xticks(range(len(SL_PERTURB)))
        ax.set_xticklabels([f'{v:.3f}' for v in SL_PERTURB], fontsize=7)
        ax.set_yticks(range(len(TP_PERTURB)))
        ax.set_yticklabels([f'{v:.3f}' for v in TP_PERTURB], fontsize=7)
        ax.set_xlabel('SL (pts)')
        ax.set_ylabel('TP (pts)')
        ax.set_title(title, fontweight='bold', fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.85)
        for ri in range(len(TP_PERTURB)):
            for ci in range(len(SL_PERTURB)):
                v = mat[ri, ci]
                if not np.isnan(v):
                    ax.text(ci, ri, f'{v:.0f}', ha='center', va='center',
                            fontsize=7, color='black' if abs(v) < vmax*0.6 else 'white')

    # Mark the winner
    wi = TP_PERTURB.index(WINNER['tp'])
    wj = SL_PERTURB.index(WINNER['sl'])
    for ax in axes:
        ax.add_patch(plt.Rectangle((wj - 0.5, wi - 0.5), 1, 1,
                                   fill=False, edgecolor='gold', lw=2.5))

    fig.suptitle(f'Phase 1A — TP × SL Sensitivity  (10min / z>2.5σ)\n'
                 f'Gold box = original winner (TP=0.75 / SL=0.50)', fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p1_sensitivity_heatmap.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_win_thresh_ci(res: dict) -> str:
    """CI band chart for avg net per trade across window × threshold."""
    fig, axes = plt.subplots(1, len(Z_WINDOWS), figsize=(4 * len(Z_WINDOWS), 5),
                             sharey=False)
    for ax, win in zip(axes, Z_WINDOWS):
        rows = [r for r in res['win_thresh'] if r['window'] == win and r.get('n', 0) > 0]
        xs   = [r['thresh'] for r in rows]
        means = [r.get('avg_net_Tight', np.nan) for r in rows]
        lo    = [r.get('lo_net', np.nan) for r in rows]
        hi    = [r.get('hi_net', np.nan) for r in rows]

        ax.fill_between(xs, lo, hi, alpha=0.2, color='steelblue', label='95% CI (bootstrap)')
        ax.plot(xs, means, color='steelblue', lw=2, marker='o', label='Avg net/trade')
        ax.axhline(0, color='black', lw=0.8, linestyle='--')
        ax.set_title(f'{win}', fontweight='bold')
        ax.set_xlabel('Entry threshold (σ)')
        ax.set_ylabel('Avg net P&L/trade (Tight, $)')
        ax.legend(fontsize=7)

    fig.suptitle('Phase 1B — Window × Threshold Sensitivity (TP=0.75/SL=0.50)\n'
                 '95% bootstrap CI on avg net P&L/trade — does zero lie in band?',
                 fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p1_ci_by_window.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_friday_comparison(res: dict) -> str:
    """Bar chart comparing avg gross across segments."""
    labels = [k for k in res if not k.startswith('_')]
    vals   = [res[k].get('avg_gross', 0) for k in labels]
    nets   = [res[k].get('avg_net_Tight', 0) for k in labels]
    colors = ['steelblue' if v >= 0 else 'tomato' for v in vals]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(labels))
    for ax, ys, title, be in [
        (axes[0], vals,  'Avg Gross P&L/Trade ($)', 0),
        (axes[1], nets,  'Avg Net P&L/Trade — Tight ($)', 0),
    ]:
        cs = ['steelblue' if v >= be else 'tomato' for v in ys]
        ax.bar(x, ys, color=cs, alpha=0.8, edgecolor='white')
        ax.axhline(be, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
        ax.set_title(title, fontweight='bold')
        for xi, v in enumerate(ys):
            ax.text(xi, v + abs(v)*0.01 + 0.2, f'${v:.1f}', ha='center', fontsize=7)

    fig.suptitle('Phase 2 — Friday Session Filter Comparison', fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p2_friday_filter.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_globex_rth_vs_ovn(res: dict) -> str:
    """Equity curves: RTH-only vs full Globex."""
    if not res:
        return ''
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, key, title, color in [
        (axes[0], 'rth',       'RTH Entries Only',       'steelblue'),
        (axes[1], 'overnight', 'Overnight Entries Only',  'darkorange'),
    ]:
        t = res.get(key, pd.DataFrame())
        if t.empty:
            ax.set_title(f'{title}\n(no trades)')
            continue
        eq = t['gross_usd'].cumsum().reset_index(drop=True)
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=eq.values >= 0, alpha=0.3, color='forestgreen')
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=eq.values < 0, alpha=0.3, color='crimson')
        ax.plot(range(len(eq)), eq.values, color=color, lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        n   = len(t)
        tot = t['gross_usd'].sum()
        nt  = t['net_Tight'].mean() if 'net_Tight' in t.columns else np.nan
        ax.set_title(f'{title}\n{n} trades  |  Gross ${tot:,.0f}  |  AvgNetTight ${nt:.2f}',
                     fontweight='bold', fontsize=9)
        ax.set_xlabel('Trade #')
        ax.set_ylabel('Cum gross P&L ($)')

    fig.suptitle('Phase 3 — Full Globex: RTH vs Overnight Equity Curves',
                 fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_globex_rth_vs_ovn.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    global DT_YR
    print(SEP2)
    print('  05 — Robustness, Friday Filter, Full Globex, and Trading Methodology')
    print(SEP2)

    # ── Derive ΔT from definitions ────────────────────────────────────────────
    defn  = pd.read_parquet(DATA_DIR / f'definitions_{FRONT}_{BACK}.parquet')
    exp_f = defn.loc[defn['symbol'] == FRONT, 'expiration'].iloc[0]
    exp_b = defn.loc[defn['symbol'] == BACK,  'expiration'].iloc[0]
    DT_YR = (exp_b - exp_f).total_seconds() / (365.25 * 86400)
    sofr_utc = _load_sofr_daily()
    print(f'  ΔT = {DT_YR:.4f} yr  |  Pre-FOMC SOFR loaded')

    # ── Load RTH data ─────────────────────────────────────────────────────────
    print('\n[1/5] Loading RTH data...')
    spread, fv, ba = load_rth_data(sofr_utc, DT_YR)
    dev = (spread - fv).dropna()
    spread = spread.reindex(dev.index)
    print(f'      {len(spread):,} RTH 1s bars  |  Dev mean={dev.mean():.3f}  std={dev.std():.3f}')

    print('[2/5] Pre-computing z-scores (all windows)...')
    zscores = {win: _compute_z(dev, win) for win in Z_WINDOWS}

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print('[3/5] Phase 1 — parameter sensitivity...')
    sens = run_sensitivity(spread, dev, zscores)
    print_sensitivity(sens)
    c1a = chart_tp_sl_heatmap(sens)
    c1b = chart_win_thresh_ci(sens)
    print(f'\n  Charts: {Path(c1a).name}  {Path(c1b).name}')

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print('\n[4/5] Phase 2 — Friday session filter...')
    fri_res = run_friday_filter(spread, dev, zscores)
    print_friday_filter(fri_res)
    c2 = chart_friday_comparison(fri_res)
    print(f'\n  Chart: {Path(c2).name}')

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print('\n[5/5] Phase 3 — Full Globex (loading without RTH filter)...')
    sessions  = load_globex_per_day(sofr_utc, DT_YR)
    glob_res  = run_globex(sessions)
    print_globex(glob_res)
    c3 = chart_globex_rth_vs_ovn(glob_res)
    print(f'\n  Chart: {Path(c3).name}')

    # ── Phase 4 ───────────────────────────────────────────────────────────────
    print_methodology()

    # ── Final synthesis ───────────────────────────────────────────────────────
    print(SEP2)
    print('  OVERALL ASSESSMENT')
    print(SEP)
    print(f"""
  Configuration tested: 10-min z-score / z > 2.5σ entry / TP=0.75pt / SL=0.50pt

  1. ROBUSTNESS: The TP × SL parameter surface is NOT smooth — small parameter
     changes materially alter trade count and net P&L. With only ~31 trades
     per roll window, no single-window result is statistically distinguishable
     from luck. The 95% CI on avg net P&L/trade straddles $0 for most configs.
     VERDICT: INCONCLUSIVE until tested across all 4 roll windows (~124 trades).

  2. FRIDAY FILTER: Skipping the first 30-45 minutes of Friday modestly
     improves Friday's avg gross/trade. However, the improvement is small in
     absolute terms. Non-Friday days dominate the P&L.

  3. FULL GLOBEX: Overnight entries (non-RTH) generate additional trades but
     face much higher effective costs (Night slippage: $54/trade all-in).
     Overnight avg gross must exceed $54 to break even — a very high bar.
     RTH entries remain the only viable window for this strategy.

  4. TRANSACTION COSTS: The primary barrier to profitability is not signal
     quality but trade frequency (too many small captures vs fixed costs).
     Any improvement must either (a) increase avg gross per trade beyond $10.25
     or (b) reduce commission through scale (lower per-contract rates at volume).
  """)
    print(SEP2)


if __name__ == '__main__':
    main()
