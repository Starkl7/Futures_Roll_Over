#!/usr/bin/env python3
"""
04_threshold_analysis.py — Threshold / Exit-Mode Grid Search + Friday Diagnostic

Phase 1 : Threshold × Window grid search (entry z ∈ {2.0,2.5,3.0,3.5,4.0},
          windows ∈ {30s,1min,2min,5min,10min}) — z-exit baseline.

Phase 2 : Exit-mode comparison for the best combinations:
          z-exit(0.5σ), z-exit(0.0σ), fixed TP+SL, TP+Trailing-stop,
          Partial-TP (two tranches) at multiple parameter sets.

Phase 3 : Friday deep-dive — why does Sep 13 structurally fail?
          OU half-life by day, deviation trend, long/short asymmetry,
          bid-ask width, synthetic roll-demand proxy.

Usage:
    cd /Users/stark/Desktop/Projects/Futures_RollOver
    .venv/bin/python notebooks/04_threshold_analysis.py
"""

import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
OUT_DIR  = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Contract / roll window ────────────────────────────────────────────────────
FRONT, BACK = 'ESU4', 'ESZ4'
ROLL_START  = '2024-09-12'
FOMC_UTC    = pd.Timestamp('2024-09-18 18:00:00', tz='UTC')
RTH_OPEN, RTH_CLOSE = '12:30', '19:15'

TICK_SIZE  = 0.25
MULTIPLIER = 50.0
TICK_VALUE = TICK_SIZE * MULTIPLIER   # $12.50

DIV_YIELD    = 0.0130
FOMC_CUT_BPS = 0.0050

# ── Phase 1 grid ──────────────────────────────────────────────────────────────
Z_WINDOWS   = ['30s', '1min', '2min', '5min', '10min']
Z_THRESHOLDS = [2.0, 2.5, 3.0, 3.5, 4.0]
MIN_HOLD    = 5       # seconds (bars); prevents single-bar churn

# ── Phase 2 exit configs ──────────────────────────────────────────────────────
# mode = 'z_revert'  → exit when |z| < exit_z
# mode = 'pts'       → fixed TP + SL in index pts
# mode = 'trail'     → TP + trailing stop (trail_pts behind peak)
# mode = 'partial'   → 50% at tp1, 50% with trailing stop (trail_pts) or at tp2
EXIT_CONFIGS = [
    {'mode': 'z_revert', 'exit_z': 0.5,                          'name': 'Z-exit 0.5σ (baseline)'},
    {'mode': 'z_revert', 'exit_z': 0.0,                          'name': 'Z-exit 0.0σ (full revert)'},
    {'mode': 'pts',      'tp': 0.50, 'sl': 0.50,                 'name': 'TP=0.50pt / SL=0.50pt'},
    {'mode': 'pts',      'tp': 0.75, 'sl': 0.50,                 'name': 'TP=0.75pt / SL=0.50pt'},
    {'mode': 'pts',      'tp': 1.00, 'sl': 0.75,                 'name': 'TP=1.00pt / SL=0.75pt'},
    {'mode': 'trail',    'tp': 0.50, 'sl': 0.50, 'trail': 0.25,  'name': 'TP=0.50pt + Trail=0.25pt'},
    {'mode': 'trail',    'tp': 1.00, 'sl': 0.75, 'trail': 0.25,  'name': 'TP=1.00pt + Trail=0.25pt'},
    {'mode': 'partial',  'tp1': 0.25, 'tp2': 0.75, 'sl': 0.50,   'name': 'Partial: TP1=0.25 / TP2=0.75 / SL=0.50'},
    {'mode': 'partial',  'tp1': 0.50, 'trail': 0.25, 'sl': 0.75, 'name': 'Partial: TP1=0.50 + Trail=0.25 / SL=0.75'},
]

# ── Cost structure ────────────────────────────────────────────────────────────
TC_BASE = 4.00        # $2/contract RT × 2 legs
SLIPPAGE = {
    'Tight': 0.5 * TICK_VALUE,    # $6.25
    'Mid':   1.0 * TICK_VALUE,    # $12.50
    'Wide':  2.0 * TICK_VALUE,    # $25.00
}
TOTAL_COSTS = {k: v + TC_BASE for k, v in SLIPPAGE.items()}

DAY_META = {
    '2024-09-12': ('Thu1', 'steelblue'),
    '2024-09-13': ('Fri',  'crimson'),
    '2024-09-15': ('Sun',  'gray'),
    '2024-09-16': ('Mon',  'darkorange'),
    '2024-09-17': ('Tue',  'purple'),
    '2024-09-18': ('Wed*', 'forestgreen'),
    '2024-09-19': ('Thu2', 'teal'),
}

SEP  = '─' * 120
SEP2 = '═' * 120

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_front_mid_rth() -> pd.Series:
    """Return RTH 1s front-month midprice series (used for FV)."""
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    cols  = ['bid_px_00', 'ask_px_00', 'symbol']
    parts = []
    for f in files:
        df  = pd.read_parquet(f, columns=cols)
        mid = (df[df['symbol'] == FRONT]
               .assign(m=lambda d: (d['bid_px_00'] + d['ask_px_00']) / 2)
               ['m'].resample('1s').last().ffill()
               .pipe(lambda s: s.between_time(RTH_OPEN, RTH_CLOSE)))
        parts.append(mid)
    return pd.concat(parts).sort_index()


def load_rth_data() -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Returns:
      spread  : ESZ4_mid − ESU4_mid  (1s RTH)
      ba_sum  : sum of bid-ask widths of both legs (1s RTH)
      front_sz: mean front bid+ask size (book depth proxy, 1s RTH)
    """
    COLS = ['bid_px_00', 'ask_px_00', 'bid_sz_00', 'ask_sz_00', 'symbol']
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    print(f'Loading {len(files)} day files', end='', flush=True)
    parts = []
    for f in files:
        df    = pd.read_parquet(f, columns=COLS)
        df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['ba']  = df['ask_px_00'] - df['bid_px_00']
        df['sz']  = df['bid_sz_00'] + df['ask_sz_00']
        wide = (
            df.groupby('symbol')[['mid', 'ba', 'sz']]
            .resample('1s').last().ffill()
            .unstack('symbol')
            .between_time(RTH_OPEN, RTH_CLOSE)
        )
        parts.append(wide)
        print('.', end='', flush=True)
    print(' done')

    full    = pd.concat(parts).sort_index()
    spread  = (full[('mid', BACK)] - full[('mid', FRONT)]).dropna()
    ba_sum  = (full[('ba',  FRONT)] + full[('ba',  BACK)]).reindex(spread.index).fillna(0.5)
    front_sz = full[('sz',  FRONT)].reindex(spread.index).fillna(0)
    back_sz  = full[('sz',  BACK )].reindex(spread.index).fillna(0)
    # Synthetic roll-demand proxy: back book share = back_sz / (front_sz + back_sz)
    roll_demand = (back_sz / (front_sz + back_sz + 1e-9)).reindex(spread.index)
    return spread, ba_sum, roll_demand


def build_fv(spread: pd.Series) -> pd.Series:
    sofr_file = DATA_DIR / 'SOFR.csv'
    sofr_raw  = pd.read_csv(sofr_file, parse_dates=['observation_date'],
                             index_col='observation_date')
    sofr_s    = sofr_raw.iloc[:, 0].dropna() / 100.0

    defn  = pd.read_parquet(DATA_DIR / f'definitions_{FRONT}_{BACK}.parquet')
    exp_f = defn.loc[defn['symbol'] == FRONT, 'expiration'].iloc[0]
    exp_b = defn.loc[defn['symbol'] == BACK,  'expiration'].iloc[0]
    dt_yr = (exp_b - exp_f).total_seconds() / (365.25 * 86400)

    sofr_idx  = pd.DatetimeIndex(sofr_s.index).tz_localize('UTC')
    sofr_utc  = pd.Series(sofr_s.values, index=sofr_idx)
    daily_idx = pd.date_range(spread.index[0].normalize(),
                              spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()

    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    )
    pre_sofr = float(r_f[r_f.index < FOMC_UTC].iloc[-1])
    r_f[r_f.index >= FOMC_UTC] = pre_sofr - FOMC_CUT_BPS

    front_mid = _load_front_mid_rth().reindex(spread.index).ffill()
    fv        = front_mid * (r_f - DIV_YIELD) * dt_yr
    return fv


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION ENGINE  (unified: z-exit | pts TP/SL | trailing | partial TP)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_z(dev: pd.Series, window: str) -> np.ndarray:
    mu  = dev.rolling(window, min_periods=1).mean()
    sig = dev.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


def simulate(spread: pd.Series, dev: pd.Series, z: np.ndarray,
             entry_thresh: float, exit_cfg: dict, min_hold: int = MIN_HOLD) -> pd.DataFrame:
    """
    General simulation engine.

    Exit logic (exit_cfg['mode']):
      z_revert : exit when |z| < exit_cfg['exit_z']
      pts      : TP = entry ± tp pts, hard SL = entry ∓ sl pts
      trail    : TP + trailing stop; trail_pts behind running best
      partial  : 50% at tp1, 50% at tp2 (or trail); hard SL throughout
    """
    mode   = exit_cfg['mode']
    prices = spread.values
    zvals  = z
    times  = spread.index
    n      = len(prices)

    dates          = np.array([t.date() for t in times])
    is_last        = np.zeros(n, dtype=bool)
    is_last[-1]    = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last[i] = True

    trades         = []
    pos            = 0          # +1 long, -1 short, 0 flat
    entry_i        = -1
    entry_px       = np.nan
    bars_held      = 0
    pending_dir    = 0
    # partial TP state
    lot1_closed    = False
    best_px        = np.nan     # running best (for trailing stop)

    for i in range(n):
        zi = zvals[i]
        px = prices[i]

        # ── fill pending entry ───────────────────────────────────────────────
        if pending_dir != 0 and pos == 0:
            pos        = pending_dir
            entry_i    = i
            entry_px   = px
            bars_held  = 0
            lot1_closed = False
            best_px    = px
            pending_dir = 0

        # ── manage open position ─────────────────────────────────────────────
        if pos != 0:
            bars_held += 1
            # Update running best (in direction of pos)
            if pos == 1:
                best_px = max(best_px, px)
            else:
                best_px = min(best_px, px)

            exit_now   = False
            exit_reason = ''

            if is_last[i]:
                exit_now    = True
                exit_reason = 'EOD'

            elif mode == 'z_revert':
                if bars_held >= min_hold and not np.isnan(zi) and abs(zi) < exit_cfg['exit_z']:
                    exit_now    = True
                    exit_reason = 'z_revert'

            elif mode == 'pts':
                move = pos * (px - entry_px)          # +ve = favorable
                if move >= exit_cfg['tp']:
                    exit_now    = True
                    exit_reason = 'TP'
                elif move <= -exit_cfg['sl']:
                    exit_now    = True
                    exit_reason = 'SL'

            elif mode == 'trail':
                move  = pos * (px - entry_px)
                trail_stop_hit = False
                if pos == 1:
                    trail_stop_hit = (px < best_px - exit_cfg['trail'])
                else:
                    trail_stop_hit = (px > best_px + exit_cfg['trail'])
                if move >= exit_cfg['tp']:
                    exit_now    = True
                    exit_reason = 'TP'
                elif move <= -exit_cfg['sl']:
                    exit_now    = True
                    exit_reason = 'SL'
                elif trail_stop_hit and bars_held >= min_hold:
                    exit_now    = True
                    exit_reason = 'TRAIL'

            elif mode == 'partial':
                move = pos * (px - entry_px)
                if not lot1_closed and move >= exit_cfg['tp1']:
                    lot1_closed = True   # record but keep position for lot2
                if lot1_closed:
                    # lot 2: trail or fixed tp2
                    if 'trail' in exit_cfg:
                        trail_hit = ((pos == 1  and px < best_px - exit_cfg['trail']) or
                                     (pos == -1 and px > best_px + exit_cfg['trail']))
                        if trail_hit and bars_held >= min_hold:
                            exit_now    = True
                            exit_reason = 'TRAIL2'
                    elif 'tp2' in exit_cfg and move >= exit_cfg['tp2']:
                        exit_now    = True
                        exit_reason = 'TP2'
                if move <= -exit_cfg['sl']:
                    exit_now    = True
                    exit_reason = 'SL'
                if is_last[i]:
                    exit_now    = True
                    exit_reason = 'EOD'

            if exit_now:
                # For partial mode: P&L = avg of two tranche exits
                if mode == 'partial' and lot1_closed:
                    tp1_move    = exit_cfg['tp1']   # lot1 captured exactly tp1
                    lot2_move   = pos * (px - entry_px)
                    gross_pts   = 0.5 * tp1_move + 0.5 * lot2_move
                else:
                    gross_pts = pos * (px - entry_px)

                gross_usd = gross_pts * MULTIPLIER
                trades.append({
                    'entry_time'  : times[entry_i],
                    'exit_time'   : times[i],
                    'direction'   : pos,
                    'entry_spread': entry_px,
                    'exit_spread' : px,
                    'gross_pts'   : gross_pts,
                    'gross_usd'   : gross_usd,
                    'bars_held'   : bars_held,
                    'exit_reason' : exit_reason,
                    'eod_close'   : exit_reason == 'EOD',
                    'post_fomc'   : times[entry_i] >= FOMC_UTC,
                    'max_fav_px'  : best_px,
                })
                pos         = 0
                entry_i     = -1
                bars_held   = 0
                best_px     = np.nan
                lot1_closed = False
                pending_dir = 0

        # ── check for new signal (edge-triggered) ────────────────────────────
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
    df['mfe_pts']   = df.apply(  # max favorable excursion (pts from entry)
        lambda r: abs(r['max_fav_px'] - r['entry_spread']), axis=1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# AGGREGATION HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _pf(t: pd.DataFrame) -> float:
    w = t.loc[t['gross_usd'] > 0, 'gross_usd'].sum()
    l = t.loc[t['gross_usd'] < 0, 'gross_usd'].sum()
    return round(w / abs(l), 2) if l != 0 else float('inf')


def agg(t: pd.DataFrame) -> dict:
    if t is None or t.empty:
        return {}
    n = len(t)
    return {
        'n'            : n,
        'wr'           : (t['gross_usd'] > 0).mean() * 100,
        'avg_hold'     : t['bars_held'].mean(),
        'avg_gross'    : t['gross_usd'].mean(),
        'tot_gross'    : t['gross_usd'].sum(),
        'pf'           : _pf(t),
        'mdd'          : t['drawdown'].min(),
        'eod_pct'      : t['eod_close'].mean() * 100,
        'avg_mfe'      : t['mfe_pts'].mean(),
        **{f'avg_net_{k}': t[f'net_{k}'].mean() for k in SLIPPAGE},
        **{f'tot_net_{k}': t[f'net_{k}'].sum()  for k in SLIPPAGE},
        **{f'wr_net_{k}': (t[f'net_{k}'] > 0).mean() * 100 for k in SLIPPAGE},
    }


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — THRESHOLD × WINDOW GRID
# ─────────────────────────────────────────────────────────────────────────────
def run_phase1(spread, dev, zscores):
    """
    zscores : {window_str: np.ndarray}
    Returns list of dicts for the results table.
    """
    baseline_cfg = {'mode': 'z_revert', 'exit_z': 0.5}
    rows = []
    for win in Z_WINDOWS:
        z = zscores[win]
        for thresh in Z_THRESHOLDS:
            trades = simulate(spread, dev, z, thresh, baseline_cfg)
            r = agg(trades)
            r['window'] = win
            r['thresh'] = thresh
            rows.append(r)
    return rows


def print_phase1(rows):
    print()
    print('  PHASE 1 — THRESHOLD × WINDOW GRID  (exit: z < 0.5σ, baseline)')
    print('  Columns: Trades | WinRate | AvgHold | Gross/Tr | NetTight/Tr | NetMid/Tr | PFactor | MDD')
    print(SEP)

    # Print as pivot-style grouped by window
    for win in Z_WINDOWS:
        win_rows = [r for r in rows if r.get('window') == win and r.get('n', 0) > 0]
        if not win_rows:
            continue
        print(f'\n  ── {win} z-score window ──')
        print(f"  {'Thresh':>7}  {'Trades':>6}  {'WinRate':>7}  {'Hold':>5}  "
              f"{'Gross/Tr':>9}  {'TotGross':>9}  {'Net(Tight)/Tr':>14}  "
              f"{'Net(Mid)/Tr':>12}  {'PF':>6}  {'MDD':>9}")
        for r in win_rows:
            breakeven_ok = '✓' if r.get('avg_net_Tight', -999) >= 0 else ' '
            print(
                f"  {r['thresh']:>7.1f}  {r['n']:>6}  {r['wr']:>6.1f}%  "
                f"{r['avg_hold']:>4.0f}s  ${r['avg_gross']:>8.2f}  "
                f"${r['tot_gross']:>8,.0f}  ${r.get('avg_net_Tight',0):>12.2f}  "
                f"${r.get('avg_net_Mid',0):>10.2f}  {r['pf']:>6.2f}  "
                f"${r['mdd']:>8,.0f} {breakeven_ok}"
            )
    print()
    print('  ✓ = avg_net_Tight ≥ 0  (covers tight-slippage + commission)')

    # Identify best combos by avg_net_Tight
    best = sorted([r for r in rows if r.get('n', 0) > 0],
                  key=lambda r: r.get('avg_net_Tight', -1e9), reverse=True)[:5]
    print()
    print('  TOP 5 COMBOS BY AVERAGE NET P&L/TRADE (tight slippage):')
    print(f"  {'Window':<6}  {'Thresh':>6}  {'Trades':>6}  {'Gross/Tr':>9}  "
          f"{'Net(Tight)/Tr':>14}  {'Net(Mid)/Tr':>12}")
    for r in best:
        print(f"  {r['window']:<6}  {r['thresh']:>6.1f}  {r['n']:>6}  "
              f"${r['avg_gross']:>8.2f}  ${r.get('avg_net_Tight',0):>12.2f}  "
              f"${r.get('avg_net_Mid',0):>10.2f}")
    return best


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — EXIT MODE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def run_phase2(spread, dev, zscores, best_combos):
    """
    For each of the top combos from Phase 1, test all EXIT_CONFIGS.
    """
    results = {}
    for combo in best_combos[:3]:
        win    = combo['window']
        thresh = combo['thresh']
        z      = zscores[win]
        key    = f"{win}/z{thresh}"
        results[key] = []
        for ecfg in EXIT_CONFIGS:
            trades = simulate(spread, dev, z, thresh, ecfg)
            r = agg(trades)
            r['exit_name'] = ecfg['name']
            r['window']    = win
            r['thresh']    = thresh
            results[key].append(r)
    return results


def print_phase2(results):
    print()
    print('  PHASE 2 — EXIT MODE COMPARISON  (top 3 threshold×window combos)')
    print(SEP)
    for key, rows in results.items():
        print(f'\n  ── Combo: {key} ──')
        print(f"  {'Exit Mode':<45}  {'Trades':>6}  {'WinRate':>7}  "
              f"{'Gross/Tr':>9}  {'Net(Tight)/Tr':>14}  {'Net(Mid)/Tr':>12}  "
              f"{'EOD%':>6}  {'AvgHold':>7}")
        for r in rows:
            if not r.get('n'):
                continue
            marker = ' ✓' if r.get('avg_net_Tight', -999) >= 0 else '  '
            print(
                f"  {r['exit_name']:<45}{marker}"
                f"  {r['n']:>5}  {r['wr']:>6.1f}%  ${r['avg_gross']:>8.2f}  "
                f"${r.get('avg_net_Tight',0):>12.2f}  ${r.get('avg_net_Mid',0):>10.2f}  "
                f"{r.get('eod_pct',0):>5.0f}%  {r.get('avg_hold',0):>6.0f}s"
            )


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — FRIDAY DEEP DIVE
# ─────────────────────────────────────────────────────────────────────────────
def ou_half_life(series: pd.Series) -> float:
    """
    Estimate Ornstein-Uhlenbeck half-life in bars via lag-1 OLS:
      Δx_t = α + β·x_{t-1} + ε
    κ = -β (mean-reversion speed per bar)
    half-life = ln(2)/κ   (in bars)
    Returns inf if κ ≤ 0 (no mean-reversion; explosive or RW).
    """
    x   = series.dropna()
    dx  = x.diff().dropna()
    lag = x.shift(1).reindex(dx.index).dropna()
    dx  = dx.reindex(lag.index)
    if len(lag) < 10:
        return np.inf
    beta = np.cov(lag, dx)[0, 1] / np.var(lag) if np.var(lag) > 0 else 0
    kappa = -beta
    return np.log(2) / kappa if kappa > 0 else np.inf


def acf(series: pd.Series, max_lag: int = 30) -> np.ndarray:
    """Sample autocorrelation function up to max_lag."""
    x    = series.dropna().values
    x    = x - x.mean()
    var  = np.var(x)
    if var == 0:
        return np.zeros(max_lag + 1)
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(np.mean(x[:-k] * x[k:]) / var)
    return np.array(out)


def run_phase3(spread: pd.Series, dev: pd.Series,
               ba_sum: pd.Series, roll_demand: pd.Series,
               zscores: dict, best_combos: list) -> dict:
    """
    Compute all Friday diagnostic metrics and return structured results.
    """
    # Use 5s resampled deviation for autocorrelation (1s is noisy but also fine)
    dev_5s   = dev.resample('5s').last().ffill()
    ba_5s    = ba_sum.resample('5s').last().ffill()
    rd_5s    = roll_demand.resample('5s').last().ffill()
    spread_5s = spread.resample('5s').last().ffill()

    days     = sorted({str(t.date()) for t in dev.index})
    diagnostics = {}

    for d in days:
        mask   = dev_5s.index.date == pd.Timestamp(d).date()
        d_dev  = dev_5s[mask]
        d_ba   = ba_5s[mask]
        d_rd   = rd_5s[mask]
        d_spr  = spread_5s[mask]

        if len(d_dev) < 10:
            continue

        # OU half-life (mean-reversion speed)
        hl = ou_half_life(d_dev)

        # 30-lag ACF of deviation
        ac = acf(d_dev, max_lag=30)

        # Slope of deviation vs time (linear trend = pts/hour)
        t_hr  = np.arange(len(d_dev)) * 5 / 3600      # hours
        slope, intercept, r_val, p_val, _ = spstats.linregress(t_hr, d_dev.values)

        # Mean / std BA
        mean_ba = d_ba.mean()
        std_ba  = d_ba.std()

        # Roll-demand proxy: mean back-book share in each 30-min window
        rd_hourly = d_rd.resample('30min').mean()

        # Hourly deviation mean and std
        dev_hourly = d_dev.resample('30min').agg(['mean', 'std'])

        diagnostics[d] = {
            'ou_hl_bars'    : hl,
            'ou_hl_min'     : hl * 5 / 60,   # convert bars (5s) → minutes
            'acf'           : ac,
            'dev_slope_pph' : slope,          # pts per hour
            'dev_r2'        : r_val ** 2,
            'dev_p'         : p_val,
            'mean_ba'       : mean_ba,
            'dev_hourly'    : dev_hourly,
            'rd_hourly'     : rd_hourly,
            'n_bars'        : len(d_dev),
        }

    # Long/short P&L asymmetry by day (using best combo baseline)
    best_win    = best_combos[0]['window']
    best_thresh = best_combos[0]['thresh']
    z_arr       = zscores[best_win]
    base_cfg    = {'mode': 'z_revert', 'exit_z': 0.5}
    all_trades  = simulate(spread, dev, z_arr, best_thresh, base_cfg)

    asym_by_day = {}
    if not all_trades.empty:
        for d in days:
            mask = all_trades['entry_time'].dt.date == pd.Timestamp(d).date()
            dt   = all_trades[mask]
            if dt.empty:
                continue
            longs  = dt[dt['direction'] ==  1]
            shorts = dt[dt['direction'] == -1]
            asym_by_day[d] = {
                'n_long'    : len(longs),
                'n_short'   : len(shorts),
                'wr_long'   : (longs['gross_usd']  > 0).mean() * 100 if not longs.empty  else np.nan,
                'wr_short'  : (shorts['gross_usd'] > 0).mean() * 100 if not shorts.empty else np.nan,
                'ag_long'   : longs['gross_usd'].mean()  if not longs.empty  else np.nan,
                'ag_short'  : shorts['gross_usd'].mean() if not shorts.empty else np.nan,
                'tot_long'  : longs['gross_usd'].sum()   if not longs.empty  else 0.0,
                'tot_short' : shorts['gross_usd'].sum()  if not shorts.empty else 0.0,
            }

    return {'by_day': diagnostics, 'asym': asym_by_day,
            'best_trades': all_trades,
            'best_win': best_win, 'best_thresh': best_thresh}


def print_phase3(p3: dict):
    diag  = p3['by_day']
    asym  = p3['asym']

    print()
    print('  PHASE 3 — FRIDAY DEEP DIVE: STRUCTURAL DIAGNOSTICS BY DAY')
    print(SEP)
    print()
    print('  A. Mean-Reversion Speed (Ornstein-Uhlenbeck Half-Life)')
    print(f"  {'Date':<12}  {'Day':<5}  {'HL(min)':>8}  {'Status':<30}  "
          f"{'Dev Slope(pts/hr)':>18}  {'R²':>6}  {'p-val':>8}")
    print('  ' + '─' * 85)
    for d, r in diag.items():
        label, _  = DAY_META.get(d, ('?', ''))
        hl        = r['ou_hl_min']
        slope     = r['dev_slope_pph']
        r2        = r['dev_r2']
        pval      = r['dev_p']
        if hl < 1:
            status = 'FAST reversion (<1 min)'
        elif hl < 5:
            status = 'Moderate reversion (1–5 min)'
        elif hl < 30:
            status = 'SLOW reversion (5–30 min)'
        elif np.isinf(hl):
            status = '⚠ NO reversion (trending/RW)'
        else:
            status = f'Very slow (>{hl:.0f} min)'
        trend_flag = ' ← TRENDING' if pval < 0.05 and abs(slope) > 0.2 else ''
        hl_str     = f'{hl:>7.1f}' if not np.isinf(hl) else '    ∞'
        print(f'  {d}  {label:<5}  {hl_str}  {status:<30}  '
              f'{slope:>+17.3f}  {r2:>6.4f}  {pval:>8.4f}{trend_flag}')

    print()
    print('  B. Bid-Ask Spread Width by Day  (proxy for execution cost)')
    print(f"  {'Date':<12}  {'Day':<5}  {'Mean B/A (pts)':>14}  "
          f"{'Vs Thu1 (basis)':>16}  {'Interpretation'}")
    print('  ' + '─' * 75)
    ref_ba = diag.get('2024-09-12', {}).get('mean_ba', np.nan)
    for d, r in diag.items():
        label, _ = DAY_META.get(d, ('?', ''))
        ba       = r['mean_ba']
        rel      = (ba / ref_ba - 1) * 100 if not np.isnan(ref_ba) and ref_ba > 0 else 0
        interp   = ('Tightest (roll not started)'   if label == 'Thu1' else
                    'Widest  ← roll surge begins'    if label == 'Fri'  else
                    'Very wide (crossover volume)'   if label == 'Mon'  else
                    'Narrowing (OI settled in ESZ4)' if label in ('Tue', 'Thu2') else
                    'FOMC volatility'                if label == 'Wed*' else
                    'Overnight (thin)')
        print(f'  {d}  {label:<5}  {ba:>14.4f}  {rel:>+14.1f}%  {interp}')

    print()
    print('  C. Long vs Short P&L Asymmetry by Day')
    print(f"  {'Date':<12}  {'Day':<5}  {'#Long':>6}  {'WR_L':>6}  "
          f"{'Avg_L':>8}  {'TotL':>9}  ||  {'#Short':>7}  {'WR_S':>6}  "
          f"{'Avg_S':>8}  {'TotS':>9}  {'Asymmetry'}")
    print('  ' + '─' * 110)
    for d, r in asym.items():
        label, _ = DAY_META.get(d, ('?', ''))
        wr_l = f"{r['wr_long']:.0f}%" if not np.isnan(r['wr_long']) else '  —'
        wr_s = f"{r['wr_short']:.0f}%" if not np.isnan(r['wr_short']) else '  —'
        ag_l = f"${r['ag_long']:.2f}"   if not np.isnan(r['ag_long'])  else '  —'
        ag_s = f"${r['ag_short']:.2f}"  if not np.isnan(r['ag_short']) else '  —'
        # Asymmetry flag
        if not np.isnan(r['ag_long']) and not np.isnan(r['ag_short']):
            if r['ag_short'] < -2 and r['ag_long'] > 0:
                asym_flag = '⚠ SHORT trades bleed (roll buy pressure ↑)'
            elif r['ag_long'] < -2 and r['ag_short'] > 0:
                asym_flag = '⚠ LONG trades bleed (roll sell pressure ↑)'
            elif abs(r['ag_long'] - r['ag_short']) < 1:
                asym_flag = 'Symmetric (noise-driven)'
            else:
                asym_flag = ''
        else:
            asym_flag = ''
        print(f'  {d}  {label:<5}  {r["n_long"]:>6}  {wr_l:>5}  {ag_l:>7}  '
              f'${r["tot_long"]:>8,.0f}  ||  {r["n_short"]:>6}  {wr_s:>5}  '
              f'{ag_s:>7}  ${r["tot_short"]:>8,.0f}  {asym_flag}')

    print()
    print('  D. Roll-Demand Proxy (mean back-book share × 30-min bucket) — Friday vs Monday')
    for d in ['2024-09-13', '2024-09-16']:
        if d not in diag:
            continue
        label, _ = DAY_META.get(d, ('?', ''))
        rd = diag[d].get('rd_hourly')
        if rd is None:
            continue
        print(f'\n  {d} ({label}):')
        for ts, val in rd.items():
            et = ts.tz_localize('UTC') if ts.tzinfo is None else ts
            et_str = (et - pd.Timedelta('4h')).strftime('%H:%M ET')
            bar = '█' * int(val * 50)
            print(f'    {et_str}  {val:.3f}  {bar}')


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
WIN_COLORS = {'30s':'#e74c3c','1min':'#e67e22','2min':'#f1c40f',
              '5min':'#27ae60','10min':'#2980b9'}
DAY_COLORS = {d: v[1] for d, v in DAY_META.items()}
DAY_LABELS = {d: v[0] for d, v in DAY_META.items()}


def chart_phase1_heatmap(rows: list[dict]) -> str:
    """Net P&L / trade (tight) as threshold × window heatmap."""
    vals = np.full((len(Z_WINDOWS), len(Z_THRESHOLDS)), np.nan)
    for r in rows:
        if not r.get('n'):
            continue
        ri = Z_WINDOWS.index(r['window'])
        ci = Z_THRESHOLDS.index(r['thresh'])
        vals[ri, ci] = r.get('avg_net_Tight', np.nan)

    vmax = max(abs(np.nanmax(vals)), abs(np.nanmin(vals)), 1)
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(vals, cmap='RdYlGn', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(Z_THRESHOLDS)))
    ax.set_xticklabels([f'z>{t}σ' for t in Z_THRESHOLDS])
    ax.set_yticks(range(len(Z_WINDOWS)))
    ax.set_yticklabels(Z_WINDOWS)
    ax.set_title('Avg Net P&L / Trade — Tight Slippage ($6.25 + $4 TC)\n'
                 'Green = profitable per trade | Red = loss per trade',
                 fontweight='bold')
    plt.colorbar(im, ax=ax, label='Avg net P&L ($)')
    for ri in range(len(Z_WINDOWS)):
        for ci in range(len(Z_THRESHOLDS)):
            v = vals[ri, ci]
            if not np.isnan(v):
                ax.text(ci, ri, f'${v:.1f}', ha='center', va='center',
                        fontsize=8, color='black' if abs(v) < vmax * 0.6 else 'white')
    fig.tight_layout()
    out = str(OUT_DIR / 'p1_threshold_heatmap.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase1_tradecount(rows: list[dict]) -> str:
    """Trade count and total gross P&L across thresholds per window."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    for win_i, win in enumerate(Z_WINDOWS):
        wr = [r for r in rows if r.get('window') == win and r.get('n', 0) > 0]
        xs = [r['thresh'] for r in wr]
        ys_n = [r['n'] for r in wr]
        ys_g = [r['tot_gross'] for r in wr]
        c = WIN_COLORS[win]
        axes[0].plot(xs, ys_n,  color=c, marker='o', lw=1.8, label=win)
        axes[1].plot(xs, ys_g,  color=c, marker='o', lw=1.8, label=win)

    for ax, title, ylabel in zip(
        axes,
        ['Trade Count vs Entry Threshold', 'Total Gross P&L ($) vs Entry Threshold'],
        ['# Trades (7 days)', 'Total Gross P&L ($)']
    ):
        ax.set_xlabel('Entry z-score threshold (σ)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_xticks(Z_THRESHOLDS)
        ax.axhline(0, color='black', lw=0.5)

    fig.tight_layout()
    out = str(OUT_DIR / 'p1_threshold_tradecount.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase2_exits(results: dict) -> str:
    """Exit mode comparison: avg net P&L per trade (tight + mid) per combo."""
    combos = list(results.keys())
    n_combos = len(combos)
    fig, axes = plt.subplots(n_combos, 1, figsize=(14, 4 * n_combos), squeeze=False)

    for ax, key in zip(axes[:, 0], combos):
        rows  = results[key]
        names = [r['exit_name'] for r in rows if r.get('n', 0) > 0]
        nt    = [r.get('avg_net_Tight', 0) for r in rows if r.get('n', 0) > 0]
        nm    = [r.get('avg_net_Mid',   0) for r in rows if r.get('n', 0) > 0]
        x     = np.arange(len(names))
        ax.bar(x - 0.2, nt, 0.35, label='Net (Tight)',  color='steelblue', alpha=0.8)
        ax.bar(x + 0.2, nm, 0.35, label='Net (Mid)',    color='tomato',    alpha=0.8)
        ax.axhline(0, color='black', lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('Avg net P&L/trade ($)')
        ax.set_title(f'Exit Mode Comparison — {key}', fontweight='bold')
        ax.legend(fontsize=9)
        # Add breakeven line
        ax.axhline(-TOTAL_COSTS['Tight'] + TOTAL_COSTS['Tight'], color='gray',
                   lw=0.5, linestyle=':')

    fig.tight_layout()
    out = str(OUT_DIR / 'p2_exit_modes.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_ou(diag: dict) -> str:
    """OU half-life + deviation slope by day."""
    days   = list(diag.keys())
    labels = [DAY_LABELS.get(d, d[-5:]) for d in days]
    colors = [DAY_COLORS.get(d, 'gray') for d in days]
    hls    = [min(diag[d]['ou_hl_min'], 120) for d in days]   # cap at 2h for chart
    slopes = [diag[d]['dev_slope_pph'] for d in days]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    bars = ax.bar(range(len(days)), hls, color=colors, alpha=0.8, edgecolor='white')
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Half-life (minutes)  [capped at 120]')
    ax.set_title('OU Half-Life of FV Deviation by Day\n'
                 'Shorter = faster mean-reversion = better for strategy',
                 fontweight='bold')
    ax.axhline(5,  color='green', lw=1, linestyle='--', label='5-min (fast)')
    ax.axhline(30, color='orange', lw=1, linestyle='--', label='30-min (slow)')
    ax.legend(fontsize=8)
    for bar, v in zip(bars, hls):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1,
                f'{v:.0f}m', ha='center', fontsize=8)

    ax = axes[1]
    bar_colors = ['red' if s > 0.2 else ('blue' if s < -0.2 else 'gray') for s in slopes]
    bars = ax.bar(range(len(days)), slopes, color=bar_colors, alpha=0.8, edgecolor='white')
    ax.axhline(0, color='black', lw=0.8)
    ax.axhline( 0.2, color='red',  lw=1, linestyle=':', alpha=0.6)
    ax.axhline(-0.2, color='blue', lw=1, linestyle=':', alpha=0.6)
    ax.set_xticks(range(len(days)))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Deviation trend (pts/hour)')
    ax.set_title('Intraday FV Deviation Slope by Day\n'
                 'Positive = spread gets richer throughout session',
                 fontweight='bold')
    for bar, v in zip(bars, slopes):
        ax.text(bar.get_x() + bar.get_width()/2, v + (0.02 if v >= 0 else -0.06),
                f'{v:+.2f}', ha='center', fontsize=8)

    fig.tight_layout()
    out = str(OUT_DIR / 'p3_ou_halflife.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_acf(diag: dict) -> str:
    """ACF of deviation by day — shows persistence vs reversion."""
    days   = [d for d in diag if d != '2024-09-15']   # skip Sun (no RTH)
    labels = [DAY_LABELS.get(d, d[-5:]) for d in days]
    colors = [DAY_COLORS.get(d, 'gray') for d in days]
    lags   = np.arange(31)

    fig, ax = plt.subplots(figsize=(12, 5))
    for d, label, color in zip(days, labels, colors):
        ac = diag[d]['acf']
        lw = 2.5 if label in ('Fri', 'Mon') else 1.2
        ls = '-' if label not in ('Fri',) else '--'
        ax.plot(lags[1:], ac[1:], color=color, lw=lw, ls=ls, label=label, alpha=0.9)

    ax.axhline(0, color='black', lw=0.8)
    ax.axhline(0.1, color='gray', lw=0.5, linestyle=':')
    conf = 1.96 / np.sqrt(max(d['n_bars'] for d in diag.values()))
    ax.fill_between(lags[1:], -conf, conf, alpha=0.1, color='gray',
                    label='95% CI (approx)')
    ax.set_xlabel('Lag (5-second bars)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation of FV Deviation by Day (5s bars, RTH)\n'
                 'Positive at lag>0 = persistence (trending); Negative = mean-reversion',
                 fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_acf_by_day.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_dev_intraday(diag: dict) -> str:
    """Intraday deviation trend (30-min rolling mean) by day."""
    days   = [d for d in diag if d != '2024-09-15']
    n = len(days)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 4), sharey=False)
    for ax, d in zip(axes, days):
        label, color = DAY_META.get(d, ('?', 'gray'))
        dh = diag[d]['dev_hourly']
        x  = np.arange(len(dh))
        ax.fill_between(x, dh['mean'] - dh['std'], dh['mean'] + dh['std'],
                        alpha=0.2, color=color)
        ax.plot(x, dh['mean'], color=color, lw=2, marker='o', markersize=4)
        ax.axhline(dh['mean'].mean(), color='black', lw=0.6, linestyle='--', alpha=0.5)
        slope = diag[d]['dev_slope_pph']
        ax.set_title(f'{label}\nslope {slope:+.2f}pt/hr', fontsize=9, fontweight='bold',
                     color=color)
        ax.set_xlabel('30-min buckets (RTH)')
        ax.set_ylabel('Mean deviation (pts)')
        tick_locs = list(range(0, len(dh), max(1, len(dh)//4)))
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([dh.index[i].strftime('%H:%M') for i in tick_locs],
                            fontsize=7, rotation=30)
    fig.suptitle('Intraday FV Deviation by 30-min Bucket (mean ± 1σ)', fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_dev_intraday.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_asymmetry(asym: dict) -> str:
    """Long vs short: win rate and avg gross by day."""
    days   = [d for d in asym if d != '2024-09-15']
    labels = [DAY_LABELS.get(d, d[-5:]) for d in days]
    x      = np.arange(len(days))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    w = 0.35
    for ax, metric, title, suffix in zip(
        axes,
        ['wr', 'ag'],
        ['Win Rate: Long vs Short  (% of trades gross > 0)',
         'Avg Gross P&L: Long vs Short'],
        ['%', '$']
    ):
        yl = [asym[d][f'{metric}_long']  for d in days]
        ys = [asym[d][f'{metric}_short'] for d in days]
        yl_clean = [v if not np.isnan(v) else 0 for v in yl]
        ys_clean = [v if not np.isnan(v) else 0 for v in ys]
        ax.bar(x - w/2, yl_clean, w, color='steelblue', alpha=0.8, label='Long (buy spread)', edgecolor='white')
        ax.bar(x + w/2, ys_clean, w, color='tomato',    alpha=0.8, label='Short (sell spread)', edgecolor='white')
        ax.axhline(0, color='black', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_title(title, fontweight='bold')
        ax.legend(fontsize=8)
        for i, (vl, vs) in enumerate(zip(yl_clean, ys_clean)):
            ax.text(i - w/2, vl + abs(vl)*0.02 + 0.5, f'{vl:.0f}{suffix}',
                    ha='center', fontsize=7)
            ax.text(i + w/2, vs + abs(vs)*0.02 + 0.5, f'{vs:.0f}{suffix}',
                    ha='center', fontsize=7)
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_long_short_asymmetry.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_roll_demand(diag: dict) -> str:
    """Roll-demand proxy (back book share) over RTH, Thu1 vs Fri vs Mon."""
    focus = {'2024-09-12': ('Thu1', 'steelblue'),
             '2024-09-13': ('Fri',  'crimson'),
             '2024-09-16': ('Mon',  'darkorange')}
    fig, ax = plt.subplots(figsize=(11, 4))
    for d, (label, color) in focus.items():
        if d not in diag:
            continue
        rd = diag[d]['rd_hourly']
        x  = np.arange(len(rd))
        lw = 2.5 if label == 'Fri' else 1.5
        ax.plot(x, rd.values, color=color, lw=lw, marker='o', markersize=5,
                label=label, alpha=0.9)
        ticks = list(range(0, len(rd), max(1, len(rd)//6)))
        ax.set_xticks(ticks)
        ax.set_xticklabels([rd.index[i].strftime('%H:%M ET') for i in ticks],
                            fontsize=8, rotation=20)
    ax.set_ylabel('Back-book share (ESZ4 sz / total sz)')
    ax.set_title('Synthetic Roll Demand Proxy — Back-Book Share (30-min avg)\n'
                 'Rising = institutions increasing ESZ4 book dominance during session',
                 fontweight='bold')
    ax.legend(fontsize=10)
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_roll_demand.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def chart_phase3_equity_by_day(p3: dict) -> str:
    """Equity curve split by day for best combo."""
    trades = p3['best_trades']
    if trades.empty:
        return ''
    days   = sorted({str(t.date()) for t in trades['entry_time']})
    n      = len(days)
    fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 4), sharey=False)
    for ax, d in zip(axes, days):
        label, color = DAY_META.get(d, ('?', 'gray'))
        dt  = trades[trades['entry_time'].dt.date == pd.Timestamp(d).date()].copy()
        if dt.empty:
            ax.set_title(d[-5:])
            continue
        eq  = dt['gross_usd'].cumsum()
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=(eq.values >= 0), alpha=0.3, color='forestgreen')
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=(eq.values <  0), alpha=0.3, color='crimson')
        ax.plot(range(len(eq)), eq.values, color=color, lw=1.5)
        ax.axhline(0, color='black', lw=0.5)
        ax.set_title(f'{label}  |  {len(dt)} trades\nGross ${eq.iloc[-1]:,.0f}',
                     fontsize=8, fontweight='bold', color=color)
    fig.suptitle(f'Daily Equity Curves — {p3["best_win"]} / z>{p3["best_thresh"]}σ',
                 fontweight='bold')
    fig.tight_layout()
    out = str(OUT_DIR / 'p3_daily_equity.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(SEP2)
    print('  ESU4/ESZ4 — Threshold / Exit-Mode Analysis + Friday Diagnostic')
    print(SEP2)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('\n[1/5] Loading data...')
    spread, ba_sum, roll_demand = load_rth_data()
    print(f'      {len(spread):,} RTH 1s bars')

    print('[2/5] Building fair-value...')
    fv  = build_fv(spread)
    dev = (spread - fv).dropna()
    spread = spread.reindex(dev.index)
    print(f'      Dev mean={dev.mean():.3f}  std={dev.std():.3f}  '
          f'min={dev.min():.2f}  max={dev.max():.2f}')

    # Pre-compute all z-score arrays (one per window)
    print('[3/5] Pre-computing z-scores...')
    zscores = {win: _compute_z(dev, win) for win in Z_WINDOWS}
    print(f'      Done for {len(Z_WINDOWS)} windows.')

    # ── Phase 1 ───────────────────────────────────────────────────────────────
    print('[4/5] Phase 1 — threshold × window grid search...')
    p1_rows  = run_phase1(spread, dev, zscores)
    best     = print_phase1(p1_rows)

    # Phase 1 charts
    c1a = chart_phase1_heatmap(p1_rows)
    c1b = chart_phase1_tradecount(p1_rows)
    print(f'\n  Charts: {Path(c1a).name}  {Path(c1b).name}')

    # ── Phase 2 ───────────────────────────────────────────────────────────────
    print('\n[5/5] Phase 2 — exit mode comparison...')
    p2_results = run_phase2(spread, dev, zscores, best)
    print_phase2(p2_results)
    c2 = chart_phase2_exits(p2_results)
    print(f'\n  Chart: {Path(c2).name}')

    # ── Phase 3 ───────────────────────────────────────────────────────────────
    print('\n[6/6] Phase 3 — Friday deep dive...')
    ba_aligned = ba_sum.reindex(dev.index).ffill()
    rd_aligned = roll_demand.reindex(dev.index).ffill()
    p3 = run_phase3(spread, dev, ba_aligned, rd_aligned, zscores, best)
    print_phase3(p3)

    c3a = chart_phase3_ou(p3['by_day'])
    c3b = chart_phase3_acf(p3['by_day'])
    c3c = chart_phase3_dev_intraday(p3['by_day'])
    c3d = chart_phase3_asymmetry(p3['asym'])
    c3e = chart_phase3_roll_demand(p3['by_day'])
    c3f = chart_phase3_equity_by_day(p3)
    print(f'\n  Charts: {" ".join(Path(c).name for c in [c3a,c3b,c3c,c3d,c3e,c3f])}')

    # ── Final synthesis ───────────────────────────────────────────────────────
    print()
    print(SEP2)
    print('  SYNTHESIS — WHY THE STRATEGY STRUGGLES AND WHAT CHANGES HELP')
    print(SEP)

    # Find best combo after phase 2
    all_p2_rows = []
    for rows in p2_results.values():
        all_p2_rows.extend(rows)
    best_p2 = max([r for r in all_p2_rows if r.get('n', 0) > 0],
                  key=lambda r: r.get('avg_net_Tight', -1e9))

    print(f'\n  Best exit config found:')
    print(f'    Window  : {best_p2["window"]}   Threshold : z>{best_p2["thresh"]}σ')
    print(f'    Exit    : {best_p2["exit_name"]}')
    print(f'    Trades  : {best_p2["n"]}   WinRate: {best_p2["wr"]:.1f}%   AvgHold: {best_p2.get("avg_hold",0):.0f}s')
    print(f'    Gross/Tr: ${best_p2["avg_gross"]:.2f}   Net(Tight)/Tr: ${best_p2.get("avg_net_Tight",0):.2f}   Net(Mid)/Tr: ${best_p2.get("avg_net_Mid",0):.2f}')

    print()
    print('  Breakeven thresholds (gross/trade to cover all-in costs):')
    for k, v in TOTAL_COSTS.items():
        print(f'    {k:<8}: ${v:.2f}  =  {v/MULTIPLIER:.4f} pts  =  {v/TICK_VALUE:.2f} ticks')

    print()
    print('  Friday root causes (from Phase 3):')
    fri_diag = p3['by_day'].get('2024-09-13', {})
    fri_asym = p3['asym'].get('2024-09-13', {})
    hl_fri = fri_diag.get('ou_hl_min', np.nan)
    sl_fri = fri_diag.get('dev_slope_pph', 0)
    hl_mon = p3['by_day'].get('2024-09-16', {}).get('ou_hl_min', np.nan)
    print(f'    1. OU half-life : {hl_fri:.1f} min (Friday)  vs  {hl_mon:.1f} min (Monday)')
    print(f'    2. Trend slope  : {sl_fri:+.3f} pts/hr (Friday)')
    ag_s = fri_asym.get('ag_short', np.nan)
    ag_l = fri_asym.get('ag_long',  np.nan)
    print(f'    3. Short avg gross: ${ag_s:.2f}   Long avg gross: ${ag_l:.2f}')
    print(f'    4. Mean B/A width: {fri_diag.get("mean_ba", 0):.4f} pts')
    print(SEP2)


if __name__ == '__main__':
    main()
