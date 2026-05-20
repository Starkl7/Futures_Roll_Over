#!/usr/bin/env python3
"""
07_tearsheet.py

Four-page quantitative tear sheet for the ESU4/ESZ4 ES calendar spread
mean-reversion strategy.

Config : 10-min rolling z-score  |  z > ±2.5σ entry (edge-triggered)
         TP = +0.75 pts ($37.50)  |  SL = −0.50 pts (−$25.00)
         RTH only (12:30–19:15 UTC)  |  Friday open 30-min filter applied

Data   : Databento GLBX.MDP3  |  MBP-10 1-second bars  |  Sep 12–19 2024

Output : reports/ESU4_ESZ4_TearSheet.pdf  (3-page professional tear sheet)

Usage  :
    cd /Users/stark/Desktop/Projects/Futures_RollOver
    .venv/bin/python notebooks/07_tearsheet.py
"""

import glob
import json
import warnings
from pathlib import Path
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
REPORTS  = Path(__file__).parent.parent / 'reports'
REPORTS.mkdir(exist_ok=True)
OUT_PDF  = REPORTS / 'ESU4_ESZ4_TearSheet.pdf'

# ── Roll window ───────────────────────────────────────────────────────────────
FRONT      = 'ESU4'
BACK       = 'ESZ4'
ROLL_START = '2024-09-12'
FOMC_UTC   = pd.Timestamp('2024-09-18 18:00:00', tz='UTC')
RTH_OPEN   = '12:30'    # UTC (08:30 ET)
RTH_CLOSE  = '19:15'    # UTC (15:15 ET)
FRI_SKIP_MIN    = 30    # skip first 30 min of any Friday RTH

# ── FOMC blackout window ──────────────────────────────────────────────────────
# Block new entry signals and force-close open positions around the FOMC
# announcement. Pre-blackout prevents stale-SOFR phantom signals; post-blackout
# lets the 10-min rolling window re-anchor entirely on post-cut data before
# the next entry is allowed (post_min must be >= WINDOW).
FOMC_PRE_MIN  = 60      # halt entries 60 min before announcement (17:00 UTC)
FOMC_POST_MIN = 30      # resume entries 30 min after announcement (18:30 UTC)

# ── Volume regime gate ────────────────────────────────────────────────────────
# Gate open when daily back-contract volume share is in (LOW, HIGH).
# LOW  = 0.05: roll has begun but not yet dominated by back-month flow
# HIGH = 0.80: most OI/volume has migrated; spread goes quiet → exit
# At 5%/80% this naturally includes Thu1 (7.1%) and excludes Sep 18 (83.2%)
# and Sep 19 (86.2%), making the FOMC calendar rule redundant for daily gating.
VOL_GATE_LOW  = 0.05
VOL_GATE_HIGH = 0.80

# ── Contract mechanics ────────────────────────────────────────────────────────
TICK = 0.25
MULT = 50.0
TICKV = TICK * MULT     # $12.50

# ── Signal parameters ─────────────────────────────────────────────────────────
WINDOW    = '10min'
THRESHOLD = 2.5
TP        = 0.75        # pts → $37.50 gross
SL        = 0.50        # pts → $25.00 gross
DIV_YIELD = 0.013
FOMC_CUT  = 0.005       # 50bps

# ── Costs ─────────────────────────────────────────────────────────────────────
TC_INST   = 4.00        # institutional round-trip, 2 legs
TC_RETAIL = 7.40        # IB retail round-trip, 2 legs
SLIP = {'Tight': 6.25, 'Mid': 12.50, 'Wide': 25.00}
ALLIN_INST   = {k: v + TC_INST   for k, v in SLIP.items()}
ALLIN_RETAIL = SLIP['Tight'] + TC_RETAIL    # tight slippage + retail TC

# ── Style ─────────────────────────────────────────────────────────────────────
NAVY   = '#0d1b2a'
STEEL  = '#2471a3'
GREEN  = '#1e8449'
RED    = '#c0392b'
ORANGE = '#e67e22'
GOLD   = '#f39c12'
LGRAY  = '#f7f9fb'
MGRAY  = '#d0d3d4'
DGRAY  = '#555555'
WHITE  = '#ffffff'

DAY_COLORS = {
    '2024-09-12': '#4a90d9',   # Thu1
    '2024-09-13': '#e74c3c',   # Fri
    '2024-09-15': '#95a5a6',   # Sun
    '2024-09-16': '#e67e22',   # Mon (crossover)
    '2024-09-17': '#8e44ad',   # Tue
    '2024-09-18': '#27ae60',   # Wed FOMC
    '2024-09-19': '#16a085',   # Thu2
}
DAY_LABELS = {
    '2024-09-12': 'Thu Sep 12',
    '2024-09-13': 'Fri Sep 13',
    '2024-09-15': 'Sun Sep 15',
    '2024-09-16': 'Mon Sep 16*',
    '2024-09-17': 'Tue Sep 17',
    '2024-09-18': 'Wed Sep 18†',
    '2024-09-19': 'Thu Sep 19',
}

plt.rcParams.update({
    'font.family':        'DejaVu Sans',
    'font.size':          8.5,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.color':         MGRAY,
    'grid.alpha':         0.5,
    'grid.linewidth':     0.5,
    'axes.labelcolor':    DGRAY,
    'xtick.color':        DGRAY,
    'ytick.color':        DGRAY,
    'axes.titlepad':      6,
})


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING  (same infrastructure as 05_robustness_globex.py)
# ─────────────────────────────────────────────────────────────────────────────
def _load_sofr_daily() -> pd.Series:
    raw = pd.read_csv(DATA_DIR / 'SOFR.csv', parse_dates=['observation_date'],
                      index_col='observation_date')
    s = raw.iloc[:, 0].dropna() / 100.0
    return pd.Series(s.values, index=pd.DatetimeIndex(s.index).tz_localize('UTC'))


def _load_volume_gate() -> dict:
    """
    Returns {calendar_date: gate_open} from daily ohlcv1d volume back-share.

    Gate open when VOL_GATE_LOW < back_share < VOL_GATE_HIGH.
    Uses same-day volume (appropriate for backtesting; for live use prior-day).
    Falls back to all-open if ohlcv1d files are missing.
    """
    files = sorted(glob.glob(str(DATA_DIR / f'ohlcv1d_{FRONT}_{BACK}_*.parquet')))
    if not files:
        print('  WARNING: no ohlcv1d files found — volume gate disabled (all days open)')
        return {}
    vol = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    piv = vol.pivot_table(index=vol.index.date, columns='symbol',
                          values='volume', aggfunc='sum')
    front_v = piv[FRONT] if FRONT in piv.columns else pd.Series(0, index=piv.index)
    back_v  = piv[BACK]  if BACK  in piv.columns else pd.Series(0, index=piv.index)
    back_share = back_v / (front_v + back_v + 1e-9)
    gate = {d: bool(VOL_GATE_LOW < float(s) < VOL_GATE_HIGH)
            for d, s in back_share.items()}
    # Print gate status for transparency
    for d in sorted(gate):
        ts = pd.Timestamp(str(d))
        if ts >= pd.Timestamp(ROLL_START) - pd.Timedelta('3d'):
            status = 'OPEN ✓' if gate[d] else ('LOW  –' if float(back_share.get(d, 0)) <= VOL_GATE_LOW else 'HIGH ✗')
            print(f'    {d}  {ts.day_name()[:3]}  back_share={float(back_share.get(d,0)):.1%}  gate={status}')
    return gate


def _build_fv(spread: pd.Series, front_mid: pd.Series,
              sofr_utc: pd.Series, dt_yr: float) -> pd.Series:
    daily_idx  = pd.date_range(spread.index[0].normalize(),
                               spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()
    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    ).ffill()
    pre = r_f.index < FOMC_UTC
    if pre.any():
        pre_rate = float(r_f[pre].iloc[-1])
        r_f[~pre] = pre_rate - FOMC_CUT
    return front_mid.reindex(r_f.index).ffill() * (r_f - DIV_YIELD) * dt_yr


def load_rth_data(sofr_utc: pd.Series, dt_yr: float):
    COLS  = ['bid_px_00', 'ask_px_00', 'symbol']
    files = sorted(glob.glob(
        str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    print(f'  Loading {len(files)} RTH day files', end='', flush=True)
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=COLS)
        df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        wide = (df.groupby('symbol')[['mid']]
                .resample('1s').last().ffill()
                .unstack('symbol')
                .between_time(RTH_OPEN, RTH_CLOSE))
        parts.append(wide)
        print('.', end='', flush=True)
    print(' done')
    full      = pd.concat(parts).sort_index()
    spread    = (full[('mid', BACK)] - full[('mid', FRONT)]).dropna()
    front_mid = full[('mid', FRONT)].reindex(spread.index).ffill()
    fv        = _build_fv(spread, front_mid, sofr_utc, dt_yr)
    dev       = (spread - fv).dropna()
    return spread.reindex(dev.index), fv.reindex(dev.index), dev


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION — enhanced to track MAE / MFE / entry_z / exit_type
# ─────────────────────────────────────────────────────────────────────────────
def _compute_z(dev: pd.Series) -> np.ndarray:
    mu  = dev.rolling(WINDOW, min_periods=1).mean()
    sig = dev.rolling(WINDOW, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


def _build_entry_mask(spread: pd.Series, vol_gate: dict) -> np.ndarray:
    """
    Returns bool array: True = new entry signal allowed at this bar.

    Blocked by:
      1. Volume regime gate: daily back-share outside (VOL_GATE_LOW, VOL_GATE_HIGH).
         Excludes pre-roll days (ESZ4 not yet liquid) and post-roll days (spread
         going quiet after most OI has migrated). At (5%, 80%) this naturally
         closes the gate on Sep 18 (83.2%) and Sep 19 (86.2%) without any
         calendar rule.
      2. Friday open filter: first FRI_SKIP_MIN of any Friday RTH.
      3. FOMC blackout: FOMC_PRE_MIN before to FOMC_POST_MIN after announcement.
         Belt-and-suspenders for windows where FOMC falls earlier in the roll
         week before the volume gate would close.
    """
    idx  = spread.index
    mask = np.ones(len(idx), dtype=bool)

    # ── Volume regime gate (daily) ────────────────────────────────────────────
    if vol_gate:
        dates = np.array([t.date() for t in idx])
        for i, d in enumerate(dates):
            if not vol_gate.get(d, True):   # default open if date not in gate
                mask[i] = False

    # ── Friday opening filter ─────────────────────────────────────────────────
    fri_open_t = pd.Timestamp('2000-01-01 12:30').time()
    fri_cut_t  = (pd.Timestamp('2000-01-01 12:30') +
                  pd.Timedelta(minutes=FRI_SKIP_MIN)).time()
    is_fri     = (idx.weekday == 4)
    in_fri_open = (idx.time >= fri_open_t) & (idx.time < fri_cut_t)
    mask[is_fri & in_fri_open] = False

    # ── FOMC blackout ─────────────────────────────────────────────────────────
    fomc_block_start = FOMC_UTC - pd.Timedelta(minutes=FOMC_PRE_MIN)
    fomc_block_end   = FOMC_UTC + pd.Timedelta(minutes=FOMC_POST_MIN)
    in_fomc_block    = (idx >= fomc_block_start) & (idx < fomc_block_end)
    mask[in_fomc_block] = False

    return mask


def simulate(spread: pd.Series, dev: pd.Series, z: np.ndarray,
             entry_mask: np.ndarray) -> pd.DataFrame:
    """
    Edge-triggered simulation with TP/SL exits.
    Tracks MAE, MFE, entry_z, and exit_type for each trade.

    Force-close events (in priority order):
      1. FOMC announcement bar — close any open position before the move
      2. RTH end-of-day bar   — no overnight carry
      3. TP / SL              — profit target or stop loss
    """
    prices = spread.values
    times  = spread.index
    n      = len(prices)

    dates   = np.array([t.date() for t in times])
    is_last = np.zeros(n, dtype=bool)
    is_last[-1] = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last[i] = True

    # Force-close at FOMC announcement bar (first bar at or after FOMC_UTC)
    fomc_close = np.zeros(n, dtype=bool)
    fomc_idx   = np.searchsorted(times, FOMC_UTC)
    if fomc_idx < n:
        fomc_close[fomc_idx] = True

    trades   = []
    pos      = 0
    entry_i  = -1
    entry_px = np.nan
    entry_z  = np.nan
    min_px   = np.nan
    max_px   = np.nan
    bars_held = 0
    pending_dir = 0

    for i in range(n):
        zi = z[i]
        px = prices[i]

        # ── Fill pending entry ─────────────────────────────────────────────
        if pending_dir != 0 and pos == 0:
            pos      = pending_dir
            entry_i  = i
            entry_px = px
            entry_z  = z[i]
            min_px   = px
            max_px   = px
            bars_held = 0
            pending_dir = 0

        # ── Manage open position ───────────────────────────────────────────
        if pos != 0:
            bars_held += 1
            min_px = min(min_px, px)
            max_px = max(max_px, px)

            move = pos * (px - entry_px)
            exit_type = None
            if fomc_close[i]:
                exit_type = 'FOMC'   # force-close at announcement bar
            elif is_last[i]:
                exit_type = 'EOD'
            elif move >= TP:
                exit_type = 'TP'
            elif move <= -SL:
                exit_type = 'SL'

            if exit_type:
                gross_pts = pos * (px - entry_px)
                gross_usd = gross_pts * MULT
                if pos == 1:
                    mae = max(0.0, entry_px - min_px)   # adverse = price fell
                    mfe = max(0.0, max_px - entry_px)   # favorable = price rose
                else:
                    mae = max(0.0, max_px - entry_px)
                    mfe = max(0.0, entry_px - min_px)

                et = times[entry_i]
                trades.append({
                    'entry_time':    et,
                    'exit_time':     times[i],
                    'direction':     pos,
                    'dir_label':     'Long' if pos == 1 else 'Short',
                    'entry_spread':  entry_px,
                    'exit_spread':   px,
                    'entry_z':       float(entry_z),
                    'gross_pts':     gross_pts,
                    'gross_usd':     gross_usd,
                    'bars_held':     bars_held,
                    'hold_min':      bars_held / 60.0,
                    'exit_type':     exit_type,
                    'mae_pts':       mae,
                    'mfe_pts':       mfe,
                    'trade_date':    et.strftime('%Y-%m-%d'),
                    'day_label':     et.strftime('%a'),
                    'entry_hour_utc': et.hour + et.minute / 60.0,
                    'post_fomc':     et >= FOMC_UTC,
                })
                pos = 0; entry_i = -1; bars_held = 0
                min_px = max_px = entry_px = entry_z = np.nan
                pending_dir = 0

        # ── Detect entry signal ───────────────────────────────────────────
        if pos == 0 and pending_dir == 0 and i > 0 and entry_mask[i]:
            zp = z[i - 1]
            if not np.isnan(zi) and not np.isnan(zp):
                if zp >= -THRESHOLD and zi < -THRESHOLD:
                    pending_dir =  1
                elif zp <= THRESHOLD and zi > THRESHOLD:
                    pending_dir = -1

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    for lbl, cost in SLIP.items():
        df[f'net_{lbl}'] = df['gross_usd'] - cost - TC_INST
    df['net_retail'] = df['gross_usd'] - SLIP['Tight'] - TC_RETAIL

    df['cum_gross']     = df['gross_usd'].cumsum()
    df['peak_gross']    = df['cum_gross'].cummax()
    df['drawdown']      = df['cum_gross'] - df['peak_gross']
    df['cum_net_tight'] = df['net_Tight'].cumsum()
    df['cum_win_rate']  = (df['gross_usd'] > 0).cumsum() / (np.arange(len(df)) + 1) * 100
    df['trade_num']     = np.arange(1, len(df) + 1)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_ci(vals, n_boot=5000, ci=0.95):
    if len(vals) < 2:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(vals, size=len(vals), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def compute_stats(df: pd.DataFrame) -> dict:
    n     = len(df)
    gross = df['gross_usd']
    net_t = df['net_Tight']
    wins  = df[gross > 0]
    losses = df[gross <= 0]

    w_sum = wins['gross_usd'].sum()
    l_sum = abs(losses['gross_usd'].sum()) if len(losses) > 0 else 1e-9

    t_g, p_g = spstats.ttest_1samp(gross, 0.0)
    t_n, p_n = spstats.ttest_1samp(net_t, 0.0)
    ci_lo, ci_hi = bootstrap_ci(net_t.values)

    mdd = df['drawdown'].min()

    return {
        'n':            n,
        'wr':           (gross > 0).mean() * 100,
        'pf':           w_sum / l_sum,
        'avg_gross':    gross.mean(),
        'std_gross':    gross.std(),
        'tot_gross':    gross.sum(),
        'best':         gross.max(),
        'worst':        gross.min(),
        'avg_hold_min': df['hold_min'].mean(),
        'med_hold_min': df['hold_min'].median(),
        'max_hold_min': df['hold_min'].max(),
        'tp_pct':       (df['exit_type'] == 'TP').mean() * 100,
        'sl_pct':       (df['exit_type'] == 'SL').mean() * 100,
        'eod_pct':      (df['exit_type'] == 'EOD').mean() * 100,
        'mdd':          mdd,
        'recovery':     gross.sum() / abs(mdd) if mdd < 0 else float('inf'),
        'avg_net_tight': net_t.mean(),
        'tot_net_tight': net_t.sum(),
        'avg_net_mid':   df['net_Mid'].mean(),
        'avg_net_wide':  df['net_Wide'].mean(),
        'avg_net_retail': df['net_retail'].mean(),
        'sharpe_pt':     net_t.mean() / net_t.std() if net_t.std() > 0 else 0,
        't_gross':  t_g,  'p_gross':  p_g,
        't_net':    t_n,  'p_net':    p_n,
        'ci_lo':    ci_lo, 'ci_hi':   ci_hi,
        'long_n':   (df['direction'] ==  1).sum(),
        'short_n':  (df['direction'] == -1).sum(),
        'long_wr':  (df.loc[df['direction']== 1, 'gross_usd'] > 0).mean()*100
                    if (df['direction']== 1).any() else 0.0,
        'short_wr': (df.loc[df['direction']==-1, 'gross_usd'] > 0).mean()*100
                    if (df['direction']==-1).any() else 0.0,
        'long_avg':  df.loc[df['direction']== 1, 'gross_usd'].mean()
                    if (df['direction']== 1).any() else 0.0,
        'short_avg': df.loc[df['direction']==-1, 'gross_usd'].mean()
                    if (df['direction']==-1).any() else 0.0,
        'avg_mfe':  df['mfe_pts'].mean(),
        'avg_mae':  df['mae_pts'].mean(),
        'mfe_geTP': (df['mfe_pts'] >= TP).mean() * 100,
        'mae_geSL': (df['mae_pts'] >= SL).mean() * 100,
    }


# ─────────────────────────────────────────────────────────────────────────────
# HELPER DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def _header_bar(fig, title, subtitle, y_top=0.97, height=0.055):
    """Draw a navy header bar across the top of the figure."""
    ax = fig.add_axes([0.0, y_top - height, 1.0, height])
    ax.set_facecolor(NAVY)
    ax.axis('off')
    ax.text(0.015, 0.60, title,   color=WHITE, fontsize=13, fontweight='bold',
            va='center', transform=ax.transAxes)
    ax.text(0.015, 0.18, subtitle, color=MGRAY, fontsize=7.5,
            va='center', transform=ax.transAxes)
    ax.text(0.985, 0.50,
            f'ESU4/ESZ4  |  Sep 12–19 2024  |  Databento GLBX.MDP3',
            color=MGRAY, fontsize=7, va='center', ha='right',
            transform=ax.transAxes)


def _footer(fig, page_num, total_pages=3, y=0.012):
    fig.text(0.5, y,
             'Backtest results only. Not investment advice. '
             'All figures are single-window (31 trades) and are not statistically significant at the 5%% level on net P&L.',
             ha='center', fontsize=6, color=DGRAY, style='italic')
    fig.text(0.97, y, f'Page {page_num} / {total_pages}',
             ha='right', fontsize=6, color=DGRAY)


def _metrics_box(fig, stats, y_top, height=0.175):
    """Two-column KPI table below the header."""
    ax = fig.add_axes([0.0, y_top - height, 1.0, height])
    ax.set_facecolor(LGRAY)
    ax.axis('off')

    s = stats
    sig_g = '✓ p<0.05' if s['p_gross'] < 0.05 else ('~ p<0.10' if s['p_gross'] < 0.10 else '✗ n.s.')
    sig_n = '✓ p<0.05' if s['p_net']   < 0.05 else ('~ p<0.10' if s['p_net']   < 0.10 else '✗ n.s.')

    left_col = [
        ('TRADE STATISTICS', None),
        ('Total Trades',         f'{s["n"]}'),
        ('Win Rate',             f'{s["wr"]:.1f}%'),
        ('Profit Factor',        f'{s["pf"]:.2f}'),
        ('Avg Hold Time',        f'{s["avg_hold_min"]:.0f} min  (med {s["med_hold_min"]:.0f})'),
        ('Exit — TP / SL / EOD', f'{s["tp_pct"]:.0f}% / {s["sl_pct"]:.0f}% / {s["eod_pct"]:.0f}%'),
        ('Long trades',          f'{s["long_n"]}  (WR {s["long_wr"]:.0f}%  avg ${s["long_avg"]:.0f})'),
        ('Short trades',         f'{s["short_n"]}  (WR {s["short_wr"]:.0f}%  avg ${s["short_avg"]:.0f})'),
    ]
    right_col = [
        ('P&L SUMMARY', None),
        ('Avg Gross / Trade',    f'${s["avg_gross"]:.2f}'),
        ('Total Gross P&L',      f'${s["tot_gross"]:,.0f}'),
        ('Best / Worst Trade',   f'${s["best"]:.0f}  /  ${s["worst"]:.0f}'),
        ('Max Drawdown (gross)', f'${s["mdd"]:,.0f}'),
        ('Recovery Factor',      f'{s["recovery"]:.1f}×'),
        ('Avg Net — Tight',      f'${s["avg_net_tight"]:.2f}  (all-in ${ALLIN_INST["Tight"]:.2f})'),
        ('Avg Net — Mid',        f'${s["avg_net_mid"]:.2f}  (all-in ${ALLIN_INST["Mid"]:.2f})'),
    ]
    stat_col = [
        ('STATISTICAL TESTS', None),
        ('t-stat (gross vs 0)',  f'{s["t_gross"]:.3f}  p={s["p_gross"]:.4f}  {sig_g}'),
        ('t-stat (net vs 0)',    f'{s["t_net"]:.3f}  p={s["p_net"]:.4f}  {sig_n}'),
        ('95% CI avg net (Tight)', f'[${s["ci_lo"]:.1f},  ${s["ci_hi"]:.1f}]'),
        ('Per-trade Sharpe',     f'{s["sharpe_pt"]:.3f}  (net Tight / std)'),
        ('MFE ≥ TP (0.75)',      f'{s["mfe_geTP"]:.0f}% of trades'),
        ('MAE ≥ SL (0.50)',      f'{s["mae_geSL"]:.0f}% of trades'),
        ('Avg MFE / MAE',        f'{s["avg_mfe"]:.3f} pts  /  {s["avg_mae"]:.3f} pts'),
    ]

    def _draw_col(entries, x_label, x_val, y_start, row_h=0.115):
        for idx, (label, val) in enumerate(entries):
            y = y_start - idx * row_h
            if val is None:
                ax.text(x_label, y, label, color=NAVY, fontsize=7.5,
                        fontweight='bold', va='top', transform=ax.transAxes)
            else:
                ax.text(x_label, y, label + ':',
                        color=DGRAY, fontsize=7.5, va='top', transform=ax.transAxes)
                ax.text(x_val, y, val, color=NAVY, fontsize=7.5,
                        fontweight='bold', va='top', ha='right', transform=ax.transAxes)

    _draw_col(left_col,  0.010, 0.250, 0.95)
    _draw_col(right_col, 0.340, 0.615, 0.95)
    _draw_col(stat_col,  0.650, 0.990, 0.95)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW: EQUITY CURVE + DRAWDOWN
# ─────────────────────────────────────────────────────────────────────────────
def page1(df: pd.DataFrame, stats: dict, pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)

    _header_bar(fig,
                'ESU4/ESZ4 Calendar Spread — Mean Reversion Strategy',
                '10-min z-score  |  z > ±2.5σ entry  |  TP = +0.75 pt  |  SL = −0.50 pt  |  RTH only  |  Friday open 30-min filtered')
    _metrics_box(fig, stats, y_top=0.915)
    _footer(fig, 1)

    chart_top = 0.735
    gs = gridspec.GridSpec(2, 1, top=chart_top, bottom=0.06,
                           hspace=0.08, height_ratios=[3, 1],
                           left=0.06, right=0.97)

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    x   = df['trade_num'].values

    ax1.fill_between(x, df['cum_gross'], 0,
                     where=df['cum_gross'] >= 0, alpha=0.12, color=GREEN)
    ax1.fill_between(x, df['cum_gross'], 0,
                     where=df['cum_gross'] < 0,  alpha=0.12, color=RED)
    ax1.plot(x, df['cum_gross'],     color=STEEL,  lw=2.0, label='Cumulative Gross')
    ax1.plot(x, df['cum_net_tight'], color=GREEN,  lw=1.5, ls='--',
             label=f'Cumulative Net (Tight  all-in ${ALLIN_INST["Tight"]:.2f})')

    net_mid_cum = df['net_Mid'].cumsum()
    ax1.plot(x, net_mid_cum, color=ORANGE, lw=1.2, ls=':',
             label=f'Cumulative Net (Mid  all-in ${ALLIN_INST["Mid"]:.2f})')
    ax1.axhline(0, color=DGRAY, lw=0.7, ls='-')

    # Annotate FOMC threshold
    if df['post_fomc'].any():
        fomc_row = df[df['post_fomc']].iloc[0]
        fomc_x   = int(fomc_row['trade_num'])
        y_ann    = df['cum_gross'].iloc[:fomc_x - 1].iloc[-1] if fomc_x > 1 else 0
        ax1.axvline(fomc_x, color=GOLD, lw=1.2, ls='--', alpha=0.8)
        ax1.text(fomc_x + 0.3, y_ann + 15, 'FOMC\n−50bps',
                 color=GOLD, fontsize=7, va='bottom')

    ax1.set_ylabel('Cumulative P&L  ($)', color=DGRAY)
    ax1.set_title('Equity Curve — Gross and Net P&L by Trade Sequence', fontsize=9.5, color=NAVY)
    ax1.legend(fontsize=7.5, loc='upper left', framealpha=0.85)
    ax1.yaxis.set_major_formatter(lambda v, _: f'${v:,.0f}')
    ax1.set_xlim(1, len(df))
    ax1.set_xlabel('')

    # Annotate final totals
    ax1.annotate(f'Gross: ${df["cum_gross"].iloc[-1]:,.0f}',
                 xy=(len(df), df['cum_gross'].iloc[-1]),
                 xytext=(len(df) - 3, df['cum_gross'].iloc[-1] + 20),
                 fontsize=7, color=STEEL, ha='right',
                 arrowprops=dict(arrowstyle='->', color=STEEL, lw=0.8))

    # ── Drawdown ─────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.fill_between(x, df['drawdown'], 0,
                     where=df['drawdown'] <= 0, alpha=0.35, color=RED)
    ax2.plot(x, df['drawdown'], color=RED, lw=1.2)
    ax2.axhline(0, color=DGRAY, lw=0.6)
    ax2.set_ylabel('Drawdown  ($)', color=DGRAY, fontsize=8)
    ax2.set_xlabel('Trade Number', color=DGRAY)
    ax2.set_title('Drawdown from Peak (Gross)', fontsize=9, color=NAVY)
    ax2.yaxis.set_major_formatter(lambda v, _: f'${v:,.0f}')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — TRADE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page2(df: pd.DataFrame, stats: dict, pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)
    _header_bar(fig, 'Trade Analysis', 'P&L Distribution  |  Hold Times  |  Direction  |  Cost Sensitivity')
    _footer(fig, 2)

    gs = gridspec.GridSpec(2, 3, top=0.88, bottom=0.08,
                           hspace=0.40, wspace=0.38,
                           left=0.07, right=0.97)

    # ── 1. Gross P&L histogram ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    g    = df['gross_usd'].values
    bmin = int(np.floor(min(g) / 5) * 5) - 5
    bmax = int(np.ceil(max(g) / 5) * 5) + 5
    bins = np.arange(bmin, bmax + 5, 5)
    _, _, patches = ax.hist(g, bins=bins, edgecolor='white', linewidth=0.4, alpha=0.85)
    for patch, left in zip(patches, bins[:-1]):
        patch.set_facecolor(GREEN if left + 2.5 >= 0 else RED)
    for lbl, cost, col in [('Tight', ALLIN_INST['Tight'], GREEN),
                            ('Mid',   ALLIN_INST['Mid'],   ORANGE)]:
        ax.axvline(cost, color=col, lw=1.3, ls='--', label=f'BE {lbl} ${cost:.0f}')
    ax.axvline(0, color=DGRAY, lw=0.7)
    ax.set_title('Gross P&L Distribution', color=NAVY, fontsize=9)
    ax.set_xlabel('Gross P&L per trade  ($)')
    ax.set_ylabel('Trade count')
    ax.legend(fontsize=7)
    ax.text(0.98, 0.97, f'μ=${stats["avg_gross"]:.0f}\nσ=${stats["std_gross"]:.0f}',
            transform=ax.transAxes, va='top', ha='right', fontsize=7.5, color=NAVY,
            bbox=dict(boxstyle='round,pad=0.3', fc=LGRAY, ec=MGRAY))

    # ── 2. Hold time histogram ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    hold = df['hold_min'].values
    bins_h = np.arange(0, max(hold) + 15, 15)
    ax.hist(hold, bins=bins_h, color=STEEL, edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(stats['avg_hold_min'], color=ORANGE, lw=1.5, ls='--',
               label=f'Mean {stats["avg_hold_min"]:.0f} min')
    ax.axvline(stats['med_hold_min'], color=GREEN,  lw=1.5, ls=':',
               label=f'Median {stats["med_hold_min"]:.0f} min')
    ax.set_title('Hold Time Distribution', color=NAVY, fontsize=9)
    ax.set_xlabel('Hold time  (minutes)')
    ax.set_ylabel('Trade count')
    ax.legend(fontsize=7)

    # ── 3. Exit type breakdown ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    exit_types = ['TP', 'SL', 'EOD', 'FOMC']
    exit_cols  = [GREEN, RED, ORANGE, GOLD]
    for et, col in zip(exit_types, exit_cols):
        sub = df[df['exit_type'] == et]
        if sub.empty:
            continue
        avg = sub['gross_usd'].mean()
        cnt = len(sub)
        ax.bar(et, avg, color=col, alpha=0.85, edgecolor='white', width=0.55)
        ax.text(et, avg + (3 if avg >= 0 else -6),
                f'n={cnt}\n${avg:.0f}', ha='center', fontsize=7.5, color=NAVY)
    ax.axhline(0, color=DGRAY, lw=0.7)
    ax.axhline(ALLIN_INST['Tight'], color=GREEN, lw=1.2, ls='--', alpha=0.7,
               label=f'BE Tight ${ALLIN_INST["Tight"]:.0f}')
    ax.set_title('Avg Gross P&L by Exit Type', color=NAVY, fontsize=9)
    ax.set_ylabel('Avg gross P&L  ($)')
    ax.legend(fontsize=7)

    # ── 4. Long vs Short breakdown ────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    dirs = ['Long', 'Short']
    for j, (d, col) in enumerate(zip([1, -1], [STEEL, ORANGE])):
        sub = df[df['direction'] == d]
        if sub.empty:
            continue
        avg = sub['gross_usd'].mean()
        cnt = len(sub)
        wr  = (sub['gross_usd'] > 0).mean() * 100
        ax.bar(dirs[j], avg, color=col, alpha=0.85, edgecolor='white', width=0.55)
        ax.text(dirs[j], avg + (3 if avg >= 0 else -6),
                f'n={cnt}  WR={wr:.0f}%\n${avg:.0f}', ha='center', fontsize=7.5, color=NAVY)
    ax.axhline(0, color=DGRAY, lw=0.7)
    ax.axhline(ALLIN_INST['Tight'], color=GREEN, lw=1.2, ls='--', alpha=0.7,
               label=f'BE Tight ${ALLIN_INST["Tight"]:.0f}')
    ax.set_title('Avg Gross P&L — Long vs Short', color=NAVY, fontsize=9)
    ax.set_ylabel('Avg gross P&L  ($)')
    ax.legend(fontsize=7)

    # ── 5. Cumulative win rate ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(df['trade_num'], df['cum_win_rate'], color=STEEL, lw=1.8)
    ax.axhline(50,  color=RED,   lw=0.8, ls='--', alpha=0.7, label='50% (random)')
    ax.axhline(stats['wr'], color=GREEN, lw=1.2, ls=':',
               label=f'Final WR {stats["wr"]:.1f}%')
    ax.set_ylim(0, 100)
    ax.set_title('Rolling Win Rate', color=NAVY, fontsize=9)
    ax.set_xlabel('Trade Number')
    ax.set_ylabel('Win rate  (%)')
    ax.legend(fontsize=7)

    # ── 6. Cost sensitivity bar chart ─────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    scenarios = [
        ('Gross',          df['gross_usd'].mean(),     STEEL),
        ('Net Tight\n(inst.)', df['net_Tight'].mean(), GREEN),
        ('Net Mid\n(inst.)',   df['net_Mid'].mean(),   ORANGE),
        ('Net Wide\n(inst.)',  df['net_Wide'].mean(),  GOLD),
        ('Net Tight\n(retail)',df['net_retail'].mean(),RED),
    ]
    for i, (lbl, val, col) in enumerate(scenarios):
        ax.bar(i, val, color=col, alpha=0.85, edgecolor='white', width=0.65)
        va = 'bottom' if val >= 0 else 'top'
        offset = 0.5 if val >= 0 else -0.5
        ax.text(i, val + offset, f'${val:.1f}', ha='center', fontsize=7.5,
                color=NAVY, va=va)
    ax.axhline(0, color=DGRAY, lw=0.7)
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s[0] for s in scenarios], fontsize=7.5)
    ax.set_title('Avg P&L per Trade by Cost Scenario', color=NAVY, fontsize=9)
    ax.set_ylabel('Avg P&L  ($)')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — TEMPORAL & RISK ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
def page3(df: pd.DataFrame, stats: dict, pdf: PdfPages):
    fig = plt.figure(figsize=(11, 8.5))
    fig.patch.set_facecolor(WHITE)
    _header_bar(fig, 'Temporal & Risk Analysis',
                'Per-Day Breakdown  |  Entry Timing  |  MAE / MFE  |  Entry Z-Score')
    _footer(fig, 3)

    gs = gridspec.GridSpec(2, 3, top=0.88, bottom=0.08,
                           hspace=0.42, wspace=0.40,
                           left=0.07, right=0.97)

    # ── 1. Per-day performance bar chart ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, :2])  # span 2 columns
    dates_ord = sorted(df['trade_date'].unique())
    xs = np.arange(len(dates_ord))
    w  = 0.30

    gross_by_day = df.groupby('trade_date')['gross_usd'].mean().reindex(dates_ord).fillna(0)
    net_by_day   = df.groupby('trade_date')['net_Tight'].mean().reindex(dates_ord).fillna(0)
    n_by_day     = df.groupby('trade_date').size().reindex(dates_ord).fillna(0)

    cols = [DAY_COLORS.get(d, STEEL) for d in dates_ord]
    bars_g = ax.bar(xs - w/2, gross_by_day, w, color=cols, alpha=0.85,
                    edgecolor='white', label='Avg Gross')
    bars_n = ax.bar(xs + w/2, net_by_day,   w, color=cols, alpha=0.45,
                    edgecolor='white', label='Avg Net (Tight)')

    for i, (xp, n, g, nt) in enumerate(zip(xs, n_by_day, gross_by_day, net_by_day)):
        ax.text(xp, max(g, nt, 0) + 1.5, f'n={int(n)}',
                ha='center', fontsize=7, color=NAVY)

    ax.axhline(0, color=DGRAY, lw=0.7)
    ax.axhline(ALLIN_INST['Tight'], color=GREEN, lw=1.2, ls='--', alpha=0.7,
               label=f'BE Tight ${ALLIN_INST["Tight"]:.0f}')
    ax.set_xticks(xs)
    ax.set_xticklabels([DAY_LABELS.get(d, d) for d in dates_ord], fontsize=7.5, rotation=15)
    ax.set_ylabel('Avg gross / net P&L  ($)')
    ax.set_title('Per-Day Performance  (* = volume crossover   † = FOMC −50bps)',
                 color=NAVY, fontsize=9)
    ax.legend(fontsize=7.5)

    # ── 2. Pre/Post FOMC ─────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    for j, (label, mask, col) in enumerate([
        ('Pre-FOMC',  ~df['post_fomc'], STEEL),
        ('Post-FOMC',  df['post_fomc'], GOLD),
    ]):
        sub = df[mask]
        if sub.empty:
            continue
        avg = sub['gross_usd'].mean()
        cnt = len(sub)
        wr  = (sub['gross_usd'] > 0).mean() * 100
        ax.bar(j, avg, color=col, alpha=0.85, edgecolor='white', width=0.55)
        ax.text(j, avg + (2 if avg >= 0 else -4),
                f'n={cnt}  WR={wr:.0f}%\n${avg:.0f}', ha='center', fontsize=7.5, color=NAVY)
    ax.axhline(0, color=DGRAY, lw=0.7)
    ax.axhline(ALLIN_INST['Tight'], color=GREEN, lw=1.2, ls='--', alpha=0.7)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Pre-FOMC', 'Post-FOMC'], fontsize=8.5)
    ax.set_title('Pre vs Post FOMC\n(Sep 18, 18:00 UTC)', color=NAVY, fontsize=9)
    ax.set_ylabel('Avg gross P&L  ($)')

    # ── 3. MFE distribution ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    mfe = df['mfe_pts'].values
    bins_m = np.arange(0, max(mfe) + 0.125, 0.125)
    ax.hist(mfe, bins=bins_m, color=GREEN, edgecolor='white', linewidth=0.4, alpha=0.85)
    ax.axvline(TP, color=RED, lw=1.8, ls='--', label=f'TP = {TP} pts')
    ax.axvline(np.mean(mfe), color=NAVY, lw=1.2, ls=':', label=f'Mean {np.mean(mfe):.2f} pts')
    ax.set_title('Max Favorable Excursion (MFE)', color=NAVY, fontsize=9)
    ax.set_xlabel('MFE  (pts in trade direction)')
    ax.set_ylabel('Trade count')
    ax.legend(fontsize=7)
    pct = (mfe >= TP).mean() * 100
    ax.text(0.98, 0.97, f'{pct:.0f}% reached TP',
            transform=ax.transAxes, va='top', ha='right', fontsize=7.5, color=GREEN,
            bbox=dict(boxstyle='round,pad=0.3', fc=LGRAY, ec=MGRAY))

    # ── 4. MAE distribution ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    mae = df['mae_pts'].values
    bins_a = np.arange(0, max(mae) + 0.125, 0.125)
    w_mae = df.loc[df['gross_usd'] > 0, 'mae_pts'].values
    l_mae = df.loc[df['gross_usd'] <= 0, 'mae_pts'].values
    ax.hist(w_mae, bins=bins_a, color=GREEN, edgecolor='white', linewidth=0.4,
            alpha=0.70, label='Winners')
    ax.hist(l_mae, bins=bins_a, color=RED,   edgecolor='white', linewidth=0.4,
            alpha=0.70, label='Losers')
    ax.axvline(SL, color=NAVY, lw=1.8, ls='--', label=f'SL = {SL} pts')
    ax.set_title('Max Adverse Excursion (MAE)\nWinners vs Losers', color=NAVY, fontsize=9)
    ax.set_xlabel('MAE  (pts against trade direction)')
    ax.set_ylabel('Trade count')
    ax.legend(fontsize=7)

    # ── 5. Entry z-score distribution ────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    ez = df['entry_z'].values
    bins_z = np.arange(min(ez) - 0.2, max(ez) + 0.4, 0.2)
    w_ez = df.loc[df['gross_usd'] > 0,  'entry_z'].values
    l_ez = df.loc[df['gross_usd'] <= 0, 'entry_z'].values
    ax.hist(w_ez, bins=bins_z, color=GREEN, edgecolor='white', linewidth=0.4,
            alpha=0.70, label='Winners')
    ax.hist(l_ez, bins=bins_z, color=RED,   edgecolor='white', linewidth=0.4,
            alpha=0.70, label='Losers')
    ax.axvline( THRESHOLD, color=NAVY, lw=1.5, ls='--', alpha=0.8, label=f'Short threshold +{THRESHOLD}σ')
    ax.axvline(-THRESHOLD, color=NAVY, lw=1.5, ls='--', alpha=0.8, label=f'Long threshold −{THRESHOLD}σ')
    ax.set_title('Entry Z-Score at Fill Bar\n(Winners vs Losers)', color=NAVY, fontsize=9)
    ax.set_xlabel('Z-score at entry fill bar')
    ax.set_ylabel('Trade count')
    ax.legend(fontsize=6.5)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# PRINT CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame, stats: dict):
    SEP = '─' * 100
    print()
    print('═' * 100)
    print('  ESU4/ESZ4 TEAR SHEET — CONSOLE SUMMARY')
    print('═' * 100)
    print(f'  Config: {WINDOW} z-score  |  z > ±{THRESHOLD}σ  |  TP={TP}pt  |  SL={SL}pt  |  Friday open {FRI_SKIP_MIN}-min filtered')
    print(SEP)
    print(f'  TOTAL TRADES     : {stats["n"]}')
    print(f'  WIN RATE         : {stats["wr"]:.1f}%')
    print(f'  PROFIT FACTOR    : {stats["pf"]:.2f}')
    fomc_pct = (df['exit_type'] == 'FOMC').mean() * 100
    print(f'  EXIT BREAKDOWN   : TP={stats["tp_pct"]:.0f}%  SL={stats["sl_pct"]:.0f}%  '
          f'EOD={stats["eod_pct"]:.0f}%  FOMC={fomc_pct:.0f}%')
    print(f'  AVG HOLD TIME    : {stats["avg_hold_min"]:.0f} min  (median {stats["med_hold_min"]:.0f}  max {stats["max_hold_min"]:.0f})')
    print(SEP)
    print(f'  AVG GROSS/TRADE  : ${stats["avg_gross"]:.2f}')
    print(f'  TOTAL GROSS P&L  : ${stats["tot_gross"]:,.0f}')
    print(f'  BEST / WORST     : ${stats["best"]:.2f}  /  ${stats["worst"]:.2f}')
    print(f'  MAX DRAWDOWN     : ${stats["mdd"]:,.0f}')
    print(f'  RECOVERY FACTOR  : {stats["recovery"]:.1f}x')
    print(SEP)
    print(f'  NET (TIGHT inst.): ${stats["avg_net_tight"]:.2f}/trade   total ${stats["tot_net_tight"]:,.0f}'
          f'   (all-in ${ALLIN_INST["Tight"]:.2f})')
    print(f'  NET (MID   inst.): ${stats["avg_net_mid"]:.2f}/trade'
          f'   (all-in ${ALLIN_INST["Mid"]:.2f})')
    print(f'  NET (WIDE  inst.): ${stats["avg_net_wide"]:.2f}/trade'
          f'   (all-in ${ALLIN_INST["Wide"]:.2f})')
    print(f'  NET (TIGHT retail): ${stats["avg_net_retail"]:.2f}/trade'
          f'   (all-in ${ALLIN_RETAIL:.2f})')
    print(SEP)
    print(f'  t-stat GROSS     : {stats["t_gross"]:.3f}  p={stats["p_gross"]:.4f}'
          f'  {"✓ p<0.05" if stats["p_gross"]<0.05 else "n.s."}')
    print(f'  t-stat NET       : {stats["t_net"]:.3f}  p={stats["p_net"]:.4f}'
          f'  {"✓ p<0.05" if stats["p_net"]<0.05 else "n.s. — 95% CI straddles $0"}')
    print(f'  95% CI NET TIGHT : [${stats["ci_lo"]:.1f},  ${stats["ci_hi"]:.1f}]')
    print(f'  PER-TRADE SHARPE : {stats["sharpe_pt"]:.3f}')
    print(SEP)
    print(f'  LONG  : {stats["long_n"]} trades  WR={stats["long_wr"]:.0f}%  avg ${stats["long_avg"]:.2f}')
    print(f'  SHORT : {stats["short_n"]} trades  WR={stats["short_wr"]:.0f}%  avg ${stats["short_avg"]:.2f}')
    print(SEP)
    print(f'  AVG MFE : {stats["avg_mfe"]:.3f} pts   MFE ≥ TP in {stats["mfe_geTP"]:.0f}% of trades')
    print(f'  AVG MAE : {stats["avg_mae"]:.3f} pts   MAE ≥ SL in {stats["mae_geSL"]:.0f}% of trades')
    print(SEP)

    print()
    print(f'  {"Date":<12} {"Day":<8} {"N":>4}  {"WR%":>6}  {"AvgGross":>9}  '
          f'{"AvgNetTight":>12}  {"ExitTP/SL/EOD":>14}  {"AvgHold":>8}')
    print('  ' + '─' * 80)
    for d in sorted(df['trade_date'].unique()):
        sub = df[df['trade_date'] == d]
        day = sub['day_label'].iloc[0]
        tp_n  = (sub['exit_type'] == 'TP').sum()
        sl_n  = (sub['exit_type'] == 'SL').sum()
        eod_n = (sub['exit_type'] == 'EOD').sum()
        print(
            f'  {d}  {day:<8} {len(sub):>4}  '
            f'{(sub["gross_usd"]>0).mean()*100:>5.0f}%  '
            f'${sub["gross_usd"].mean():>8.2f}  '
            f'${sub["net_Tight"].mean():>10.2f}  '
            f'{tp_n:>3}TP/{sl_n:>2}SL/{eod_n:>2}EOD  '
            f'{sub["hold_min"].mean():>7.0f} min'
        )

    # ── FOMC day diagnostic ───────────────────────────────────────────────────
    fomc_day = df[df['trade_date'] == '2024-09-18']
    fomc_pre_entries  = fomc_day[fomc_day['entry_time'] < FOMC_UTC - pd.Timedelta(minutes=FOMC_PRE_MIN)]
    fomc_blk_entries  = fomc_day[(fomc_day['entry_time'] >= FOMC_UTC - pd.Timedelta(minutes=FOMC_PRE_MIN)) &
                                  (fomc_day['entry_time'] <  FOMC_UTC + pd.Timedelta(minutes=FOMC_POST_MIN))]
    fomc_post_entries = fomc_day[fomc_day['entry_time'] >= FOMC_UTC + pd.Timedelta(minutes=FOMC_POST_MIN)]
    fomc_close_trades = fomc_day[fomc_day['exit_type'] == 'FOMC']

    fomc_block_start = (FOMC_UTC - pd.Timedelta(minutes=FOMC_PRE_MIN)).strftime('%H:%M UTC')
    fomc_block_end   = (FOMC_UTC + pd.Timedelta(minutes=FOMC_POST_MIN)).strftime('%H:%M UTC')

    print()
    print(f'  FOMC DAY DIAGNOSTIC  (Sep 18, 2024 — announcement 18:00 UTC = 14:00 ET)')
    print(f'  Blackout: {fomc_block_start} → {fomc_block_end}  '
          f'(−{FOMC_PRE_MIN} min pre / +{FOMC_POST_MIN} min post)')
    print('  ' + '─' * 78)

    def _fomc_row(label, subset):
        if subset.empty:
            print(f'  {label:<42}  —')
            return
        wr  = (subset['gross_usd'] > 0).mean() * 100
        avg = subset['gross_usd'].mean()
        print(f'  {label:<42}  n={len(subset):>3}  WR={wr:>4.0f}%  avg gross ${avg:>8.2f}')

    _fomc_row('Pre-blackout entries  (12:30 – 17:00 UTC)', fomc_pre_entries)
    _fomc_row('Blackout BLOCKED      (17:00 – 18:30 UTC)', fomc_blk_entries)
    _fomc_row('Post-blackout entries (18:30 – 19:15 UTC)', fomc_post_entries)
    _fomc_row('Force-closed AT announcement  (18:00 UTC)', fomc_close_trades)

    print()
    print('  Root cause: stale SOFR (5.33%) overstates FV while market pre-prices the')
    print('  expected cut. Spread looks artificially cheap → phantom long signal.')
    print(f'  Post-blackout z-score is clean: {FOMC_POST_MIN} min blackout > {WINDOW} window.')
    print()
    print('═' * 100)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    results_dir = (Path(__file__).parent.parent / 'results' /
                   f'{FRONT}_{BACK}_{ROLL_START.replace("-", "")}')

    if results_dir.exists() and (results_dir / 'trades.parquet').exists():
        print(f'Loading saved results from {results_dir}/')
        df     = pd.read_parquet(results_dir / 'trades.parquet')
        ts     = pd.read_parquet(results_dir / 'timeseries.parquet')
        fv     = ts['fv']
        spread = ts['spread']
        with open(results_dir / 'stats.json') as f:
            stats = json.load(f)
        print(f'  {len(df)} trades loaded')
    else:
        print('No saved results found — running full pipeline.')
        print('  Tip: run `python notebooks/run_backtest.py --window W1` to save results.')
        print()
        sofr_utc = _load_sofr_daily()

        defn  = pd.read_parquet(DATA_DIR / f'definitions_{FRONT}_{BACK}.parquet')
        exp_f = defn.loc[defn['symbol'] == FRONT, 'expiration'].iloc[0]
        exp_b = defn.loc[defn['symbol'] == BACK,  'expiration'].iloc[0]
        dt_yr = (exp_b - exp_f).total_seconds() / (365.25 * 86400)
        print(f'  ΔT = {dt_yr:.6f} yr  ({(exp_b - exp_f).days} days)')

        spread, fv, dev = load_rth_data(sofr_utc, dt_yr)
        print(f'  {len(spread):,} RTH 1s bars loaded')

        print('Loading volume gate...')
        vol_gate  = _load_volume_gate()
        open_days = sum(1 for v in vol_gate.values() if v)
        print(f'  Gate ({VOL_GATE_LOW:.0%}–{VOL_GATE_HIGH:.0%}): {open_days} active days')

        print('Running simulation...')
        z     = _compute_z(dev)
        emask = _build_entry_mask(spread, vol_gate)
        df    = simulate(spread, dev, z, emask)
        print(f'  {len(df)} trades generated')

        stats = compute_stats(df)

    print_summary(df, stats)

    print(f'\nGenerating tear sheet → {OUT_PDF}')
    with PdfPages(str(OUT_PDF)) as pdf:
        page1(df, stats, pdf)
        page2(df, stats, pdf)
        page3(df, stats, pdf)

        info = pdf.infodict()
        info['Title']   = 'ESU4/ESZ4 Calendar Spread Strategy — Tear Sheet'
        info['Author']  = 'Quantitative Research'
        info['Subject'] = 'ES Futures Roll Window Alpha Strategy Backtest'

    print(f'Done.  File: {OUT_PDF}  ({OUT_PDF.stat().st_size / 1024:.0f} KB)')


if __name__ == '__main__':
    main()
