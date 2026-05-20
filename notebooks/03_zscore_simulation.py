#!/usr/bin/env python3
"""
03_zscore_simulation.py — ESU4/ESZ4 Z-Score Calendar Spread Backtest

Simulates a mean-reversion strategy on the ESZ4-ESU4 calendar spread:
  Signal : rolling z-score of (observed spread − fair-value)
  Entry  : z crosses ±2σ threshold (edge-triggered; filled at NEXT 1s bar)
  Exit   : |z| falls below ±0.5σ (with min 5-bar hold) OR end-of-RTH force-close
  Scope  : RTH only (12:30–19:15 UTC), no overnight positions

Tests 5 rolling z-score windows × 3 slippage scenarios.

Usage:
    cd /Users/stark/Desktop/Projects/Futures_RollOver
    .venv/bin/python notebooks/03_zscore_simulation.py
"""

import glob
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')          # non-interactive; saves to file
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore', category=FutureWarning)
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path('/Volumes/SEAGATE/Databento_Futures')
OUT_DIR    = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Contract / roll window ────────────────────────────────────────────────────
FRONT, BACK = 'ESU4', 'ESZ4'
ROLL_START  = '2024-09-12'
ROLL_END    = '2024-09-19'
FOMC_UTC    = pd.Timestamp('2024-09-18 18:00:00', tz='UTC')

# ── Contract specs ────────────────────────────────────────────────────────────
TICK_SIZE   = 0.25          # index points per tick
MULTIPLIER  = 50.0          # USD per index point
TICK_VALUE  = TICK_SIZE * MULTIPLIER  # $12.50

# ── FV model ──────────────────────────────────────────────────────────────────
DIV_YIELD    = 0.0130        # S&P 500 trailing dividend yield
FOMC_CUT_BPS = 0.0050        # 50bp cut announced at FOMC_UTC

# ── Strategy parameters ───────────────────────────────────────────────────────
Z_WINDOWS  = ['30s', '1min', '2min', '5min', '10min']
ENTRY_Z    = 2.0             # |z| threshold to open a position
EXIT_Z     = 0.5             # |z| threshold to close (mean-reversion complete)
MIN_HOLD   = 5               # minimum bars (seconds) before allowing exit

# RTH session (UTC)
RTH_OPEN  = '12:30'
RTH_CLOSE = '19:15'

# ── Transaction costs ─────────────────────────────────────────────────────────
# Commission: $2.00/contract round-trip (institutional CME+NFA rate)
# Calendar spread = 2 contracts → $4.00 base TC per trade
TC_BASE = 4.00

# 3 slippage scenarios — total round-trip cost for the spread (both legs)
#   Tight : ~half-tick crossing per leg → $6.25 total RT
#   Mid   : 1-tick per leg RT           → $12.50 total RT
#   Wide  : 2-tick per leg RT           → $25.00 total RT
SLIPPAGE = {
    'Tight (½ tick)': 0.5 * TICK_VALUE,    #  $6.25
    'Mid (1 tick)':   1.0 * TICK_VALUE,    # $12.50
    'Wide (2 ticks)': 2.0 * TICK_VALUE,    # $25.00
}
SLIP_LABELS = list(SLIPPAGE.keys())
SLIP_VALS   = list(SLIPPAGE.values())

# Total all-in cost per trade for each scenario
TOTAL_COSTS = {k: v + TC_BASE for k, v in SLIPPAGE.items()}

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_rth_spread() -> tuple[pd.Series, pd.Series]:
    """
    Load all 7 days of mbp-10, resample to 1s midprice, return:
      spread : ESZ4_mid − ESU4_mid  (RTH only, 1s bars, forward-filled)
      ba_sum : sum of bid-ask widths of both legs (for liquidity context)
    """
    COLS = ['bid_px_00', 'ask_px_00', 'symbol']
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    if not files:
        sys.exit(f'ERROR: no mbp10 files found in {DATA_DIR}')

    print(f'Loading {len(files)} day files', end='', flush=True)
    day_frames = []
    for f in files:
        df = pd.read_parquet(f, columns=COLS)
        df['mid']  = (df['bid_px_00'] + df['ask_px_00']) / 2
        df['ba']   = df['ask_px_00'] - df['bid_px_00']
        wide = (
            df.groupby('symbol')[['mid', 'ba']]
            .resample('1s').last().ffill()
            .unstack('symbol')
            .between_time(RTH_OPEN, RTH_CLOSE)
        )
        day_frames.append(wide)
        print('.', end='', flush=True)
    print(' done')

    full  = pd.concat(day_frames).sort_index()
    spread = full[('mid', BACK)] - full[('mid', FRONT)]
    ba_sum = full[('ba', FRONT)] + full[('ba', BACK)]
    return spread.dropna(), ba_sum.reindex(spread.index).fillna(0.5)


def build_fair_value(spread: pd.Series) -> pd.Series:
    """
    Cost-of-carry fair value using daily SOFR (no look-ahead bias).
    FOMC intraday override: post-FOMC_UTC bars use SOFR − 50bp.
    """
    sofr_file = DATA_DIR / 'SOFR.csv'
    sofr_raw  = pd.read_csv(sofr_file, parse_dates=['observation_date'],
                             index_col='observation_date')
    sofr_raw.columns = ['sofr_pct']
    sofr_s = sofr_raw['sofr_pct'].dropna() / 100.0

    # Contract specs (re-derive from data)
    defn  = pd.read_parquet(DATA_DIR / f'definitions_{FRONT}_{BACK}.parquet')
    exp_f = defn.loc[defn['symbol'] == FRONT, 'expiration'].iloc[0]
    exp_b = defn.loc[defn['symbol'] == BACK,  'expiration'].iloc[0]
    dt_yr = (exp_b - exp_f).total_seconds() / (365.25 * 86400)

    # Map each 1s bar to its calendar-day SOFR (forward-fill weekends/holidays)
    sofr_idx = pd.DatetimeIndex(sofr_s.index).tz_localize('UTC')
    sofr_utc = pd.Series(sofr_s.values, index=sofr_idx)
    daily_idx = pd.date_range(spread.index[0].normalize(),
                              spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()

    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    )

    # Intraday FOMC override (announced cut is public information at FOMC_UTC)
    pre_sofr = float(r_f[r_f.index < FOMC_UTC].iloc[-1])
    r_f[r_f.index >= FOMC_UTC] = pre_sofr - FOMC_CUT_BPS

    # Proxy front-month midprice for spot (same source)
    front_1s = pd.read_parquet(
        next(iter(sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet'))))),
        columns=['bid_px_00', 'ask_px_00', 'symbol'],
    )
    # Re-load front mid from spread computation implicitly: use spread + front reconstructed
    # Simpler: load all day files again just for front mid
    cols = ['bid_px_00', 'ask_px_00', 'symbol']
    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    fmids = []
    for f in files:
        df = pd.read_parquet(f, columns=cols)
        fm = (df[df['symbol'] == FRONT]
              .assign(mid=lambda d: (d['bid_px_00'] + d['ask_px_00']) / 2)
              ['mid'].resample('1s').last().ffill()
              .pipe(lambda s: s.between_time(RTH_OPEN, RTH_CLOSE)))
        fmids.append(fm)
    front_mid = pd.concat(fmids).sort_index().reindex(spread.index).ffill()

    fv = front_mid * (r_f - DIV_YIELD) * dt_yr
    return fv, pre_sofr, pre_sofr - FOMC_CUT_BPS, dt_yr


# ─────────────────────────────────────────────────────────────────────────────
# 2. SIMULATION ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def simulate(spread: pd.Series, fv: pd.Series, window: str) -> pd.DataFrame:
    """
    Run edge-triggered z-score strategy for one rolling window.

    Entry rule  : z crosses ±ENTRY_Z → fill at NEXT bar's midprice.
    Exit rule   : |z| < EXIT_Z and bars_held ≥ MIN_HOLD → fill same bar.
                  OR last RTH bar of the session → force-close EOD.
    Direction   : z ↘ below −2 → long spread (ESZ4 cheap; expect it to rise)
                  z ↗ above +2 → short spread (ESZ4 rich; expect it to fall)

    Returns a DataFrame with one row per completed trade.
    """
    dev = spread - fv
    mu  = dev.rolling(window, min_periods=1).mean()
    sig = dev.rolling(window, min_periods=1).std().replace(0, np.nan)
    z   = ((dev - mu) / sig).values

    prices = spread.values
    times  = spread.index
    n      = len(prices)

    # Pre-compute last-bar-of-session flags
    dates          = np.array([t.date() for t in times])
    is_last_of_day = np.zeros(n, dtype=bool)
    is_last_of_day[-1] = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last_of_day[i] = True

    # FOMC bar index (first bar at or after FOMC_UTC)
    fomc_idx = int(np.searchsorted(times, FOMC_UTC))

    trades     = []
    position   = 0       # 0=flat, +1=long, -1=short
    entry_idx  = -1
    entry_px   = np.nan
    bars_held  = 0
    pending_direction = 0   # signal fired at bar i, fill at i+1

    for i in range(n):
        zi = z[i]
        px = prices[i]

        # ── Execute pending entry at this bar (bar after signal) ──────────────
        if pending_direction != 0 and position == 0:
            position  = pending_direction
            entry_idx = i
            entry_px  = px
            bars_held = 0
            pending_direction = 0

        # ── Manage open position ──────────────────────────────────────────────
        if position != 0:
            bars_held += 1
            is_eod = is_last_of_day[i]

            mean_reverted = (not np.isnan(zi) and
                             abs(zi) < EXIT_Z and
                             bars_held >= MIN_HOLD)

            if mean_reverted or is_eod:
                gross_pts = position * (px - entry_px)
                gross_usd = gross_pts * MULTIPLIER
                trades.append({
                    'entry_time'  : times[entry_idx],
                    'exit_time'   : times[i],
                    'direction'   : position,          # +1 long, -1 short
                    'entry_spread': entry_px,
                    'exit_spread' : px,
                    'entry_dev'   : dev.iloc[entry_idx],
                    'exit_dev'    : dev.iloc[i],
                    'entry_z'     : z[entry_idx] if not np.isnan(z[entry_idx]) else np.nan,
                    'exit_z'      : zi,
                    'bars_held'   : bars_held,
                    'gross_pts'   : gross_pts,
                    'gross_usd'   : gross_usd,
                    'eod_close'   : is_eod and not mean_reverted,
                    'post_fomc'   : times[entry_idx] >= FOMC_UTC,
                })
                position      = 0
                entry_idx     = -1
                entry_px      = np.nan
                bars_held     = 0
                pending_direction = 0

        # ── Check for new signal (edge-triggered; only when flat) ─────────────
        if position == 0 and pending_direction == 0 and i > 0:
            zi_prev = z[i - 1]
            if not np.isnan(zi) and not np.isnan(zi_prev):
                if zi_prev >= -ENTRY_Z and zi < -ENTRY_Z:
                    pending_direction = 1    # spread cheapened → go long
                elif zi_prev <= ENTRY_Z and zi > ENTRY_Z:
                    pending_direction = -1   # spread richened → go short

    df = pd.DataFrame(trades)
    if df.empty:
        return df

    # Attach net P&L columns for all slippage scenarios
    for label, slip_cost in SLIPPAGE.items():
        total_cost = slip_cost + TC_BASE
        df[f'net_{label}'] = df['gross_usd'] - total_cost

    # Attach running equity (cumulative gross, for drawdown)
    df['cum_gross'] = df['gross_usd'].cumsum()
    df['peak']      = df['cum_gross'].cummax()
    df['drawdown']  = df['cum_gross'] - df['peak']

    return df


def max_drawdown(trades: pd.DataFrame) -> float:
    """Maximum peak-to-trough drawdown in USD."""
    if trades.empty or 'drawdown' not in trades.columns:
        return 0.0
    return float(trades['drawdown'].min())


def profit_factor(trades: pd.DataFrame) -> float:
    """Sum of wins / |sum of losses| on gross USD."""
    wins   = trades.loc[trades['gross_usd'] > 0, 'gross_usd'].sum()
    losses = trades.loc[trades['gross_usd'] < 0, 'gross_usd'].sum()
    return wins / abs(losses) if losses != 0 else float('inf')


# ─────────────────────────────────────────────────────────────────────────────
# 3. RESULTS AGGREGATION
# ─────────────────────────────────────────────────────────────────────────────
def summarise(trades: pd.DataFrame, label: str = '') -> dict:
    if trades.empty:
        return {'window': label, 'n': 0}

    n          = len(trades)
    win_rate   = (trades['gross_usd'] > 0).mean() * 100
    avg_hold_s = trades['bars_held'].mean()
    avg_gross  = trades['gross_usd'].mean()
    avg_gross_t= trades['gross_pts'].mean()
    tot_gross  = trades['gross_usd'].sum()
    mdd        = max_drawdown(trades)
    pf         = profit_factor(trades)
    eod_pct    = trades['eod_close'].mean() * 100

    row = {
        'window'    : label,
        'n_trades'  : n,
        'win_rate'  : win_rate,
        'avg_hold_s': avg_hold_s,
        'avg_gross_t': avg_gross_t,
        'avg_gross_$': avg_gross,
        'tot_gross_$': tot_gross,
        'max_dd_$'  : mdd,
        'pfactor'   : pf,
        'eod_pct'   : eod_pct,
    }
    for lbl in SLIP_LABELS:
        col = f'net_{lbl}'
        row[f'avg_net_{lbl}'] = trades[col].mean()
        row[f'tot_net_{lbl}'] = trades[col].sum()
        row[f'wr_net_{lbl}']  = (trades[col] > 0).mean() * 100

    return row


def split_fomc(trades: pd.DataFrame):
    """Return (pre_fomc_trades, post_fomc_trades)."""
    if trades.empty:
        return trades, trades
    return (trades[~trades['post_fomc']].copy(),
            trades[ trades['post_fomc']].copy())


# ─────────────────────────────────────────────────────────────────────────────
# 4. PRINT REPORTS
# ─────────────────────────────────────────────────────────────────────────────
SEP  = '─' * 110
SEP2 = '═' * 110

def print_header():
    print(SEP2)
    print('  ESU4/ESZ4 — Z-Score Calendar Spread Mean-Reversion Backtest')
    print(f'  Roll window: Sep 12–19 2024   |   RTH only (08:30–15:15 ET)')
    print(f'  Entry: |z| > {ENTRY_Z}σ (edge-triggered)   '
          f'Exit: |z| < {EXIT_Z}σ or EOD force-close   Min hold: {MIN_HOLD}s')
    print()
    print(f'  Contract       : ESU4 (front) / ESZ4 (back)')
    print(f'  Tick           : {TICK_SIZE} pts = ${TICK_VALUE:.2f}/contract')
    print(f'  Multiplier     : ${MULTIPLIER:.0f}/pt')
    print()
    print(f'  Base commission: ${TC_BASE:.2f}/trade (2 contracts × $2.00 RT)')
    for k, v in SLIPPAGE.items():
        print(f'  Slippage {k:<18}: ${v:>5.2f} RT   →  Total cost: ${v+TC_BASE:.2f}/trade')
    print(SEP2)


def print_summary_table(rows: list[dict]):
    print()
    print('  WINDOW COMPARISON — ALL DAYS (pre + post FOMC)')
    print(SEP)
    hdr = (f"  {'Window':<8}  {'Trades':>6}  {'WinRate':>7}  {'AvgHold':>7}  "
           f"{'Gross/Tr':>8}  {'TotGross':>9}  {'MaxDD':>8}  {'PFactor':>7}  "
           f"{'EOD%':>5}  "
           + '  '.join(f"Net({lbl.split()[0]})".rjust(11) for lbl in SLIP_LABELS))
    print(hdr)
    print(SEP)
    for r in rows:
        if r.get('n_trades', 0) == 0:
            print(f"  {r['window']:<8}  {'—':>6}")
            continue
        net_cols = '  '.join(
            f"${r.get(f'tot_net_{lbl}', 0):>9,.0f}" for lbl in SLIP_LABELS
        )
        print(
            f"  {r['window']:<8}  {r['n_trades']:>6}  {r['win_rate']:>6.1f}%  "
            f"{r['avg_hold_s']:>6.0f}s  "
            f"${r['avg_gross_$']:>7.2f}  ${r['tot_gross_$']:>8,.0f}  "
            f"${r['max_dd_$']:>7,.0f}  {r['pfactor']:>7.2f}  "
            f"{r['eod_pct']:>4.0f}%  "
            + net_cols
        )
    print(SEP)


def print_slippage_detail(rows: list[dict]):
    print()
    print('  AVERAGE NET P&L PER TRADE  (each window × each slippage scenario)')
    print(SEP)
    hdr = f"  {'Window':<8}  {'Trades':>6}  {'Gross/Tr':>9}" + \
          ''.join(f"  {'Net('+lbl.split()[0]+')':>14}" for lbl in SLIP_LABELS)
    print(hdr)
    print(SEP)
    for r in rows:
        if r.get('n_trades', 0) == 0:
            continue
        net_cols = ''.join(
            f"  ${r.get(f'avg_net_{lbl}', 0):>12.2f}" for lbl in SLIP_LABELS
        )
        be = r['avg_gross_$']
        print(
            f"  {r['window']:<8}  {r['n_trades']:>6}  ${r['avg_gross_$']:>8.2f}"
            + net_cols
        )
    print(SEP)
    print('  Breakeven gross P&L/trade:')
    for k, v in TOTAL_COSTS.items():
        print(f'    {k:<20} ${v:.2f}')


def print_fomc_split(pre_rows: list[dict], post_rows: list[dict]):
    print()
    print('  PRE-FOMC vs POST-FOMC SPLIT  (Sep 18 14:00 ET divider)')
    print(SEP)
    hdr = (f"  {'Window':<8}  {'Period':<12}  {'Trades':>6}  "
           f"{'WinRate':>7}  {'Gross/Tr':>9}  {'TotGross':>10}  "
           + '  '.join(f"Net({lbl.split()[0]})".rjust(9) for lbl in SLIP_LABELS))
    print(hdr)
    print(SEP)
    for pr, po in zip(pre_rows, post_rows):
        w = pr['window']
        for period, r in [('Pre-FOMC', pr), ('Post-FOMC', po)]:
            if r.get('n_trades', 0) == 0:
                print(f"  {w:<8}  {period:<12}  {'—':>6}")
                w = ''
                continue
            net_cols = '  '.join(
                f"${r.get(f'tot_net_{lbl}', 0):>8,.0f}" for lbl in SLIP_LABELS
            )
            print(
                f"  {w:<8}  {period:<12}  {r['n_trades']:>6}  "
                f"{r['win_rate']:>6.1f}%  ${r['avg_gross_$']:>8.2f}  "
                f"${r['tot_gross_$']:>9,.0f}  " + net_cols
            )
            w = ''
    print(SEP)


def print_daily_breakdown(all_trades: dict[str, pd.DataFrame], best_window: str):
    trades = all_trades[best_window]
    if trades.empty:
        return
    print()
    print(f'  DAILY BREAKDOWN  (window: {best_window})')
    print(SEP)
    print(f"  {'Date':<12}  {'Day':<5}  {'Trades':>6}  "
          f"{'WinRate':>7}  {'Gross/Tr':>9}  {'TotGross':>10}  "
          f"{'NetTight':>10}  {'NetMid':>8}  {'NetWide':>8}")
    print(SEP)

    DAY_NAMES = {
        '2024-09-12': 'Thu1',
        '2024-09-13': 'Fri ',
        '2024-09-15': 'Sun ',
        '2024-09-16': 'Mon ',
        '2024-09-17': 'Tue ',
        '2024-09-18': 'Wed*',   # FOMC day
        '2024-09-19': 'Thu2',
    }

    for date_str, day_label in DAY_NAMES.items():
        mask = trades['entry_time'].dt.date == pd.Timestamp(date_str).date()
        day_t = trades[mask]
        if day_t.empty:
            print(f'  {date_str}  {day_label}  {"—":>6}')
            continue
        n      = len(day_t)
        wr     = (day_t['gross_usd'] > 0).mean() * 100
        ag     = day_t['gross_usd'].mean()
        tg     = day_t['gross_usd'].sum()
        n0 = SLIP_LABELS[0]; n1 = SLIP_LABELS[1]; n2 = SLIP_LABELS[2]
        nt = day_t[f'net_{n0}'].sum()
        nm = day_t[f'net_{n1}'].sum()
        nw = day_t[f'net_{n2}'].sum()
        print(f'  {date_str}  {day_label}  {n:>6}  {wr:>6.1f}%  '
              f'${ag:>8.2f}  ${tg:>9,.0f}  ${nt:>9,.0f}  '
              f'${nm:>7,.0f}  ${nw:>7,.0f}')
    print(SEP)
    print('  * = FOMC day (Sep 18, 50bp cut at 14:00 ET)')


# ─────────────────────────────────────────────────────────────────────────────
# 5. CHARTS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_COLORS = {
    '30s' : '#e74c3c',
    '1min': '#e67e22',
    '2min': '#f1c40f',
    '5min': '#27ae60',
    '10min':'#2980b9',
}

def plot_equity_curves(all_trades: dict[str, pd.DataFrame],
                       spread: pd.Series) -> str:
    fig, axes = plt.subplots(len(Z_WINDOWS), 1,
                             figsize=(14, 3.0 * len(Z_WINDOWS)),
                             sharex=False)
    fig.suptitle('Cumulative Gross P&L — Equity Curves by Z-Score Window\n'
                 'ESU4/ESZ4 Sep 2024 Roll Window, RTH Only',
                 fontsize=11, fontweight='bold')

    for ax, win in zip(axes, Z_WINDOWS):
        trades = all_trades[win]
        col    = WINDOW_COLORS[win]
        if trades.empty:
            ax.text(0.5, 0.5, 'No trades', transform=ax.transAxes, ha='center')
            ax.set_title(win)
            continue

        eq   = trades['cum_gross']
        peak = trades['peak']
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=(eq.values >= 0), alpha=0.3, color='forestgreen')
        ax.fill_between(range(len(eq)), eq.values, 0,
                        where=(eq.values <  0), alpha=0.3, color='crimson')
        ax.plot(range(len(eq)), eq.values, color=col, lw=1.5, label='Cum gross')
        ax.plot(range(len(eq)), peak.values, color='gray', lw=0.8,
                linestyle='--', alpha=0.7, label='Running peak')

        # Mark FOMC boundary
        fomc_trade_idx = trades[trades['post_fomc']].index
        if len(fomc_trade_idx) > 0:
            first_post = fomc_trade_idx[0]
            pos = trades.index.get_loc(first_post)
            ax.axvline(pos, color='crimson', lw=1.2, linestyle=':', alpha=0.7,
                       label='FOMC')

        n = len(trades)
        tot = trades['gross_usd'].sum()
        mdd = max_drawdown(trades)
        ax.set_title(f'{win} window  |  {n} trades  |  '
                     f'Total gross: ${tot:,.0f}  |  Max DD: ${mdd:,.0f}',
                     fontsize=9)
        ax.set_ylabel('Cum P&L ($)')
        ax.axhline(0, color='black', lw=0.5)
        ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    out = str(OUT_DIR / 'equity_curves.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_pnl_distributions(all_trades: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(1, len(Z_WINDOWS),
                             figsize=(4 * len(Z_WINDOWS), 4), sharey=False)
    fig.suptitle('Trade-Level Gross P&L Distributions by Window',
                 fontsize=11, fontweight='bold')

    for ax, win in zip(axes, Z_WINDOWS):
        trades = all_trades[win]
        col    = WINDOW_COLORS[win]
        if trades.empty:
            ax.set_title(win)
            continue

        vals = trades['gross_usd'].values
        ax.hist(vals, bins=40, color=col, alpha=0.75, edgecolor='white')
        ax.axvline(0,              color='black',       lw=1.0)
        ax.axvline(vals.mean(),    color='navy',        lw=1.2, linestyle='--',
                   label=f'Mean ${vals.mean():.0f}')
        # Mark breakeven thresholds for each slippage scenario
        for slip_lbl, sv in zip(['Tight', 'Mid', 'Wide'],
                                  [TOTAL_COSTS[k] for k in SLIP_LABELS]):
            ax.axvline(sv, color='gray', lw=0.8, linestyle=':', alpha=0.6)
        ax.set_title(f'{win}\nn={len(trades)}', fontsize=9)
        ax.set_xlabel('Gross P&L ($)')
        ax.legend(fontsize=7)

    fig.tight_layout()
    out = str(OUT_DIR / 'pnl_distributions.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_hold_time(all_trades: dict[str, pd.DataFrame]) -> str:
    fig, axes = plt.subplots(1, len(Z_WINDOWS),
                             figsize=(4 * len(Z_WINDOWS), 4))
    fig.suptitle('Trade Hold-Time Distribution (seconds)', fontsize=11,
                 fontweight='bold')

    for ax, win in zip(axes, Z_WINDOWS):
        trades = all_trades[win]
        col    = WINDOW_COLORS[win]
        if trades.empty:
            ax.set_title(win)
            continue

        bars = trades['bars_held'].values
        ax.hist(bars, bins=min(50, max(bars)), color=col, alpha=0.75,
                edgecolor='white')
        ax.axvline(bars.mean(), color='black', lw=1.2, linestyle='--',
                   label=f'Mean {bars.mean():.0f}s')
        eod = trades['eod_close'].mean() * 100
        ax.set_title(f'{win}  |  EOD-closed: {eod:.0f}%', fontsize=9)
        ax.set_xlabel('Hold time (bars = seconds)')
        ax.legend(fontsize=7)

    fig.tight_layout()
    out = str(OUT_DIR / 'hold_times.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_net_by_slippage(rows: list[dict]) -> str:
    labels = [r['window'] for r in rows if r.get('n_trades', 0) > 0]
    gross  = [r['tot_gross_$']                    for r in rows if r.get('n_trades', 0) > 0]
    nets   = {lbl: [r.get(f'tot_net_{lbl}', 0)   for r in rows if r.get('n_trades', 0) > 0]
              for lbl in SLIP_LABELS}

    x = np.arange(len(labels))
    w = 0.18
    colors_slip = ['#2ecc71', '#f39c12', '#e74c3c']

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * w, gross, width=w, color='#3498db', alpha=0.85,
           label='Gross', edgecolor='white')
    for i, (lbl, color) in enumerate(zip(SLIP_LABELS, colors_slip)):
        offset = (i - 0.5) * w
        ax.bar(x + offset, nets[lbl], width=w, color=color, alpha=0.85,
               label=f'Net {lbl.split()[0]}', edgecolor='white')

    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Total P&L ($)')
    ax.set_title('Total Gross vs Net P&L by Window × Slippage Scenario\n'
                 'ESU4/ESZ4 Sep 2024 Roll Window, RTH',
                 fontweight='bold')
    ax.legend(fontsize=9)
    for i, (g, row) in enumerate(zip(gross, [r for r in rows if r.get('n_trades', 0) > 0])):
        ax.text(i - 1.5 * w, g + 50, f'${g:,.0f}', ha='center', fontsize=7)

    fig.tight_layout()
    out = str(OUT_DIR / 'net_by_slippage.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_daily_heatmap(all_trades: dict[str, pd.DataFrame]) -> str:
    """Net P&L per day per window, displayed as a heatmap."""
    DAY_KEYS = ['2024-09-12', '2024-09-13', '2024-09-15',
                '2024-09-16', '2024-09-17', '2024-09-18', '2024-09-19']
    DAY_LBL  = ['Thu1', 'Fri', 'Sun', 'Mon', 'Tue', 'Wed*', 'Thu2']
    net_lbl  = SLIP_LABELS[1]   # use mid-slippage for heatmap

    matrix = np.zeros((len(Z_WINDOWS), len(DAY_KEYS)))
    for ri, win in enumerate(Z_WINDOWS):
        trades = all_trades[win]
        if trades.empty:
            continue
        for ci, dk in enumerate(DAY_KEYS):
            mask = trades['entry_time'].dt.date == pd.Timestamp(dk).date()
            matrix[ri, ci] = trades.loc[mask, f'net_{net_lbl}'].sum()

    fig, ax = plt.subplots(figsize=(11, 4))
    vmax = max(abs(matrix).max(), 1)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto',
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(DAY_LBL)))
    ax.set_xticklabels(DAY_LBL)
    ax.set_yticks(range(len(Z_WINDOWS)))
    ax.set_yticklabels(Z_WINDOWS)
    ax.set_title(f'Daily Net P&L Heatmap — Mid Slippage ({net_lbl})\n'
                 '* = FOMC day', fontweight='bold')
    plt.colorbar(im, ax=ax, label='Net P&L ($)')

    for ri in range(len(Z_WINDOWS)):
        for ci in range(len(DAY_KEYS)):
            val = matrix[ri, ci]
            ax.text(ci, ri, f'${val:,.0f}', ha='center', va='center',
                    fontsize=7, color='black' if abs(val) < vmax * 0.6 else 'white')

    fig.tight_layout()
    out = str(OUT_DIR / 'daily_heatmap.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


def plot_direction_breakdown(all_trades: dict[str, pd.DataFrame]) -> str:
    """Longs vs shorts: trade count and avg P&L per window."""
    fig, axes = plt.subplots(2, len(Z_WINDOWS),
                             figsize=(3.5 * len(Z_WINDOWS), 7))
    fig.suptitle('Long vs Short: Trade Count and Average Gross P&L',
                 fontsize=11, fontweight='bold')

    for col_i, win in enumerate(Z_WINDOWS):
        trades = all_trades[win]
        color  = WINDOW_COLORS[win]
        longs  = trades[trades['direction'] ==  1] if not trades.empty else pd.DataFrame()
        shorts = trades[trades['direction'] == -1] if not trades.empty else pd.DataFrame()

        # Row 0: trade counts
        ax = axes[0, col_i]
        ax.bar(['Long', 'Short'],
               [len(longs), len(shorts)],
               color=['steelblue', 'tomato'], alpha=0.8, edgecolor='white')
        ax.set_title(f'{win}', fontsize=9)
        ax.set_ylabel('# Trades' if col_i == 0 else '')

        # Row 1: avg gross P&L
        ax = axes[1, col_i]
        ag_l = longs['gross_usd'].mean() if not longs.empty else 0
        ag_s = shorts['gross_usd'].mean() if not shorts.empty else 0
        ax.bar(['Long', 'Short'], [ag_l, ag_s],
               color=['steelblue', 'tomato'], alpha=0.8, edgecolor='white')
        ax.axhline(0, color='black', lw=0.5)
        ax.set_ylabel('Avg Gross P&L ($)' if col_i == 0 else '')

        for bar_ax in [axes[0, col_i], axes[1, col_i]]:
            for patch in bar_ax.patches:
                h = patch.get_height()
                bar_ax.text(patch.get_x() + patch.get_width() / 2,
                            h + abs(h) * 0.02,
                            f'{h:.0f}', ha='center', fontsize=7)

    fig.tight_layout()
    out = str(OUT_DIR / 'direction_breakdown.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_header()

    # Load data
    print('\n[1/3] Loading RTH spread data...')
    spread, ba_sum = load_rth_spread()
    print(f'      {len(spread):,} 1-second RTH bars loaded')

    print('[2/3] Building fair-value model (daily SOFR + FOMC intraday override)...')
    fv, sofr_pre, sofr_post, dt_yr = build_fair_value(spread)
    dev = spread - fv
    print(f'      SOFR pre-FOMC: {sofr_pre*100:.4f}%   post-FOMC: {sofr_post*100:.4f}%')
    print(f'      ΔT: {dt_yr:.4f} yr   |   Dev mean: {dev.mean():.3f} pts   '
          f'std: {dev.std():.3f} pts')

    # Run simulations
    print('[3/3] Running simulations...')
    all_trades = {}
    for win in Z_WINDOWS:
        trades = simulate(spread, fv, win)
        all_trades[win] = trades
        n_pre  = len(trades[~trades['post_fomc']]) if not trades.empty else 0
        n_post = len(trades[ trades['post_fomc']]) if not trades.empty else 0
        print(f'      {win:<6}  {len(trades):>4} total trades  '
              f'({n_pre} pre-FOMC + {n_post} post-FOMC)')

    # Aggregate
    rows      = [summarise(all_trades[w], w) for w in Z_WINDOWS]
    pre_rows  = [summarise(*split_fomc(all_trades[w])[:1], w) for w in Z_WINDOWS]
    post_rows = [summarise(*split_fomc(all_trades[w])[1:],  w) for w in Z_WINDOWS]

    # Print reports
    print_summary_table(rows)
    print_slippage_detail(rows)
    print_fomc_split(pre_rows, post_rows)

    # Find best window (highest total gross P&L)
    best_win = max(Z_WINDOWS,
                   key=lambda w: all_trades[w]['gross_usd'].sum()
                   if not all_trades[w].empty else -1e9)
    print_daily_breakdown(all_trades, best_win)

    # Charts
    print(f'\nGenerating charts → {OUT_DIR}/')
    paths = [
        plot_equity_curves(all_trades, spread),
        plot_pnl_distributions(all_trades),
        plot_hold_time(all_trades),
        plot_net_by_slippage(rows),
        plot_daily_heatmap(all_trades),
        plot_direction_breakdown(all_trades),
    ]
    for p in paths:
        print(f'  Saved: {Path(p).name}')

    # Final summary
    print()
    print(SEP2)
    print('  BOTTOM LINE')
    print(SEP)
    best_row = next(r for r in rows if r['window'] == best_win)
    print(f'  Best window (gross):  {best_win}')
    print(f'  Total gross P&L    :  ${best_row["tot_gross_$"]:,.0f}')
    for lbl in SLIP_LABELS:
        tot = best_row.get(f'tot_net_{lbl}', 0)
        avg = best_row.get(f'avg_net_{lbl}', 0)
        bvl = (trades['gross_usd'] > TOTAL_COSTS[lbl]).mean() * 100 \
              if not (trades := all_trades[best_win]).empty else 0
        sign = '+' if tot >= 0 else ''
        print(f'  Net ({lbl:<20}):  {sign}${tot:,.0f}  '
              f'(avg/trade {sign}${avg:.2f}  '
              f'| {bvl:.0f}% of trades cover costs)')
    print()
    print('  Break-even gross P&L per trade to cover ALL costs:')
    for lbl in SLIP_LABELS:
        be = TOTAL_COSTS[lbl]
        be_pts = be / MULTIPLIER
        print(f'    {lbl:<22}: ${be:.2f}  =  {be_pts:.4f} pts  =  '
              f'{be_pts/TICK_SIZE:.1f} ticks')
    print(SEP2)


if __name__ == '__main__':
    main()
