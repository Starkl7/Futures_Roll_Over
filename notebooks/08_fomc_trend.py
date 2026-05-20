#!/usr/bin/env python3
"""
08_fomc_trend.py — FOMC-Day Trend Strategy + Sep 19 Mean-Reversion Diagnostic

Two questions:
  Q1: Can EMA momentum trend-following on Sep 18 post-announcement (18:30–19:15 UTC)
      replace the blackout dead zone and generate positive P&L?
  Q2: How does mean-reversion perform on Sep 19 (Thu2) in isolation, after the
      cut is fully absorbed and the spread has re-anchored to the new FV?

Sep 18 regime breakdown:
  12:30–17:00 UTC  mean-reversion (pre-blackout) — same params as 07_tearsheet.py
  17:00–18:30 UTC  blackout: no new entries; FOMC force-close at 18:00 UTC
  18:30–19:15 UTC  trend-following: EMA(1-min) × EMA(5-min) crossover on raw spread

Counterfactual shown for comparison:
  Sep 18 post-FOMC mean-rev (naive z-score, no blackout) — confirms the
  blackout was correctly motivated.

Data  : Databento GLBX.MDP3 | MBP-10 1s | /Volumes/SEAGATE/Databento_Futures
Output: notebooks/figures/08_sep18_overview.png
        notebooks/figures/08_sep19_analysis.png
"""

import glob
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.float_format', '{:.4f}'.format)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
OUT_DIR  = Path(__file__).parent / 'figures'
OUT_DIR.mkdir(exist_ok=True)

# ── Contract / roll window ────────────────────────────────────────────────────
FRONT, BACK = 'ESU4', 'ESZ4'
ROLL_START  = '2024-09-12'
RTH_OPEN, RTH_CLOSE = '12:30', '19:15'
FRI_SKIP_MIN = 30

# ── FOMC event ────────────────────────────────────────────────────────────────
FOMC_UTC       = pd.Timestamp('2024-09-18 18:00:00', tz='UTC')
FOMC_PRE_MIN   = 60    # blackout opens 60 min before announcement
FOMC_POST_MIN  = 30    # blackout closes 30 min after announcement
FOMC_BLK_START = FOMC_UTC - pd.Timedelta(minutes=FOMC_PRE_MIN)   # 17:00 UTC
FOMC_RESUME    = FOMC_UTC + pd.Timedelta(minutes=FOMC_POST_MIN)  # 18:30 UTC

# ── Volume regime gate ────────────────────────────────────────────────────────
VOL_GATE_LOW  = 0.05   # back-share below → roll not started; skip
VOL_GATE_HIGH = 0.80   # back-share above → roll done, spread going quiet; skip

SEP18 = pd.Timestamp('2024-09-18').date()
SEP19 = pd.Timestamp('2024-09-19').date()

# ── Contract specs ────────────────────────────────────────────────────────────
MULT  = 50.0
TICK  = 0.25

# ── Mean-rev params (identical to 07_tearsheet.py) ────────────────────────────
MR_WINDOW    = '10min'
MR_THRESHOLD = 2.5
MR_TP        = 0.75   # index pts
MR_SL        = 0.50   # index pts

# ── Trend params (EMA momentum, post-FOMC window only) ────────────────────────
EMA_FAST_BARS = 60    # 1-min EMA
EMA_SLOW_BARS = 300   # 5-min EMA
TR_TP         = 0.75  # index pts
TR_SL         = 0.50  # index pts

# ── FV model ──────────────────────────────────────────────────────────────────
DIV_YIELD = 0.013
FOMC_CUT  = 0.005     # 50bp

# ── Transaction costs ─────────────────────────────────────────────────────────
TC_INST = 4.00
SLIP    = {'Tight': 6.25, 'Mid': 12.50, 'Wide': 25.00}
ALLIN   = {k: v + TC_INST for k, v in SLIP.items()}

# ── Colours ───────────────────────────────────────────────────────────────────
NAVY  = '#0d1b2a'; STEEL = '#2471a3'; GREEN = '#1e8449'; RED   = '#c0392b'
GOLD  = '#f39c12'; ORANGE = '#e67e22'; MGRAY = '#d0d3d4'; DGRAY = '#555555'

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.5,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.color': MGRAY, 'grid.alpha': 0.45,
    'grid.linewidth': 0.5,
})

SEP  = '─' * 110
SEP2 = '═' * 110


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def _load_sofr() -> pd.Series:
    raw = pd.read_csv(DATA_DIR / 'SOFR.csv', parse_dates=['observation_date'],
                      index_col='observation_date')
    s = raw.iloc[:, 0].dropna() / 100.0
    return pd.Series(s.values, index=pd.DatetimeIndex(s.index).tz_localize('UTC'))


def _load_volume_gate() -> dict:
    """
    Returns {calendar_date: gate_open} from daily ohlcv1d volume back-share.
    Gate open when VOL_GATE_LOW < back_share < VOL_GATE_HIGH.
    """
    files = sorted(glob.glob(str(DATA_DIR / f'ohlcv1d_{FRONT}_{BACK}_*.parquet')))
    if not files:
        return {}
    vol = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    piv = vol.pivot_table(index=vol.index.date, columns='symbol',
                          values='volume', aggfunc='sum')
    front_v = piv[FRONT] if FRONT in piv.columns else pd.Series(0, index=piv.index)
    back_v  = piv[BACK]  if BACK  in piv.columns else pd.Series(0, index=piv.index)
    back_share = back_v / (front_v + back_v + 1e-9)
    return {d: bool(VOL_GATE_LOW < float(s) < VOL_GATE_HIGH)
            for d, s in back_share.items()}


def _build_fv(spread: pd.Series, front_mid: pd.Series,
              sofr_utc: pd.Series, dt_yr: float) -> pd.Series:
    daily_idx  = pd.date_range(spread.index[0].normalize(),
                               spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()
    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    ).ffill()
    pre_mask = r_f.index < FOMC_UTC
    if pre_mask.any():
        pre_rate = float(r_f[pre_mask].iloc[-1])
        r_f[~pre_mask] = pre_rate - FOMC_CUT
    return front_mid.reindex(r_f.index).ffill() * (r_f - DIV_YIELD) * dt_yr


def load_data():
    sofr = _load_sofr()

    defn  = pd.read_parquet(DATA_DIR / f'definitions_{FRONT}_{BACK}.parquet')
    exp_f = defn.loc[defn['symbol'] == FRONT, 'expiration'].iloc[0]
    exp_b = defn.loc[defn['symbol'] == BACK,  'expiration'].iloc[0]
    dt_yr = (exp_b - exp_f).total_seconds() / (365.25 * 86400)

    files = sorted(glob.glob(str(DATA_DIR / f'mbp10_{FRONT}_{BACK}_{ROLL_START}_*.parquet')))
    print(f'  Loading {len(files)} day-files', end='', flush=True)
    parts = []
    for f in files:
        df   = pd.read_parquet(f, columns=['bid_px_00', 'ask_px_00', 'symbol'])
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
    fv        = _build_fv(spread, front_mid, sofr, dt_yr)
    dev       = (spread - fv).dropna()
    spread    = spread.reindex(dev.index)
    fv        = fv.reindex(dev.index)

    print(f'  {len(spread):,} RTH 1s bars  |  ΔT={dt_yr:.4f}yr  |  '
          f'dev mean={dev.mean():.3f}  std={dev.std():.3f}')
    return spread, fv, dev, dt_yr


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION — MEAN-REVERSION  (same engine as 07_tearsheet.py)
# ─────────────────────────────────────────────────────────────────────────────
def _compute_z(dev: pd.Series) -> np.ndarray:
    mu  = dev.rolling(MR_WINDOW, min_periods=1).mean()
    sig = dev.rolling(MR_WINDOW, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


def _make_entry_mask(spread: pd.Series, vol_gate: dict,
                     fomc_blackout: bool = True) -> np.ndarray:
    idx  = spread.index
    mask = np.ones(len(idx), dtype=bool)
    # Volume regime gate
    if vol_gate:
        dates = np.array([t.date() for t in idx])
        for i, d in enumerate(dates):
            if not vol_gate.get(d, True):
                mask[i] = False
    # Friday open filter
    fri_cut = (pd.Timestamp('2000-01-01 12:30') +
               pd.Timedelta(minutes=FRI_SKIP_MIN)).time()
    is_fri  = (idx.weekday == 4)
    in_fri  = (idx.time >= pd.Timestamp('2000-01-01 12:30').time()) & (idx.time < fri_cut)
    mask[is_fri & in_fri] = False
    if fomc_blackout:
        mask[(idx >= FOMC_BLK_START) & (idx < FOMC_RESUME)] = False
    return mask


def simulate_mr(spread: pd.Series, dev: pd.Series, z_arr: np.ndarray,
                entry_mask: np.ndarray, label: str = 'mean_rev') -> pd.DataFrame:
    """
    Mean-reversion simulation: TP / SL / FOMC force-close / EOD exits.
    entry_mask controls which bars are eligible for new entry signals.
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

    fomc_bar = np.zeros(n, dtype=bool)
    fi = int(np.searchsorted(times, FOMC_UTC))
    if fi < n:
        fomc_bar[fi] = True

    trades = []
    pos = 0; ei = -1; epx = np.nan; ez = np.nan
    mnpx = np.nan; mxpx = np.nan; held = 0; pend = 0

    for i in range(n):
        zi = z_arr[i]; px = prices[i]

        if pend != 0 and pos == 0:
            pos, ei, epx, ez = pend, i, px, z_arr[i]
            mnpx = mxpx = px; held = 0; pend = 0

        if pos != 0:
            held += 1
            mnpx = min(mnpx, px); mxpx = max(mxpx, px)
            mv = pos * (px - epx); xt = None
            if fomc_bar[i]:    xt = 'FOMC'
            elif is_last[i]:   xt = 'EOD'
            elif mv >= MR_TP:  xt = 'TP'
            elif mv <= -MR_SL: xt = 'SL'
            if xt:
                gp  = pos * (px - epx)
                mae = max(0., epx - mnpx) if pos == 1 else max(0., mxpx - epx)
                mfe = max(0., mxpx - epx) if pos == 1 else max(0., epx - mnpx)
                trades.append({
                    'entry_time': times[ei], 'exit_time': times[i],
                    'direction': pos, 'entry_z': float(ez),
                    'entry_spread': epx, 'exit_spread': px,
                    'gross_pts': gp, 'gross_usd': gp * MULT,
                    'bars_held': held, 'hold_min': held / 60.,
                    'exit_type': xt, 'mae_pts': mae, 'mfe_pts': mfe,
                    'strategy': label, 'trade_date': times[ei].date(),
                })
                pos = ei = held = 0
                epx = ez = mnpx = mxpx = np.nan; pend = 0

        if pos == 0 and pend == 0 and i > 0 and entry_mask[i]:
            zp = z_arr[i - 1]
            if not np.isnan(zi) and not np.isnan(zp):
                if zp >= -MR_THRESHOLD and zi < -MR_THRESHOLD: pend =  1
                elif zp <= MR_THRESHOLD and zi > MR_THRESHOLD: pend = -1

    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    for lbl, cost in SLIP.items():
        df[f'net_{lbl}'] = df['gross_usd'] - cost - TC_INST
    df['cum_gross'] = df['gross_usd'].cumsum()
    df['peak']      = df['cum_gross'].cummax()
    df['drawdown']  = df['cum_gross'] - df['peak']
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION — TREND FOLLOWING  (Sep 18 post-FOMC only)
# ─────────────────────────────────────────────────────────────────────────────
def simulate_trend(spread: pd.Series) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    EMA(1-min) × EMA(5-min) momentum on the raw spread.
    Restricted to Sep 18 post-FOMC RTH: 18:30–19:15 UTC (≈45 min, 2700 bars).

    Entry: edge-triggered crossover — fill at next bar.
      fast > slow (crosses above) → long spread  (expecting continuation up)
      fast < slow (crosses below) → short spread (expecting continuation down)
    Exit: TP=0.75 pts | SL=0.50 pts | EOD force-close at 19:15.

    Returns: (trades_df, fast_ema_series, slow_ema_series)
    """
    seg = spread[(spread.index.date == SEP18) & (spread.index >= FOMC_RESUME)].copy()

    fast_ema = seg.ewm(span=EMA_FAST_BARS, adjust=False).mean()
    slow_ema = seg.ewm(span=EMA_SLOW_BARS, adjust=False).mean()

    if len(seg) < EMA_SLOW_BARS + 10:
        return pd.DataFrame(), fast_ema, slow_ema

    prices = seg.values; times = seg.index; n = len(prices)
    fv_arr = fast_ema.values; sv_arr = slow_ema.values
    is_last = np.zeros(n, dtype=bool); is_last[-1] = True

    trades = []; pos = 0; ei = -1; epx = np.nan; held = 0; pend = 0
    mnpx = np.nan; mxpx = np.nan

    for i in range(n):
        px = prices[i]

        if pend != 0 and pos == 0:
            pos, ei, epx = pend, i, px; mnpx = mxpx = px; held = 0; pend = 0

        if pos != 0:
            held += 1; mnpx = min(mnpx, px); mxpx = max(mxpx, px)
            mv = pos * (px - epx); xt = None
            if is_last[i]:    xt = 'EOD'
            elif mv >= TR_TP: xt = 'TP'
            elif mv <= -TR_SL: xt = 'SL'
            if xt:
                gp  = pos * (px - epx)
                mae = max(0., epx - mnpx) if pos == 1 else max(0., mxpx - epx)
                mfe = max(0., mxpx - epx) if pos == 1 else max(0., epx - mnpx)
                trades.append({
                    'entry_time': times[ei], 'exit_time': times[i],
                    'direction': pos, 'entry_z': np.nan,
                    'entry_spread': epx, 'exit_spread': px,
                    'gross_pts': gp, 'gross_usd': gp * MULT,
                    'bars_held': held, 'hold_min': held / 60.,
                    'exit_type': xt, 'mae_pts': mae, 'mfe_pts': mfe,
                    'strategy': 'trend_ema', 'trade_date': times[ei].date(),
                })
                pos = ei = held = 0; epx = mnpx = mxpx = np.nan; pend = 0

        if pos == 0 and pend == 0 and i > 0:
            if not np.any(np.isnan([fv_arr[i], sv_arr[i], fv_arr[i-1], sv_arr[i-1]])):
                if fv_arr[i-1] <= sv_arr[i-1] and fv_arr[i] > sv_arr[i]:
                    pend =  1   # fast crossed above slow → uptrend → long
                elif fv_arr[i-1] >= sv_arr[i-1] and fv_arr[i] < sv_arr[i]:
                    pend = -1   # fast crossed below slow → downtrend → short

    if not trades:
        return pd.DataFrame(), fast_ema, slow_ema
    df = pd.DataFrame(trades)
    for lbl, cost in SLIP.items():
        df[f'net_{lbl}'] = df['gross_usd'] - cost - TC_INST
    df['cum_gross'] = df['gross_usd'].cumsum()
    return df, fast_ema, slow_ema


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
def summarize(df: pd.DataFrame) -> dict:
    if df.empty or len(df) == 0:
        return {'n': 0}
    g = df['gross_usd']
    wins   = g[g > 0].sum()
    losses = abs(g[g <= 0].sum())
    s = {
        'n':        len(df),
        'wr':       (g > 0).mean() * 100,
        'pf':       wins / losses if losses > 0 else float('inf'),
        'avg_g':    g.mean(),
        'tot_g':    g.sum(),
        'avg_nt':   df['net_Tight'].mean(),
        'tot_nt':   df['net_Tight'].sum(),
        'avg_nm':   df['net_Mid'].mean(),
        'tot_nm':   df['net_Mid'].sum(),
        'tp_pct':   (df['exit_type'] == 'TP').mean() * 100,
        'sl_pct':   (df['exit_type'] == 'SL').mean() * 100,
        'avg_hold': df['hold_min'].mean(),
        'avg_mfe':  df['mfe_pts'].mean(),
        'avg_mae':  df['mae_pts'].mean(),
        'mdd':      df['drawdown'].min() if 'drawdown' in df.columns else 0.,
    }
    eod_n  = (df['exit_type'] == 'EOD').sum()
    fomc_n = (df['exit_type'] == 'FOMC').sum() if 'FOMC' in df['exit_type'].values else 0
    s['eod_pct']  = eod_n  / len(df) * 100
    s['fomc_pct'] = fomc_n / len(df) * 100
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PRINT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _row(label: str, s: dict):
    if s.get('n', 0) == 0:
        print(f'  {label:<38}  — no trades —')
        return
    exits = f"TP={s['tp_pct']:.0f}% SL={s['sl_pct']:.0f}% EOD={s['eod_pct']:.0f}%"
    if s.get('fomc_pct', 0) > 0:
        exits += f" FOMC={s['fomc_pct']:.0f}%"
    print(
        f"  {label:<38}  n={s['n']:>3}  WR={s['wr']:>5.1f}%  PF={s['pf']:>5.2f}  "
        f"AvgG=${s['avg_g']:>7.2f}  TotG=${s['tot_g']:>7,.0f}  "
        f"AvgNT=${s['avg_nt']:>7.2f}  TotNT=${s['tot_nt']:>7,.0f}  "
        f"Hold={s['avg_hold']:>4.0f}m  {exits}"
    )


def print_trade_list(df: pd.DataFrame, label: str):
    if df.empty:
        print(f'  {label}: no trades')
        return
    print(f'\n  ── {label} ──────────────────────────────────────────────────────')
    print(f"  {'#':<3}  {'Entry (UTC)':>19}  {'Exit (UTC)':>19}  "
          f"{'Dir':<5}  {'GrossUSD':>9}  {'Net(T)':>7}  {'Hold':>5}  {'Exit':<5}")
    print(f"  {'─'*90}")
    for idx, row in df.reset_index(drop=True).iterrows():
        d = 'Long' if row['direction'] == 1 else 'Short'
        print(f"  {idx+1:<3}  {str(row['entry_time'])[:19]:>19}  "
              f"{str(row['exit_time'])[:19]:>19}  {d:<5}  "
              f"${row['gross_usd']:>8.2f}  ${row['net_Tight']:>6.2f}  "
              f"{row['hold_min']:>4.0f}m  {row['exit_type']:<5}")
    print(f"  {'─'*90}")
    print(f"  {'TOTAL':<43}  {'':>19}  {'':5}  "
          f"${df['gross_usd'].sum():>8.2f}  ${df['net_Tight'].sum():>6.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────
def _mark_trades(ax, df: pd.DataFrame, y_series: pd.Series,
                 entry_color: str, exit_color: str, marker_size: int = 80):
    """Overlay entry/exit markers on a price axis."""
    if df.empty:
        return
    for _, row in df.iterrows():
        et = row['entry_time']; xt = row['exit_time']
        if et in y_series.index:
            ax.scatter(et, y_series[et], s=marker_size,
                       color=entry_color, zorder=5, marker='^' if row['direction'] == 1 else 'v')
        if xt in y_series.index:
            ax.scatter(xt, y_series[xt], s=marker_size * 0.7,
                       color=exit_color, zorder=5, marker='x')


def chart_sep18(spread, fv, mr_pre, trend_df, mr_cf, fast_ema, slow_ema):
    """
    Three-panel Sep 18 overview.
      Panel 1 — Full-day spread + FV, blackout zone, pre-FOMC MR trade markers
      Panel 2 — Post-FOMC zoom (18:30–19:15): EMA lines, trend + CF trade markers
      Panel 3 — P&L attribution bar chart: pre-MR vs trend vs CF-MR
    """
    s18 = spread[spread.index.date == SEP18]
    f18 = fv[fv.index.date == SEP18]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [3, 3, 2]})
    fig.suptitle('Sep 18 2024 — FOMC Day: Pre-FOMC Mean-Rev vs Post-FOMC Trend\n'
                 'ESU4/ESZ4 Calendar Spread (ESZ4 − ESU4)', fontsize=11, fontweight='bold')

    # ── Panel 1: Full day ─────────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(s18.index, s18.values, color=STEEL, lw=1.2, label='Spread (ESZ4−ESU4)')
    ax.plot(f18.index, f18.values, color=ORANGE, lw=1.2, ls='--', label='Fair Value', alpha=0.8)

    # Blackout region
    ax.axvspan(FOMC_BLK_START, FOMC_RESUME, color='gold', alpha=0.15,
               label=f'Blackout ({FOMC_BLK_START.strftime("%H:%M")}–{FOMC_RESUME.strftime("%H:%M")} UTC)')
    ax.axvline(FOMC_UTC, color='red', lw=1.5, ls=':', alpha=0.8, label='FOMC (18:00 UTC)')

    # FV step annotation
    fv_pre  = f18[f18.index < FOMC_UTC].iloc[-1]  if (f18.index < FOMC_UTC).any()  else np.nan
    fv_post = f18[f18.index >= FOMC_UTC].iloc[0]  if (f18.index >= FOMC_UTC).any() else np.nan
    if not np.isnan(fv_pre) and not np.isnan(fv_post):
        drop = fv_post - fv_pre
        ax.annotate(f'FV step: {drop:+.2f} pts\n(50bp cut × ΔT × spot)',
                    xy=(FOMC_UTC, fv_post), xytext=(FOMC_UTC + pd.Timedelta('8min'), fv_post - 0.6),
                    fontsize=7.5, color=ORANGE, ha='left',
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))

    _mark_trades(ax, mr_pre, s18, GREEN, GREEN)
    ax.set_ylabel('Spread (index pts)')
    ax.set_title('Full Sep 18 RTH — spread + FV + blackout zone', fontsize=9, color=NAVY)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # ── Panel 2: Post-FOMC zoom ───────────────────────────────────────────────
    ax = axes[1]
    post = s18[s18.index >= FOMC_RESUME]
    f_post = f18[f18.index >= FOMC_RESUME]

    ax.plot(post.index, post.values, color=STEEL, lw=1.5, label='Spread', zorder=3)
    ax.plot(f_post.index, f_post.values, color=ORANGE, lw=1.2, ls='--',
            label='Fair Value (post-cut)', alpha=0.8)

    if len(fast_ema) > 0:
        ax.plot(fast_ema.index, fast_ema.values, color=GREEN, lw=1.5,
                label=f'EMA fast ({EMA_FAST_BARS}s)', alpha=0.85)
        ax.plot(slow_ema.index, slow_ema.values, color=RED,   lw=1.5,
                label=f'EMA slow ({EMA_SLOW_BARS}s)', alpha=0.85)

    # Trend trades
    _mark_trades(ax, trend_df, post, GREEN, GREEN)
    for _, row in trend_df.iterrows() if not trend_df.empty else []:
        et = row['entry_time']
        if et in post.index:
            txt = 'L' if row['direction'] == 1 else 'S'
            ax.text(et, post[et] + 0.08, f"{txt} {row['exit_type']}",
                    fontsize=6.5, color=GREEN, ha='center')

    # Counterfactual mean-rev trades (what naive MR would have done here)
    _mark_trades(ax, mr_cf, post, RED, RED, marker_size=60)
    for _, row in mr_cf.iterrows() if not mr_cf.empty else []:
        et = row['entry_time']
        if et in post.index:
            txt = 'L' if row['direction'] == 1 else 'S'
            ax.text(et, post[et] - 0.14, f"{txt}(cf)",
                    fontsize=6.5, color=RED, ha='center')

    entry_patch  = mpatches.Patch(color=GREEN, label='Trend entries/exits')
    cf_patch     = mpatches.Patch(color=RED,   label='MR counterfactual (shows why blackout was right)')
    ax.legend(handles=[entry_patch, cf_patch] +
              ax.get_lines()[:4], fontsize=7, loc='upper right', framealpha=0.85)
    ax.set_ylabel('Spread (index pts)')
    ax.set_title(f'Post-FOMC zoom (18:30–19:15 UTC) — EMA trend vs counterfactual mean-rev',
                 fontsize=9, color=NAVY)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # ── Panel 3: P&L attribution bars ─────────────────────────────────────────
    ax = axes[2]
    groups   = ['Pre-FOMC\nMean-Rev', 'Post-FOMC\nTrend (EMA)', 'Post-FOMC\nMR Counterfactual']
    dfs      = [mr_pre, trend_df, mr_cf]
    colors   = [STEEL, GREEN, RED]
    x        = np.arange(len(groups))
    gross_v  = [df['gross_usd'].sum() if not df.empty else 0 for df in dfs]
    net_t_v  = [df['net_Tight'].sum() if not df.empty else 0 for df in dfs]
    n_trades = [len(df) for df in dfs]

    w = 0.30
    b1 = ax.bar(x - w/2, gross_v, w, color=colors, alpha=0.85, edgecolor='white',
                label='Gross')
    b2 = ax.bar(x + w/2, net_t_v, w, color=colors, alpha=0.40, edgecolor='white',
                label='Net (Tight inst.)')
    ax.axhline(0, color=DGRAY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(groups, fontsize=8.5)
    ax.set_ylabel('Total P&L  ($)')
    ax.set_title('Sep 18 P&L Attribution by Strategy Window', fontsize=9, color=NAVY)
    ax.legend(fontsize=8)
    for bar, gv, nt, n in zip(b1, gross_v, net_t_v, n_trades):
        ax.text(bar.get_x() + bar.get_width() / 2,
                gv + (8 if gv >= 0 else -14),
                f'n={n}\n${gv:,.0f}', ha='center', fontsize=7.5, color=NAVY)

    fig.tight_layout()
    out = str(OUT_DIR / '08_sep18_overview.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 08_sep18_overview.png')


def chart_sep19(spread, fv, dev, z_arr, mr_sep19):
    """
    Three-panel Sep 19 analysis.
      Panel 1 — Spread + FV + trade entry/exit markers
      Panel 2 — 10-min rolling z-score + threshold lines
      Panel 3 — Equity curve (cumulative gross)
    """
    s19  = spread[spread.index.date == SEP19]
    f19  = fv[fv.index.date == SEP19]
    d19  = dev[dev.index.date == SEP19]
    z19  = pd.Series(z_arr, index=spread.index)[spread.index.date == SEP19]

    fig, axes = plt.subplots(3, 1, figsize=(14, 9),
                             gridspec_kw={'height_ratios': [3, 2, 2]},
                             sharex=False)
    fig.suptitle('Sep 19 2024 (Thu2) — Mean-Reversion in Isolation\n'
                 'Post-FOMC spread at new fair value; does MR work again?',
                 fontsize=11, fontweight='bold')

    # ── Panel 1: Spread + FV ──────────────────────────────────────────────────
    ax = axes[0]
    ax.plot(s19.index, s19.values, color=STEEL, lw=1.2, label='Spread (ESZ4−ESU4)')
    ax.plot(f19.index, f19.values, color=ORANGE, lw=1.2, ls='--',
            label='Fair Value (post-cut SOFR)', alpha=0.8)

    _mark_trades(ax, mr_sep19, s19, GREEN, GREEN)
    if not mr_sep19.empty:
        for _, row in mr_sep19.iterrows():
            et = row['entry_time']
            if et in s19.index:
                d = '▲' if row['direction'] == 1 else '▼'
                ax.text(et, s19[et] + 0.1,
                        f"{d}{row['exit_type']}\n${row['gross_usd']:+.0f}",
                        fontsize=6, color=GREEN, ha='center')

    ax.set_ylabel('Spread (index pts)')
    ax.set_title('Sep 19 RTH — Spread vs Fair Value + MR Trades', fontsize=9, color=NAVY)
    ax.legend(fontsize=7.5, loc='upper right', framealpha=0.85)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # ── Panel 2: Z-score ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.plot(z19.index, z19.values, color=NAVY, lw=1.0, label='Z-score (10-min rolling)')
    ax.axhline( MR_THRESHOLD, color=RED,   lw=1.2, ls='--', label=f'+{MR_THRESHOLD}σ short entry')
    ax.axhline(-MR_THRESHOLD, color=GREEN, lw=1.2, ls='--', label=f'−{MR_THRESHOLD}σ long entry')
    ax.axhline(0, color=DGRAY, lw=0.6)
    ax.fill_between(z19.index, z19.values, MR_THRESHOLD,
                    where=(z19.values > MR_THRESHOLD),  alpha=0.2, color=RED)
    ax.fill_between(z19.index, z19.values, -MR_THRESHOLD,
                    where=(z19.values < -MR_THRESHOLD), alpha=0.2, color=GREEN)
    ax.set_ylabel('Z-score')
    ax.set_title('Rolling Z-Score of (Spread − FV)', fontsize=9, color=NAVY)
    ax.legend(fontsize=7.5, loc='upper right')
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # ── Panel 3: Equity curve ─────────────────────────────────────────────────
    ax = axes[2]
    if not mr_sep19.empty:
        eq  = mr_sep19['gross_usd'].cumsum()
        neq = mr_sep19['net_Tight'].cumsum()
        x   = range(len(eq))
        ax.fill_between(x, eq.values, 0, where=(eq.values >= 0), alpha=0.2, color=GREEN)
        ax.fill_between(x, eq.values, 0, where=(eq.values <  0), alpha=0.2, color=RED)
        ax.plot(x, eq.values,  color=STEEL, lw=2.0, label='Cumulative Gross')
        ax.plot(x, neq.values, color=GREEN, lw=1.5, ls='--',
                label=f'Cumulative Net (Tight, all-in ${ALLIN["Tight"]:.2f})')
        ax.axhline(0, color=DGRAY, lw=0.7)
        ax.set_xlabel('Trade number')
        ax.set_ylabel('Cumulative P&L  ($)')
        tot_g = mr_sep19['gross_usd'].sum()
        tot_n = mr_sep19['net_Tight'].sum()
        ax.set_title(f'Sep 19 Equity — Gross ${tot_g:,.0f}  /  Net (Tight) ${tot_n:,.0f}',
                     fontsize=9, color=NAVY)
        ax.legend(fontsize=7.5, loc='upper left')
    else:
        ax.text(0.5, 0.5, 'No trades on Sep 19', transform=ax.transAxes,
                ha='center', fontsize=11, color=DGRAY)

    fig.tight_layout()
    out = str(OUT_DIR / '08_sep19_analysis.png')
    fig.savefig(out, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: 08_sep19_analysis.png')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print(SEP2)
    print('  ESU4/ESZ4 — FOMC-Day Trend Strategy + Sep 19 Mean-Reversion Diagnostic')
    print(SEP2)

    # ── Load ──────────────────────────────────────────────────────────────────
    print('\n[1/5] Loading data...')
    spread, fv, dev, dt_yr = load_data()

    # ── Z-score (computed on full 7-day window, used for both MR sims) ────────
    print('[2/5] Computing z-scores...')
    z_arr = _compute_z(dev)
    print(f'  z mean={np.nanmean(z_arr):.4f}  std={np.nanstd(z_arr):.4f}')

    # ── Volume gate ───────────────────────────────────────────────────────────
    vol_gate = _load_volume_gate()
    print(f'  Volume gate ({VOL_GATE_LOW:.0%}/{VOL_GATE_HIGH:.0%}) active days: '
          f'{[str(d) for d, v in sorted(vol_gate.items()) if v]}')

    # ── Entry masks ───────────────────────────────────────────────────────────
    # Standard mask (with volume gate + FOMC blackout)
    mask_standard = _make_entry_mask(spread, vol_gate, fomc_blackout=True)

    # Counterfactual: allow entries ONLY in Sep 18 post-FOMC window (no gate,
    # since the gate would close Sep 18 — this shows what naive MR would do)
    mask_cf = np.zeros(len(spread), dtype=bool)
    post_fomc_sep18 = (spread.index.date == SEP18) & (spread.index >= FOMC_RESUME)
    mask_cf[post_fomc_sep18] = True

    # Sep 19 mask: allow entries only on Sep 19 (gate closed on Sep 19 anyway,
    # so this is a diagnostic — shows what MR would do if gate were overridden)
    mask_sep19 = np.zeros(len(spread), dtype=bool)
    sep19_bars = (spread.index.date == SEP19)
    mask_sep19[sep19_bars] = True

    # ── Run simulations ───────────────────────────────────────────────────────
    print('[3/5] Running simulations...')

    # Full mean-rev with blackout (extract Sep 18 pre-FOMC and Sep 19 trades)
    mr_all = simulate_mr(spread, dev, z_arr, mask_standard, label='mean_rev')
    if not mr_all.empty:
        mr_pre  = mr_all[mr_all['trade_date'] == SEP18].copy()  # Sep 18 pre-FOMC only
        mr_sep19 = mr_all[mr_all['trade_date'] == SEP19].copy()
    else:
        mr_pre = pd.DataFrame(); mr_sep19 = pd.DataFrame()

    # Post-FOMC counterfactual: naive MR in the Sep 18 post-FOMC window
    mr_cf_all = simulate_mr(spread, dev, z_arr, mask_cf, label='mr_counterfactual')
    mr_cf = mr_cf_all if not mr_cf_all.empty else pd.DataFrame()

    # Sep 19 isolated (using full-dataset z-score for proper rolling context)
    mr_sep19_iso = simulate_mr(spread, dev, z_arr, mask_sep19, label='mean_rev_sep19')

    # Trend following: Sep 18 post-FOMC
    trend_df, fast_ema, slow_ema = simulate_trend(spread)

    # Sep 19 from full run and isolated run should agree (same mask, same z)
    # Use isolated version for the Sep 19 analysis (mask_sep19 == sep19_bars)
    if mr_sep19_iso.empty and not mr_sep19.empty:
        mr_sep19_iso = mr_sep19

    trade_counts = {
        'Sep 18 pre-FOMC mean-rev': len(mr_pre),
        'Sep 18 post-FOMC trend (EMA)': len(trend_df),
        'Sep 18 post-FOMC MR counterfactual': len(mr_cf),
        'Sep 19 mean-rev': len(mr_sep19_iso),
    }
    for k, v in trade_counts.items():
        print(f'  {k}: {v} trades')

    # ── Statistics ────────────────────────────────────────────────────────────
    print('[4/5] Computing statistics...')
    s_pre   = summarize(mr_pre)
    s_trend = summarize(trend_df)
    s_cf    = summarize(mr_cf)
    s_sep19 = summarize(mr_sep19_iso)

    # Combined Sep 18 (pre-MR + trend)
    if not mr_pre.empty and not trend_df.empty:
        combined18 = pd.concat([mr_pre, trend_df], ignore_index=True)
        combined18['cum_gross'] = combined18['gross_usd'].cumsum()
        combined18['peak']      = combined18['cum_gross'].cummax()
        combined18['drawdown']  = combined18['cum_gross'] - combined18['peak']
    elif not mr_pre.empty:
        combined18 = mr_pre
    else:
        combined18 = pd.DataFrame()
    s_combined = summarize(combined18)

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    print(SEP2)
    print('  RESULTS — SEP 18 (FOMC DAY)')
    print(SEP)
    print(f'  {"Strategy":<38}  {"n":>3}  {"WR":>7}  {"PF":>5}  '
          f'{"AvgGross":>9}  {"TotGross":>9}  {"AvgNetT":>8}  {"TotNetT":>8}  '
          f'{"Hold":>5}  Exits')
    print(SEP)
    _row('Pre-FOMC mean-rev (12:30–17:00 UTC)', s_pre)
    _row(f'Post-FOMC trend EMA {EMA_FAST_BARS}s×{EMA_SLOW_BARS}s (18:30–19:15)', s_trend)
    _row('Post-FOMC MR counterfactual (18:30–19:15)', s_cf)
    print(SEP)
    _row('Sep 18 COMBINED  (pre-MR + trend)', s_combined)
    print()
    print(f'  Trend strategy: 1-min EMA crosses 5-min EMA  |  TP={TR_TP}pt  SL={TR_SL}pt')
    print(f'  MR counterfactual: z-score 10-min window, {MR_THRESHOLD}σ threshold — no blackout guard')
    print(f'  → Counterfactual shows what naive mean-rev WOULD have done post-FOMC')

    print()
    print(SEP2)
    print('  RESULTS — SEP 19 (THU2, POST-CUT REGIME)')
    print(SEP)
    print(f'  {"Strategy":<38}  {"n":>3}  {"WR":>7}  {"PF":>5}  '
          f'{"AvgGross":>9}  {"TotGross":>9}  {"AvgNetT":>8}  {"TotNetT":>8}  '
          f'{"Hold":>5}  Exits')
    print(SEP)
    _row('Sep 19 mean-rev (full RTH, no filters)', s_sep19)
    print()
    print(f'  Z-score window: {MR_WINDOW}  |  Threshold: {MR_THRESHOLD}σ  |  '
          f'TP={MR_TP}pt  SL={MR_SL}pt')
    print(f'  Z-score computed on full 7-day rolling series (realistic: no look-ahead)')

    # ── Trade lists ───────────────────────────────────────────────────────────
    print_trade_list(mr_pre,        'Sep 18 Pre-FOMC Mean-Rev Trades')
    print_trade_list(trend_df,      'Sep 18 Post-FOMC Trend Trades')
    print_trade_list(mr_cf,         'Sep 18 Post-FOMC Counterfactual MR Trades')
    print_trade_list(mr_sep19_iso,  'Sep 19 Mean-Rev Trades')

    # ── Cost sensitivity summary ───────────────────────────────────────────────
    print()
    print(SEP2)
    print('  COST SENSITIVITY — BREAKEVEN')
    print(SEP)
    print(f'  Scenario          All-in/trade  Break-even gross/trade')
    for k, v in ALLIN.items():
        be_pts = v / MULT
        print(f'  {k:<18}  ${v:>6.2f}        ${v:.2f}  ({be_pts:.4f} pts, '
              f'{be_pts/TICK:.1f} ticks)')
    print()
    if s_sep19.get('n', 0) > 0:
        print(f'  Sep 19 avg gross ${s_sep19["avg_g"]:.2f} vs breakeven:')
        for k, v in ALLIN.items():
            net_avg = s_sep19["avg_g"] - v
            sign = '+' if net_avg >= 0 else ''
            print(f'    {k:<16}: {sign}${net_avg:.2f}/trade  '
                  f'({"PROFITABLE ✓" if net_avg > 0 else "LOSS ✗"})')
    print(SEP2)

    # ── Charts ────────────────────────────────────────────────────────────────
    print('[5/5] Generating charts...')
    chart_sep18(spread, fv, mr_pre, trend_df, mr_cf, fast_ema, slow_ema)
    chart_sep19(spread, fv, dev, z_arr, mr_sep19_iso)
    print('  Done.')


if __name__ == '__main__':
    main()
