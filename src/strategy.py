#!/usr/bin/env python3
"""
strategy.py — shared computation layer for ES futures calendar spread backtest.

All functions accept explicit WindowConfig / StrategyParams parameters instead
of reading module-level globals, so the same logic runs across all roll windows.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION DATACLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WindowConfig:
    """All window-specific constants for one ES futures roll period."""
    front:      str
    back:       str
    roll_start: str           # 'YYYY-MM-DD'
    fomc_utc:   pd.Timestamp  # tz='UTC'; None if no FOMC in window
    fomc_cut:   float         # rate change in decimal (0.005 = 50bps)
    rth_open:   str           # UTC time string e.g. '13:30'
    rth_close:  str           # UTC time string e.g. '20:15'


@dataclass
class StrategyParams:
    """
    Tunable strategy parameters. Defaults match the 10-lot layered-TP config.

    Layered exits (layers):
        Each entry is (frac, tp_pts, new_sl_offset).
        frac           : fraction of n_lots to close at this TP level
        tp_pts         : TP level in spread points (move-space, always positive)
        new_sl_offset  : new stop level as move-pts from entry after this layer exits
                         (0.0 = breakeven; 0.25 = lock-in 0.25 pts; None = last layer)
        Partial fills are computed at the exact TP price (limit-order semantics).
        Remaining lots close at the current bar price on SL / EOD / FOMC.

    Regime gate (regime_gate):
        'none'         — no filtering.
        'fv_dev'       — block direction if 60-min rolling mean of (spread−FV) exceeds
                         fv_dev_threshold pts (persistent overextension vs fair value).
        'slope'        — block direction if prior-day linear slope of spread closes
                         exceeds slope_threshold pts/day (multi-day trend continuation).
        'fv_dev+slope' — union of both gates.
        'half_life'    — block both directions when rolling OU half-life of dev exceeds
                         half_life_max seconds (spread in slow-drift / trending regime).
        'kalman'       — block both directions for kalman_cooldown bars after a large
                         Kalman filter innovation (|z| > kalman_innov_thresh).
        'session'      — restrict entries to first/last N minutes of RTH only.
        'half_life+kalman' — union of half_life and kalman gates.
        'kalman+session'   — union of kalman and session gates.
        'return'       — legacy: 30-min directional return gate (co-linear with signal).

    n_lots:
        P&L in gross_usd / net_xxx is always per-lot-equivalent (total divided by
        n_lots). Multiply by n_lots for portfolio-level dollar amounts.
    """
    window:        str   = '10min'
    threshold:     float = 2.5
    tp:            float = 0.75   # reference only; simulation uses `layers`
    sl:            float = 0.50   # initial stop loss (points, before any layer exits)
    vol_gate_low:  float = 0.05
    vol_gate_high: float = 0.80
    fri_skip_min:  int   = 30
    fomc_pre_min:  int   = 60
    fomc_post_min: int   = 30
    open_blackout_min:  int = 2   # block entries for first N min after RTH open
    close_blackout_min: int = 2   # block entries for last N min before RTH close
    drift_4h_threshold: float = 0.10  # v1 drift gate: block shorts when 4h RTH drift > N pts
    div_yield:     float = 0.013
    tick:          float = 0.25
    mult:          float = 50.0
    n_lots:        int   = 10
    # Hard fees per calendar-spread lot (round-trip = 4 contract sides):
    #   Exchange: $1.15 × 4 = $4.60
    #   NFA:      $0.01 × 4 = $0.04
    #   Broker:   $0.85 × 4 = $3.40
    #   Total:               $8.04
    slip: dict = field(default_factory=lambda: {'Tight': 0.00, 'Mid': 6.25, 'Wide': 12.50})
    tc_inst:   float = 8.04   # total round-trip hard fees (exchange + NFA + broker)
    tc_retail: float = 13.40  # retail estimate (tc_inst + 0.5-tick slip per leg)

    # Layered TP/SL: (frac, tp_pts, new_sl_offset_from_entry); last entry has None sl
    layers: tuple = field(default_factory=lambda: (
        (0.90, 0.500,  0.250),   # 50% at +0.50; SL → breakeven
        (0.10, 0.750,  0.500),   # 30% at +0.75; SL → +0.25
        (0.00, 0.875,  None),    # 20% at +0.875; final layer
    ))

    # Low-z overlay: tighter 2-layer exit for low-conviction entries
    # Shorts:  abs(entry_z) < low_z_short_threshold → use low_z_layers
    # Longs:   entry_z >= low_z_long_threshold (less negative) → use low_z_layers
    low_z_short_threshold: float = 2.0
    low_z_long_threshold:  float = -2.0
    low_z_sl:              float = 0.50  # initial stop (pts) for low-z trades
    low_z_layers: tuple = field(default_factory=lambda: (
        (1.00, 0.250, 0.250),   # 50% at +0.25; SL → breakeven
        (0.00, 0.500, None),  # 50% at +0.50; final
    ))

    # High-conviction add-on: if |z_fill| > hc_threshold, add n_lots at T+2 bar
    hc_threshold: float = 3.0

    # Regime gate selection
    regime_gate:           str   = 'none'   # see docstring for all options
    fv_dev_window:         str   = '60min'  # rolling window for persistent FV deviation gate
    fv_dev_threshold:      float = 0.50     # dev gate trigger (pts)
    slope_lookback_days:   int   = 2        # prior RTH sessions for slope computation
    slope_threshold:       float = 0.75     # slope gate trigger (pts/day)
    # Legacy 'return' gate params
    regime_return_window:  int   = 1800   # 30-min lookback in 1s bars
    regime_threshold:      float = 0.50   # spread return trigger in points
    regime_check_interval: int   = 600    # re-evaluate every 10 min (in 1s bars)
    regime_suspend_min:    int   = 15     # suspend direction for 15 min after trigger
    # OU half-life gate
    half_life_window:      int   = 1800    # rolling AR(1) window in 1s bars (used when bar_res='1s')
    half_life_n_bars:      int   = 30      # rolling AR(1) window in BARS (used when bar_res != '1s')
    half_life_bar_res:     str   = '1s'    # bar resolution: '1s','1min','2min','5min','10min'
    half_life_max:         float = 1200.0  # suppress both dirs when half-life > N seconds
    # Kalman innovation gate
    kalman_Q:              float = 1e-5    # process noise (slow-drifting spread mean)
    kalman_R:              float = 0.01    # observation noise (~spread 1s variance)
    kalman_innov_thresh:   float = 3.0     # suppress after |normalized innovation| > this
    kalman_cooldown:       int   = 600     # bars to suppress post-surprise (10 min)
    # Session segmentation gate
    session_open_mins:     int   = 90      # active window: first N minutes of RTH
    session_close_mins:    int   = 90      # active window: last N minutes of RTH
    # OFI (Order Flow Imbalance) gate
    ofi_window_min:        int   = 5       # rolling window in minutes
    ofi_threshold:         float = 0.10   # normalized spread imbalance trigger ∈ (0, 2)

    @property
    def tickv(self) -> float:
        return self.tick * self.mult

    def net_cost_inst(self, slip_key: str = 'Tight') -> float:
        return self.slip[slip_key] + self.tc_inst

    def net_cost_retail(self) -> float:
        return self.slip['Tight'] + self.tc_retail


# ─────────────────────────────────────────────────────────────────────────────
# WINDOW REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# RTH times:
#   W1 (Sep 2024): EDT = UTC-4  →  08:30–15:15 ET = 12:30–19:15 UTC
#   W2 (Dec 2024): EST = UTC-5  →  08:30–15:15 ET = 13:30–20:15 UTC

WINDOWS: dict[str, WindowConfig] = {
    'W1': WindowConfig(
        front='ESU4', back='ESZ4',
        roll_start='2024-09-12',
        fomc_utc=pd.Timestamp('2024-09-18 18:00:00', tz='UTC'),
        fomc_cut=0.005,
        rth_open='12:30', rth_close='19:15',   # EDT (UTC-4)
    ),
    'W2': WindowConfig(
        front='ESZ4', back='ESH5',
        roll_start='2024-12-12',
        fomc_utc=pd.Timestamp('2024-12-18 19:00:00', tz='UTC'),
        fomc_cut=0.0025,
        rth_open='13:30', rth_close='20:15',   # EST (UTC-5)
    ),
    'W3': WindowConfig(
        front='ESH5', back='ESM5',
        roll_start='2025-03-13',
        fomc_utc=pd.Timestamp('2025-03-19 18:00:00', tz='UTC'),
        fomc_cut=0.0,                           # Fed held in Mar 2025
        rth_open='12:30', rth_close='19:15',   # EDT (UTC-4)
    ),
    'W4': WindowConfig(
        front='ESM5', back='ESU5',
        roll_start='2025-06-12',
        fomc_utc=pd.Timestamp('2025-06-18 18:00:00', tz='UTC'),
        fomc_cut=0.0,                           # Fed held in Jun 2025
        rth_open='12:30', rth_close='19:15',   # EDT (UTC-4)
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_sofr(data_dir: Path) -> pd.Series:
    raw = pd.read_csv(data_dir / 'SOFR.csv', parse_dates=['observation_date'],
                      index_col='observation_date')
    s = raw.iloc[:, 0].dropna() / 100.0
    return pd.Series(s.values, index=pd.DatetimeIndex(s.index).tz_localize('UTC'))


def load_dt_years(cfg: WindowConfig, data_dir: Path) -> float:
    defn  = pd.read_parquet(data_dir / f'definitions_{cfg.front}_{cfg.back}.parquet')
    exp_f = defn.loc[defn['symbol'] == cfg.front, 'expiration'].iloc[0]
    exp_b = defn.loc[defn['symbol'] == cfg.back,  'expiration'].iloc[0]
    dt_yr = (exp_b - exp_f).total_seconds() / (365.25 * 86400)
    print(f'  {cfg.front} expires: {exp_f}')
    print(f'  {cfg.back}  expires: {exp_b}')
    print(f'  ΔT = {dt_yr:.6f} yr  ({(exp_b - exp_f).days} days)')
    return dt_yr


def load_volume_gate(cfg: WindowConfig, params: StrategyParams, data_dir: Path) -> dict:
    files = sorted(glob.glob(str(data_dir / f'ohlcv1d_{cfg.front}_{cfg.back}_*.parquet')))
    if not files:
        print('  WARNING: no ohlcv1d files found — volume gate disabled (all days open)')
        return {}
    vol = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    piv = vol.pivot_table(index=vol.index.date, columns='symbol',
                          values='volume', aggfunc='sum')
    front_v    = piv[cfg.front] if cfg.front in piv.columns else pd.Series(0, index=piv.index)
    back_v     = piv[cfg.back]  if cfg.back  in piv.columns else pd.Series(0, index=piv.index)
    back_share = back_v / (front_v + back_v + 1e-9)
    gate = {d: bool(params.vol_gate_low < float(s) < params.vol_gate_high)
            for d, s in back_share.items()}
    for d in sorted(gate):
        ts = pd.Timestamp(str(d))
        if ts >= pd.Timestamp(cfg.roll_start) - pd.Timedelta('3d'):
            bs     = float(back_share.get(d, 0))
            status = 'OPEN ✓' if gate[d] else ('LOW  –' if bs <= params.vol_gate_low else 'HIGH ✗')
            print(f'    {d}  {ts.day_name()[:3]}  back_share={bs:.1%}  gate={status}')
    return gate


def load_volume_arc(cfg: WindowConfig, data_dir: Path) -> pd.DataFrame:
    """Daily back-share arc for the volume migration chart."""
    files = sorted(glob.glob(str(data_dir / f'ohlcv1d_{cfg.front}_{cfg.back}_*.parquet')))
    if not files:
        return pd.DataFrame()
    vol = pd.concat([pd.read_parquet(f) for f in files]).sort_index()
    piv = vol.pivot_table(index=vol.index.date, columns='symbol',
                          values='volume', aggfunc='sum')
    front_v = piv[cfg.front] if cfg.front in piv.columns else pd.Series(0, index=piv.index)
    back_v  = piv[cfg.back]  if cfg.back  in piv.columns else pd.Series(0, index=piv.index)
    arc = pd.DataFrame({
        'front_vol':  front_v,
        'back_vol':   back_v,
        'back_share': back_v / (front_v + back_v + 1e-9),
    })
    return arc[arc.index >= pd.to_datetime(cfg.roll_start).date()]


def build_fv(spread: pd.Series, front_mid: pd.Series,
             sofr_utc: pd.Series, dt_yr: float,
             cfg: WindowConfig, params: StrategyParams) -> pd.Series:
    daily_idx  = pd.date_range(spread.index[0].normalize(),
                               spread.index[-1].normalize(), freq='D', tz='UTC')
    sofr_daily = sofr_utc.reindex(daily_idx).ffill().bfill()
    r_f = pd.Series(
        sofr_daily.reindex(spread.index.normalize()).values,
        index=spread.index, dtype=float,
    ).ffill()
    if cfg.fomc_utc is not None:
        pre = r_f.index < cfg.fomc_utc
        if pre.any():
            pre_rate  = float(r_f[pre].iloc[-1])
            r_f[~pre] = pre_rate - cfg.fomc_cut
    return front_mid.reindex(r_f.index).ffill() * (r_f - params.div_yield) * dt_yr


def load_rth_bars(cfg: WindowConfig, params: StrategyParams,
                  sofr_utc: pd.Series, dt_yr: float,
                  data_dir: Path):
    """Load RTH 1-second bars. Returns (spread, fv, dev) aligned to same index."""
    COLS  = ['bid_px_00', 'ask_px_00', 'symbol']
    files = sorted(glob.glob(
        str(data_dir / f'mbp10_{cfg.front}_{cfg.back}_{cfg.roll_start}_*.parquet')))
    print(f'  Loading {len(files)} RTH day files', end='', flush=True)
    parts = []
    for f in files:
        df = pd.read_parquet(f, columns=COLS)
        df['mid'] = (df['bid_px_00'] + df['ask_px_00']) / 2
        wide = (df.groupby('symbol')[['mid']]
                .resample('1s').last().ffill()
                .unstack('symbol')
                .between_time(cfg.rth_open, cfg.rth_close))
        parts.append(wide)
        print('.', end='', flush=True)
    print(' done')
    full      = pd.concat(parts).sort_index()
    spread    = (full[('mid', cfg.back)] - full[('mid', cfg.front)]).dropna()
    front_mid = full[('mid', cfg.front)].reindex(spread.index).ffill()
    fv        = build_fv(spread, front_mid, sofr_utc, dt_yr, cfg, params)
    dev       = (spread - fv).dropna()
    return spread.reindex(dev.index), fv.reindex(dev.index), dev


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_z(dev: pd.Series, params: StrategyParams) -> np.ndarray:
    mu  = dev.rolling(params.window, min_periods=1).mean()
    sig = dev.rolling(params.window, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


def build_entry_mask(spread: pd.Series, vol_gate: dict,
                     cfg: WindowConfig, params: StrategyParams) -> np.ndarray:
    """
    Bool array: True = new entry allowed at this bar (direction-agnostic).

    Blocked by:
      1. Volume regime gate (daily back-share outside gate bounds)
      2. Friday opening filter (first fri_skip_min of any Friday RTH)
      3. FOMC blackout (fomc_pre_min before to fomc_post_min after announcement)

    Directional regime gate (trend suspension) is applied inside simulate().
    """
    idx  = spread.index
    mask = np.ones(len(idx), dtype=bool)

    if vol_gate:
        dates = np.array([t.date() for t in idx])
        for i, d in enumerate(dates):
            if not vol_gate.get(d, True):
                mask[i] = False

    fri_open_t  = pd.Timestamp(f'2000-01-01 {cfg.rth_open}').time()
    fri_cut_t   = (pd.Timestamp(f'2000-01-01 {cfg.rth_open}') +
                   pd.Timedelta(minutes=params.fri_skip_min)).time()
    is_fri      = (idx.weekday == 4)
    in_fri_open = (idx.time >= fri_open_t) & (idx.time < fri_cut_t)
    mask[is_fri & in_fri_open] = False

    if cfg.fomc_utc is not None:
        fomc_start    = cfg.fomc_utc - pd.Timedelta(minutes=params.fomc_pre_min)
        fomc_end      = cfg.fomc_utc + pd.Timedelta(minutes=params.fomc_post_min)
        in_fomc_block = (idx >= fomc_start) & (idx < fomc_end)
        mask[in_fomc_block] = False

    # RTH open / close blackout
    if params.open_blackout_min > 0:
        open_t     = pd.Timestamp(f'2000-01-01 {cfg.rth_open}').time()
        open_cut_t = (pd.Timestamp(f'2000-01-01 {cfg.rth_open}') +
                      pd.Timedelta(minutes=params.open_blackout_min)).time()
        mask[(idx.time >= open_t) & (idx.time < open_cut_t)] = False

    if params.close_blackout_min > 0:
        close_cut_t = (pd.Timestamp(f'2000-01-01 {cfg.rth_close}') -
                       pd.Timedelta(minutes=params.close_blackout_min)).time()
        close_t     = pd.Timestamp(f'2000-01-01 {cfg.rth_close}').time()
        mask[(idx.time >= close_cut_t) & (idx.time < close_t)] = False

    return mask


# ─────────────────────────────────────────────────────────────────────────────
# REGIME TREND GATE
# ─────────────────────────────────────────────────────────────────────────────

def _compute_regime_blocks(spread: pd.Series,
                            params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Directional suspension arrays for the regime trend gate.

    Every regime_check_interval bars, evaluate the regime_return_window spread
    return. If return > regime_threshold: suspend shorts for regime_suspend_min
    minutes. If return < -regime_threshold: suspend longs. Suspension extends
    (max) if re-triggered before expiry. Cross-day lookbacks are skipped.

    Returns (long_blocked, short_blocked): bool arrays aligned to spread.index.
    """
    n      = len(spread)
    prices = spread.values
    dates  = np.array([t.date() for t in spread.index])

    lookback     = params.regime_return_window
    threshold    = params.regime_threshold
    check_every  = params.regime_check_interval
    suspend_bars = params.regime_suspend_min * 60

    long_blocked  = np.zeros(n, dtype=bool)
    short_blocked = np.zeros(n, dtype=bool)
    long_until    = -1   # inclusive bar index until which longs are suspended
    short_until   = -1

    for i in range(n):
        if i <= long_until:
            long_blocked[i] = True
        if i <= short_until:
            short_blocked[i] = True

        if i >= lookback and i % check_every == 0:
            if dates[i] == dates[i - lookback]:   # same RTH session only
                ret = prices[i] - prices[i - lookback]
                if ret > threshold:
                    short_until = max(short_until, i + suspend_bars)
                elif ret < -threshold:
                    long_until = max(long_until, i + suspend_bars)

    return long_blocked, short_blocked


def _compute_fv_dev_gate(dev: pd.Series,
                          params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Persistent FV deviation gate.

    Computes a rolling mean of (spread - FV) over fv_dev_window. Blocks a direction
    if the spread has been persistently displaced from fair value, signaling that
    mean-reversion may not occur imminently.

    Returns (long_blocked, short_blocked): bool arrays aligned to dev.index.
    """
    rolling_mean = dev.rolling(params.fv_dev_window, min_periods=60).mean()
    arr = rolling_mean.fillna(0).values
    long_blocked  = arr < -params.fv_dev_threshold   # persistently below FV → longs risky
    short_blocked = arr >  params.fv_dev_threshold   # persistently above FV → shorts risky
    return long_blocked, short_blocked


def _compute_slope_gate(spread: pd.Series,
                         params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Prior-day slope filter.

    Fits a linear slope to the RTH closing prices of the last slope_lookback_days
    sessions. If slope > slope_threshold: block shorts (multi-day uptrend).
    If slope < -slope_threshold: block longs (multi-day downtrend).
    Uses only prior-day data — zero temporal overlap with the intraday signal.

    Returns (long_blocked, short_blocked): bool arrays aligned to spread.index.
    """
    n      = len(spread)
    prices = spread.values
    dates  = np.array([t.date() for t in spread.index])
    unique_dates = sorted(set(dates))

    long_blocked  = np.zeros(n, dtype=bool)
    short_blocked = np.zeros(n, dtype=bool)

    daily_closes: dict = {}
    for d in unique_dates:
        day_idx = np.where(dates == d)[0]
        if len(day_idx) > 0:
            daily_closes[d] = prices[day_idx[-1]]

    for i, d in enumerate(unique_dates):
        prior_dates = unique_dates[max(0, i - params.slope_lookback_days):i]
        if len(prior_dates) < 2:
            continue
        prior_closes = [daily_closes[pd_] for pd_ in prior_dates]
        slope = float(np.polyfit(range(len(prior_closes)), prior_closes, 1)[0])
        day_idx = np.where(dates == d)[0]
        if slope > params.slope_threshold:
            short_blocked[day_idx] = True
        elif slope < -params.slope_threshold:
            long_blocked[day_idx] = True

    return long_blocked, short_blocked


def _compute_half_life_gate(dev: pd.Series,
                             params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Rolling OU half-life gate with configurable bar resolution.

    Estimates mean-reversion speed via rolling AR(1) on dev = spread − FV.
    Half-life = ln(2) / κ (in seconds). Suppresses entries in both directions
    when the estimated half-life exceeds half_life_max (spread is in a
    slow-drift / trending regime rather than a fast-reverting one).

    Bar resolution: at '1s' the AR(1) β is near-zero almost always (noise
    dominates). Use '5min' or '10min' for meaningful regime separation.
    The gate state is forward-filled from bar resolution back to 1s.

    Returns (long_blocked, short_blocked): bool arrays aligned to dev.index.
    """
    _BAR_SECS = {'1s': 1, '1min': 60, '2min': 120, '5min': 300, '10min': 600}
    bar_res   = params.half_life_bar_res
    bar_secs  = _BAR_SECS.get(bar_res, 1)

    if bar_res == '1s':
        s_work   = pd.Series(dev.values.astype(float), index=dev.index)
        win_bars = params.half_life_window
    else:
        s_work   = dev.resample(bar_res).last().dropna().astype(float)
        win_bars = params.half_life_n_bars

    s_lag = s_work.shift(1)
    mp    = max(win_bars // 2, 5)

    roll_cov = ((s_work * s_lag).rolling(win_bars, min_periods=mp).mean()
                - s_work.rolling(win_bars, min_periods=mp).mean()
                * s_lag.rolling(win_bars, min_periods=mp).mean())
    roll_var = ((s_lag ** 2).rolling(win_bars, min_periods=mp).mean()
                - s_lag.rolling(win_bars, min_periods=mp).mean() ** 2)

    beta     = (roll_cov / roll_var).clip(lower=1e-9, upper=1 - 1e-9)
    hl_bars  = (np.log(2) / (-np.log(beta.abs()))).fillna(np.inf)
    hl_secs  = hl_bars * bar_secs
    blocked_bars = (hl_secs > params.half_life_max).astype(bool)

    if bar_res != '1s':
        blocked = blocked_bars.reindex(dev.index, method='ffill').fillna(False)
        arr     = blocked.values.astype(bool)
    else:
        arr = blocked_bars.values.astype(bool)

    return arr, arr.copy()


def _compute_kalman_gate(spread: pd.Series,
                          params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Kalman filter innovation gate.

    Fits a random-walk Kalman filter (state = slow-moving spread mean).
    Normalized innovation z_t = (spread_t − x̂_{t−1}) / √S_t ~ N(0,1) under
    model. When |z_t| > kalman_innov_thresh the spread has surprised — suppress
    new entries in both directions for kalman_cooldown bars.

    Returns (long_blocked, short_blocked): bool arrays aligned to spread.index.
    """
    Q, R     = params.kalman_Q, params.kalman_R
    thresh   = params.kalman_innov_thresh
    cooldown = params.kalman_cooldown

    prices     = spread.values.astype(float)
    n          = len(prices)
    innov_norm = np.zeros(n)
    x, P       = float(prices[0]), float(R)

    for i in range(1, n):
        P_pred        = P + Q
        innov         = prices[i] - x
        S             = P_pred + R
        innov_norm[i] = innov / np.sqrt(S)
        K             = P_pred / S
        x             = x + K * innov
        P             = (1.0 - K) * P_pred

    blocked     = np.zeros(n, dtype=bool)
    block_until = -1
    for i in range(n):
        if i <= block_until:
            blocked[i] = True
        if abs(innov_norm[i]) > thresh:
            block_until = max(block_until, i + cooldown)

    return blocked, blocked.copy()


def _compute_session_gate(spread: pd.Series, cfg: WindowConfig,
                           params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    Intraday session segmentation gate.

    Allows entries only during the first session_open_mins and last
    session_close_mins of each RTH session; blocks the midday period.
    (Schmidhuber & Safari arXiv:2501.16772: mean-reversion is cleanest
    during the opening and closing session windows.)

    Returns (long_blocked, short_blocked): bool arrays aligned to spread.index.
    """
    n       = len(spread)
    times   = spread.index
    blocked = np.ones(n, dtype=bool)   # default block; active windows open below

    open_h,  open_m  = map(int, cfg.rth_open.split(':'))
    close_h, close_m = map(int, cfg.rth_close.split(':'))

    dates = np.array([t.date() for t in times])
    for d in sorted(set(dates)):
        day_idx = np.where(dates == d)[0]
        if len(day_idx) == 0:
            continue
        rth_open_utc  = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                      hour=open_h,  minute=open_m,  tz='UTC')
        rth_close_utc = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                      hour=close_h, minute=close_m, tz='UTC')
        morning_end     = rth_open_utc  + pd.Timedelta(minutes=params.session_open_mins)
        afternoon_start = rth_close_utc - pd.Timedelta(minutes=params.session_close_mins)

        for i in day_idx:
            t = times[i]
            if t < morning_end or t >= afternoon_start:
                blocked[i] = False   # active session window

    return blocked, blocked.copy()


_OFI_CACHE_DIR = Path(__file__).parent.parent / 'results' / 'ofi_cache'

def _compute_drift_4h_gate(spread: pd.Series,
                            params: StrategyParams) -> tuple[np.ndarray, np.ndarray]:
    """
    RTH 4-hour intraday drift gate (shorts only) — strategy v1.

    Blocks a short entry at bar i when:
        spread[i] - spread[max(session_start_bar, i - 14400)] > drift_4h_threshold

    Lookback clips to the current session start so no overnight data bleeds in.
    Longs are never blocked by this gate.

    Returns (long_blocked, short_blocked).
    """
    n          = len(spread)
    prices     = spread.values
    dates      = np.array([t.date() for t in spread.index])
    lookback   = 4 * 3600        # 4 h in 1s bars
    threshold  = params.drift_4h_threshold

    # First bar index for each date
    sess_start = {}
    for i, d in enumerate(dates):
        if d not in sess_start:
            sess_start[d] = i

    short_blocked = np.zeros(n, dtype=bool)
    for i in range(n):
        anchor = max(sess_start[dates[i]], i - lookback)
        if prices[i] - prices[anchor] > threshold:
            short_blocked[i] = True

    return np.zeros(n, dtype=bool), short_blocked   # longs never blocked


def _compute_ofi_gate(spread: pd.Series, cfg, params) -> tuple[np.ndarray, np.ndarray]:
    """
    LOB imbalance gate using pre-computed 1s spread OFI.

    OFI = imb_front - imb_back, where imb = (Σ bid_sz - Σ ask_sz) / (Σ bid_sz + Σ ask_sz)

    Blocks longs  when rolling mean OFI < -threshold  (net selling on spread)
    Blocks shorts when rolling mean OFI > +threshold  (net buying  on spread)

    Requires cache built by notebooks/12_ofi_gate.py.
    """
    cache_path = _OFI_CACHE_DIR / f'{cfg.front}_{cfg.back}_ofi_1s.parquet'
    if not cache_path.exists():
        raise FileNotFoundError(
            f'OFI cache missing: {cache_path}\n'
            f'Run notebooks/12_ofi_gate.py first.'
        )
    ofi_raw = pd.read_parquet(cache_path)['ofi']
    ofi     = ofi_raw.reindex(spread.index, method='ffill').fillna(0.0)
    window  = params.ofi_window_min * 60
    rolling = ofi.rolling(window, min_periods=1).mean()
    long_blocked  = (rolling < -params.ofi_threshold).values
    short_blocked = (rolling >  params.ofi_threshold).values
    return long_blocked, short_blocked


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(spread: pd.Series, dev: pd.Series, z: np.ndarray,
             entry_mask: np.ndarray,
             cfg: WindowConfig, params: StrategyParams) -> pd.DataFrame:
    """
    Edge-triggered simulation with layered TP/SL exits and regime trend gate.

    P&L convention: gross_usd / net_xxx are per-lot-equivalent (1-lot basis).
    Multiply by params.n_lots for portfolio-level dollar amounts.

    Layered exit logic (from StrategyParams.layers):
      - Partial fills computed at exact TP price (limit-order semantics).
      - SL shifts upward after each layer exits (ratchet stop).
      - Remaining lots close at current bar price on SL / EOD / FOMC.

    Regime gate:
      - Dispatched via params.regime_gate: 'none','fv_dev','slope','fv_dev+slope','return'.
      - long_blocked[i]  → suppress new long entries at bar i.
      - short_blocked[i] → suppress new short entries at bar i.
      - Re-checked on the fill bar to cancel entries set one bar earlier.

    Force-close priority: FOMC announcement > RTH end-of-day > TP (all layers) > SL
    """
    prices = spread.values
    times  = spread.index
    n      = len(prices)
    layers = params.layers
    n_lots = params.n_lots

    dates   = np.array([t.date() for t in times])
    is_last = np.zeros(n, dtype=bool)
    is_last[-1] = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last[i] = True

    fomc_close = np.zeros(n, dtype=bool)
    if cfg.fomc_utc is not None:
        fomc_idx = np.searchsorted(times, cfg.fomc_utc)
        if fomc_idx < n:
            fomc_close[fomc_idx] = True

    gate = params.regime_gate
    if gate == 'fv_dev':
        long_blocked, short_blocked = _compute_fv_dev_gate(dev, params)
    elif gate == 'slope':
        long_blocked, short_blocked = _compute_slope_gate(spread, params)
    elif gate == 'fv_dev+slope':
        lb1, sb1 = _compute_fv_dev_gate(dev, params)
        lb2, sb2 = _compute_slope_gate(spread, params)
        long_blocked = lb1 | lb2
        short_blocked = sb1 | sb2
    elif gate == 'half_life':
        long_blocked, short_blocked = _compute_half_life_gate(dev, params)
    elif gate == 'kalman':
        long_blocked, short_blocked = _compute_kalman_gate(spread, params)
    elif gate == 'session':
        long_blocked, short_blocked = _compute_session_gate(spread, cfg, params)
    elif gate == 'half_life+kalman':
        lb1, sb1 = _compute_half_life_gate(dev, params)
        lb2, sb2 = _compute_kalman_gate(spread, params)
        long_blocked = lb1 | lb2
        short_blocked = sb1 | sb2
    elif gate == 'kalman+session':
        lb1, sb1 = _compute_kalman_gate(spread, params)
        lb2, sb2 = _compute_session_gate(spread, cfg, params)
        long_blocked = lb1 | lb2
        short_blocked = sb1 | sb2
    elif gate == 'drift_4h':
        long_blocked, short_blocked = _compute_drift_4h_gate(spread, params)
    elif gate == 'ofi':
        long_blocked, short_blocked = _compute_ofi_gate(spread, cfg, params)
    elif gate == 'return':
        long_blocked, short_blocked = _compute_regime_blocks(spread, params)
    else:  # 'none'
        long_blocked = np.zeros(n, dtype=bool)
        short_blocked = np.zeros(n, dtype=bool)

    # ── Per-trade state ───────────────────────────────────────────────────────
    pos             = 0        # +1 long / -1 short
    entry_i         = -1
    entry_px        = np.nan
    layer_idx       = 0        # next layer to TP
    sl_offset       = 0.0      # current stop as move-pts from entry
    trade_layers    = layers   # per-trade layer structure (set at fill)
    layer_lots      = []       # lots allocated per layer
    lots_open       = 0        # lots remaining in trade
    realized_pts_lot = 0.0     # accumulated per-lot-equivalent pts from partial exits
    min_px          = np.nan
    max_px          = np.nan
    bars_held       = 0
    pending_dir     = 0
    hc_watch        = False    # True at fill if |z_fill| > hc_threshold; triggers add-on at T+2
    hc_addon        = False    # True if add-on fired for current trade
    trade_n_lots    = n_lots   # actual lots in trade (2*n_lots after HC add-on)
    entry_z2_val    = np.nan   # z at T+2 bar (valid when hc_watch is True)

    trades = []

    for i in range(n):
        px = prices[i]
        zi = z[i]

        # ── Fill pending entry (next bar after signal) ────────────────────────
        if pending_dir != 0 and pos == 0:
            # Re-check regime on fill bar — cancel if direction now blocked
            if ((pending_dir ==  1 and long_blocked[i]) or
                    (pending_dir == -1 and short_blocked[i])):
                pending_dir = 0
            else:
                pos              = pending_dir
                entry_i          = i
                entry_px         = px
                layer_idx        = 0
                realized_pts_lot = 0.0
                # Low-z overlay: use tighter 2-layer structure for low-conviction entries
                entry_z_val = float(z[i])
                if ((pos == -1 and abs(entry_z_val) < params.low_z_short_threshold) or
                        (pos ==  1 and entry_z_val >= params.low_z_long_threshold)):
                    trade_layers = params.low_z_layers
                    sl_offset    = -params.low_z_sl
                else:
                    trade_layers = layers
                    sl_offset    = -params.sl
                # Allocate lots per layer; last layer absorbs rounding residual
                layer_lots = []
                rem = n_lots
                for k in range(len(trade_layers)):
                    if k < len(trade_layers) - 1:
                        lk = round(trade_layers[k][0] * n_lots)
                    else:
                        lk = rem
                    layer_lots.append(lk)
                    rem -= lk
                lots_open = n_lots
                hc_watch     = abs(entry_z_val) > params.hc_threshold
                hc_addon     = False
                trade_n_lots = n_lots
                entry_z2_val = np.nan
                min_px = max_px = px
                bars_held = 0
                pending_dir = 0

        # ── Manage open position ──────────────────────────────────────────────
        if pos != 0:
            # HC add-on: decision made at T+1 fill (|z_fill|>hc_threshold); execute at T+2.
            # No z re-check at T+2 — entry committed once hc_watch is set.
            if hc_watch and i == entry_i + 1:
                entry_z2_val = zi   # record z at T+2 for analysis only
                hc_watch = False
                blended = (entry_px + px) / 2.0
                entry_px = blended
                trade_n_lots = 2 * n_lots
                rem = trade_n_lots
                layer_lots_new = []
                for k in range(len(layers)):
                    if k < len(layers) - 1:
                        lk = round(layers[k][0] * trade_n_lots)
                    else:
                        lk = rem
                    layer_lots_new.append(lk)
                    rem -= lk
                layer_lots      = layer_lots_new
                lots_open       = trade_n_lots
                sl_offset       = -params.sl
                trade_layers    = layers
                layer_idx       = 0
                realized_pts_lot = 0.0
                hc_addon        = True
            bars_held += 1
            min_px = min(min_px, px)
            max_px = max(max_px, px)
            move   = pos * (px - entry_px)   # positive = in our favour

            # Process layer TPs (partial exits at exact TP price)
            while layer_idx < len(trade_layers):
                frac, tp_pts, new_sl = trade_layers[layer_idx]
                if move >= tp_pts:
                    realized_pts_lot += (layer_lots[layer_idx] / n_lots) * tp_pts
                    lots_open  -= layer_lots[layer_idx]
                    layer_idx  += 1
                    if new_sl is not None:
                        sl_offset = new_sl
                else:
                    break

            # Determine overall exit trigger
            exit_type = None
            if fomc_close[i]:
                exit_type = 'FOMC'
            elif is_last[i]:
                exit_type = 'EOD'
            elif lots_open <= 0:
                exit_type = 'TP'    # all layers exited via TP
            elif move <= sl_offset:
                exit_type = 'SL'

            if exit_type:
                # Remaining lots close at current bar price
                remaining_frac   = lots_open / n_lots
                final_pts_lot    = remaining_frac * move if lots_open > 0 else 0.0
                gross_pts        = realized_pts_lot + final_pts_lot
                gross_usd        = gross_pts * params.mult   # per-lot-equivalent USD

                if pos == 1:
                    mae = max(0.0, entry_px - min_px)
                    mfe = max(0.0, max_px - entry_px)
                else:
                    mae = max(0.0, max_px - entry_px)
                    mfe = max(0.0, entry_px - min_px)

                et = times[entry_i]
                trades.append({
                    'entry_time':     et,
                    'exit_time':      times[i],
                    'direction':      pos,
                    'dir_label':      'Long' if pos == 1 else 'Short',
                    'entry_spread':   entry_px,
                    'exit_spread':    px,
                    'entry_z':        float(z[entry_i]),
                    'gross_pts':      gross_pts,
                    'gross_usd':      gross_usd,
                    'bars_held':      bars_held,
                    'hold_min':       bars_held / 60.0,
                    'exit_type':      exit_type,
                    'layers_hit':     layer_idx,
                    'mae_pts':        mae,
                    'mfe_pts':        mfe,
                    'trade_date':     et.strftime('%Y-%m-%d'),
                    'day_label':      et.strftime('%a'),
                    'entry_hour_utc': et.hour + et.minute / 60.0,
                    'post_fomc':      cfg.fomc_utc is not None and et >= cfg.fomc_utc,
                    'hc_addon':       hc_addon,
                    'lot_scale':      trade_n_lots / n_lots,
                    'entry_z2':       entry_z2_val,
                })
                pos = 0; entry_i = -1; bars_held = 0; lots_open = 0
                min_px = max_px = entry_px = np.nan
                pending_dir = 0
                hc_watch = False; hc_addon = False; trade_n_lots = n_lots; entry_z2_val = np.nan

        # ── Detect entry signal (edge-triggered, fills next bar) ──────────────
        if pos == 0 and pending_dir == 0 and i > 0 and entry_mask[i]:
            zp = z[i - 1]
            if not np.isnan(zi) and not np.isnan(zp):
                if zp >= -params.threshold and zi < -params.threshold:
                    if not long_blocked[i]:
                        pending_dir =  1
                elif zp <= params.threshold and zi > params.threshold:
                    if not short_blocked[i]:
                        pending_dir = -1

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)

    # Net P&L: TC and slippage are per-lot costs; gross_usd is per-lot-equivalent
    for lbl, cost in params.slip.items():
        df[f'net_{lbl}'] = df['gross_usd'] - cost - params.tc_inst
    df['net_retail'] = df['gross_usd'] - params.slip['Tight'] - params.tc_retail

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

def bootstrap_ci(vals, n_boot: int = 5000, ci: float = 0.95):
    if len(vals) < 2:
        return np.nan, np.nan
    means = [np.mean(np.random.choice(vals, size=len(vals), replace=True))
             for _ in range(n_boot)]
    lo = np.percentile(means, (1 - ci) / 2 * 100)
    hi = np.percentile(means, (1 + ci) / 2 * 100)
    return lo, hi


def compute_stats(df: pd.DataFrame, params: StrategyParams) -> dict:
    """
    Compute summary statistics. gross_usd / net_xxx are per-lot-equivalent.
    Multiply _scaled keys by params.n_lots for portfolio-level dollar amounts.
    """
    n      = len(df)
    gross  = df['gross_usd']
    net_t  = df['net_Tight']
    wins   = df[gross > 0]
    losses = df[gross <= 0]

    w_sum = wins['gross_usd'].sum()
    l_sum = abs(losses['gross_usd'].sum()) if len(losses) > 0 else 1e-9

    t_g, p_g = spstats.ttest_1samp(gross, 0.0)
    t_n, p_n = spstats.ttest_1samp(net_t, 0.0)
    ci_lo, ci_hi = bootstrap_ci(net_t.values)

    mdd  = df['drawdown'].min()
    lots = params.n_lots

    # Layer-TP thresholds for MFE hit-rate analysis
    l1_tp = params.layers[0][1] if params.layers else params.tp
    l3_tp = params.layers[-1][1] if params.layers else params.tp

    return {
        'n':                    n,
        'n_lots':               lots,
        'wr':                   (gross > 0).mean() * 100,
        'pf':                   w_sum / l_sum,
        'avg_gross':            gross.mean(),
        'std_gross':            gross.std(),
        'tot_gross':            gross.sum(),
        'tot_gross_scaled':     gross.sum() * lots,
        'best':                 gross.max(),
        'worst':                gross.min(),
        'avg_hold_min':         df['hold_min'].mean(),
        'med_hold_min':         df['hold_min'].median(),
        'max_hold_min':         df['hold_min'].max(),
        'avg_layers_hit':       df['layers_hit'].mean() if 'layers_hit' in df.columns else 0,
        'tp_pct':               (df['exit_type'] == 'TP').mean() * 100,
        'sl_pct':               (df['exit_type'] == 'SL').mean() * 100,
        'eod_pct':              (df['exit_type'] == 'EOD').mean() * 100,
        'fomc_pct':             (df['exit_type'] == 'FOMC').mean() * 100,
        'mdd':                  mdd,
        'mdd_scaled':           mdd * lots,
        'recovery':             gross.sum() / abs(mdd) if mdd < 0 else float('inf'),
        'avg_net_tight':        net_t.mean(),
        'tot_net_tight':        net_t.sum(),
        'tot_net_tight_scaled': net_t.sum() * lots,
        'avg_net_mid':          df['net_Mid'].mean(),
        'avg_net_wide':         df['net_Wide'].mean(),
        'avg_net_retail':       df['net_retail'].mean(),
        'sharpe_pt':            net_t.mean() / net_t.std() if net_t.std() > 0 else 0,
        't_gross': t_g,  'p_gross': p_g,
        't_net':   t_n,  'p_net':   p_n,
        'ci_lo':   ci_lo, 'ci_hi':  ci_hi,
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
        'avg_mfe':   df['mfe_pts'].mean(),
        'avg_mae':   df['mae_pts'].mean(),
        'mfe_geL1':  (df['mfe_pts'] >= l1_tp).mean() * 100,   # reached Layer 1 TP (0.5 pts)
        'mfe_geL3':  (df['mfe_pts'] >= l3_tp).mean() * 100,   # reached full target (0.875 pts)
        'mfe_geTP':  (df['mfe_pts'] >= params.tp).mean() * 100, # compat: old single TP (0.75)
        'mae_geSL':  (df['mae_pts'] >= params.sl).mean() * 100,
    }
