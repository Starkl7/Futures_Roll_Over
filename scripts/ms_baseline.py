#!/usr/bin/env python3
"""
ms_baseline.py — Monoyios & Sarno (2002) ESTAR Nonlinear Mean-Reversion Baseline

Implements the ES calendar spread mean-reversion strategy motivated by:

  Monoyios, M. & Sarno, L. (2002). "Mean Reversion in Stock Index Futures
  Markets: A Nonlinear Analysis." Journal of Futures Markets, 22(4), 285–314.
  DOI: 10.1002/fut.10008

Background
----------
The paper models the log futures basis b_t = log(F_t / S_t) using an ESTAR
(Exponential Smooth-Transition Autoregressive) model:

  Δb_t = ρ·b_{t−1} + ρ*·b_{t−1}·Φ(γ; b_{t−1} − k) + ε_t
  Φ(γ; ξ) = 1 − exp(−γ²·ξ²)                         [Eq. 9, M&S 2002]

Key property: the basis is near a unit-root when close to equilibrium (Φ ≈ 0,
inner regime) and becomes strongly mean-reverting when far from equilibrium
(Φ → 1, outer regime).  This is the "arbitrage is like gravity" mechanism:
larger deviations attract stronger correction.

Adaptation to intraday calendar spreads
----------------------------------------
We replace the futures-vs-spot basis with the ES calendar spread deviation
from cost-of-carry fair value (dev_t = spread_t − FV_t), normalize it to a
z-score (ξ_t = z_t), and use Φ(γ; z_t) as a continuous regime indicator.

Entry  : z edge-crosses ±entry_z  AND  Φ ≥ phi_min (ESTAR regime gate).
Exit   : |z| < exit_z  (deviation has mean-reverted to near-equilibrium)
         OR hard SL  OR EOD  OR FOMC announcement.

Key differences from strategy.py
----------------------------------
  strategy.py  → fixed z-score threshold, layered TP/SL, 8 regime gates,
                 session splits, HC add-on.
  ms_baseline  → ESTAR Φ gate only, single SL, mean-reversion exit (no TP),
                 no session/regime/HC logic.  Pure ESTAR signal.

Usage
-----
  python notebooks/ms_baseline.py
  python notebooks/ms_baseline.py --gamma 0.3 --entry-z 2.0 --exit-z 0.5
  python notebooks/ms_baseline.py --phi-min 0.5 --entry-z 2.0 --gamma 0.5
"""

import argparse
import sys
import numpy as np
import pandas as pd
import scipy.stats as spstats
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from strategy import (
    WINDOWS, StrategyParams,
    load_sofr, load_dt_years, load_volume_gate, load_rth_bars, build_entry_mask,
)

DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')


# ── Parameters ────────────────────────────────────────────────────────────────

@dataclass
class MSParams:
    """
    Parameters for the Monoyios-Sarno ESTAR baseline.

    gamma      ESTAR speed parameter γ.  Governs how sharply Φ transitions
               from the inner (unit-root) to the outer (mean-reverting) regime.
               Estimated at ~0.5–1.5 on normalised deviations from the paper.
               Default 0.5 gives Φ≈0.22 at |z|=1, Φ≈0.63 at |z|=2, Φ≈0.89 at |z|=3.

    phi_min    Minimum Φ required to enter.  0.0 disables the ESTAR gate and
               makes entry purely threshold-based (|z| > entry_z).  Values above 0
               suppress entries in the transition zone, keeping only the outer
               high-Φ regime where mean reversion is fastest.
                 phi_min=0.0  → ESTAR gate off; same as pure z-threshold
                 phi_min=0.5  → requires |z| ≥ 1.67 at γ=0.5
                 phi_min=0.63 → requires |z| ≥ 2.0  at γ=0.5 (equivalent to entry_z=2)

    entry_z    |z| edge-crossing threshold.  Signal triggers when z crosses from
               inside the band (|z| < entry_z) to outside (|z| > entry_z).
               With phi_min=0 this is the sole entry filter.

    exit_z     Mean-reversion exit threshold.  Position closes when |z| falls
               below exit_z, interpreted as the deviation having returned to
               the ESTAR equilibrium band.  This is the M&S-motivated exit:
               the strategy harvests the mean-reversion move, not a fixed TP.

    sl         Hard stop loss in spread points.  Not in the paper (which focuses
               on unconditional convergence), but required for practical risk
               management at intraday frequency.
    """
    # Signal
    window:             str   = '10min'   # rolling z-score lookback (same as strategy.py)
    entry_z:            float = 2.0       # |z| crossing threshold for new entries
    exit_z:             float = 0.25      # mean-reversion exit: close when |z| < exit_z
    sl:                 float = 0.50      # hard stop loss in spread points
    gamma:              float = 0.5       # ESTAR γ
    phi_min:            float = 0.0       # min Φ to enter (0 = disabled)
    # Position
    n_lots:             int   = 10
    div_yield:          float = 0.013
    tick:               float = 0.25
    mult:               float = 50.0
    # Transaction costs (identical to strategy.py)
    tc_inst:            float = 8.04      # exchange + NFA + broker, round-trip
    slip_tight:         float = 0.00      # synthetic mid fill: zero slippage
    # Filters (same defaults as strategy.py)
    vol_gate_low:       float = 0.05
    vol_gate_high:      float = 0.80
    fri_skip_min:       int   = 30
    open_blackout_min:  int   = 2
    close_blackout_min: int   = 2

    @property
    def tc_total(self) -> float:
        return self.tc_inst + self.slip_tight

    def to_strategy_params(self) -> StrategyParams:
        """Minimal StrategyParams for data-loading helpers in strategy.py."""
        return StrategyParams(
            div_yield          = self.div_yield,
            vol_gate_low       = self.vol_gate_low,
            vol_gate_high      = self.vol_gate_high,
            fri_skip_min       = self.fri_skip_min,
            open_blackout_min  = self.open_blackout_min,
            close_blackout_min = self.close_blackout_min,
            fomc_pre_min       = 0,    # full-day FOMC exclusion handled in ms_simulate
            fomc_post_min      = 0,
        )


# ── ESTAR core ────────────────────────────────────────────────────────────────

def estar_phi(z: np.ndarray, gamma: float) -> np.ndarray:
    """
    ESTAR transition function  Φ(γ; z) = 1 − exp(−γ²·z²)  [Eq. 9, M&S 2002].

    Input z is the rolling-normalised FV deviation (our ξ_t in paper notation).
    Output is bounded in [0, 1]:
      Φ → 0  when z → 0     inner regime: near unit root, no profitable arbitrage
      Φ → 1  when |z| → ∞   outer regime: fast mean reversion, trade with conviction
    """
    return 1.0 - np.exp(-gamma**2 * z**2)


def _rolling_z(dev: pd.Series, window: str) -> np.ndarray:
    """Rolling z-score of FV deviation.  Returns NaN during warmup."""
    mu  = dev.rolling(window, min_periods=1).mean()
    sig = dev.rolling(window, min_periods=1).std().replace(0, np.nan)
    return ((dev - mu) / sig).values


# ── Simulation ────────────────────────────────────────────────────────────────

def ms_simulate(
    spread:     pd.Series,
    dev:        pd.Series,         # noqa: F841  (kept for API symmetry)
    z:          np.ndarray,
    phi:        np.ndarray,
    entry_mask: np.ndarray,        # from build_entry_mask (vol, fri, open/close)
    cfg,
    params:     MSParams,
) -> pd.DataFrame:
    """
    Tick-by-tick ESTAR mean-reversion backtest on 1-second RTH bars.

    Entry logic (edge-triggered)
    ----------------------------
    Signal fires at bar i when:
      (a) z[i-1] was INSIDE ±entry_z band and z[i] has crossed OUTSIDE, AND
      (b) phi[i] >= phi_min  (ESTAR: deviation is in mean-reverting outer regime), AND
      (c) entry_mask[i] is True (vol gate + Fri skip + open/close blackout), AND
      (d) bar i is NOT on the FOMC calendar day.
    Position fills at bar i+1 (T+1 next-bar execution, same as strategy.py).

    Exit logic (priority order)
    ---------------------------
      1. FOMC announcement bar   → force close.
      2. EOD last bar of session → force close.
      3. Hard SL: move ≤ −sl    → stop loss.
      4. Mean-reversion exit: |z| < exit_z → deviation back near equilibrium.
    """
    prices = spread.values
    times  = spread.index
    n      = len(prices)
    dates  = np.array([t.date() for t in times])

    # EOD: last bar of each RTH day
    is_last = np.zeros(n, dtype=bool)
    is_last[-1] = True
    for i in range(n - 1):
        if dates[i] != dates[i + 1]:
            is_last[i] = True

    # FOMC: force-close at announcement bar; block ALL entries that calendar day
    fomc_close = np.zeros(n, dtype=bool)
    fomc_day   = np.zeros(n, dtype=bool)
    if cfg.fomc_utc is not None:
        fomc_idx = int(np.searchsorted(times, cfg.fomc_utc))
        if fomc_idx < n:
            fomc_close[fomc_idx] = True
        fomc_date = cfg.fomc_utc.date()
        for i, t in enumerate(times):
            if t.date() == fomc_date:
                fomc_day[i] = True

    # Combined entry gate: vol/fri/blackout mask AND NOT FOMC day
    active = entry_mask & ~fomc_day

    # ── Per-trade state ───────────────────────────────────────────────────────
    pos         = 0
    entry_i     = -1
    entry_px    = np.nan
    min_px      = np.nan
    max_px      = np.nan
    bars_held   = 0
    pending_dir = 0
    trades      = []

    for i in range(n):
        px = prices[i]
        zi = z[i]
        pi = phi[i]

        # ── Fill pending entry at T+1 ─────────────────────────────────────────
        if pending_dir != 0 and pos == 0:
            if not active[i]:
                pending_dir = 0          # gate closed on fill bar → cancel
            else:
                pos         = pending_dir
                entry_i     = i
                entry_px    = px
                min_px      = max_px = px
                bars_held   = 0
                pending_dir = 0

        # ── Manage open position ──────────────────────────────────────────────
        if pos != 0:
            bars_held += 1
            if px < min_px:
                min_px = px
            if px > max_px:
                max_px = px
            move = pos * (px - entry_px)

            exit_type = None
            if fomc_close[i]:
                exit_type = 'FOMC'
            elif is_last[i]:
                exit_type = 'EOD'
            elif move <= -params.sl:
                exit_type = 'SL'
            elif abs(zi) < params.exit_z:
                exit_type = 'MR'     # mean-reversion exit (M&S: basis back at equilibrium)

            if exit_type is not None:
                if pos == 1:
                    mae = max(0.0, entry_px - min_px)
                    mfe = max(0.0, max_px   - entry_px)
                else:
                    mae = max(0.0, max_px   - entry_px)
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
                    'entry_phi':      float(phi[entry_i]),
                    'gross_pts':      move,
                    'gross_usd':      move * params.mult,
                    'bars_held':      bars_held,
                    'hold_min':       bars_held / 60.0,
                    'exit_type':      exit_type,
                    'mae_pts':        mae,
                    'mfe_pts':        mfe,
                    'trade_date':     et.strftime('%Y-%m-%d'),
                    'day_label':      et.strftime('%a'),
                    'entry_hour_utc': et.hour + et.minute / 60.0,
                    'post_fomc':      cfg.fomc_utc is not None and et >= cfg.fomc_utc,
                })
                pos = 0; entry_i = -1; bars_held = 0
                min_px = max_px = entry_px = np.nan
                pending_dir = 0

        # ── Detect new entry signal (edge-triggered, fills T+1) ───────────────
        if pos == 0 and pending_dir == 0 and i > 0 and active[i]:
            zp = z[i - 1]
            if not np.isnan(zi) and not np.isnan(zp):
                if pi >= params.phi_min:   # ESTAR gate
                    if zp >= -params.entry_z and zi < -params.entry_z:
                        pending_dir =  1   # spread < FV → long (expect reversion up)
                    elif zp <= params.entry_z and zi > params.entry_z:
                        pending_dir = -1   # spread > FV → short (expect reversion down)

    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df['net_Tight']  = df['gross_usd'] - params.tc_total
    df['cum_gross']  = df['gross_usd'].cumsum()
    df['peak_gross'] = df['cum_gross'].cummax()
    df['drawdown']   = df['cum_gross'] - df['peak_gross']
    df['cum_net']    = df['net_Tight'].cumsum()
    df['trade_num']  = np.arange(1, len(df) + 1)
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def _bootstrap_ci(
    vals: np.ndarray,
    n_boot: int = 5_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng  = np.random.default_rng(seed)
    boot = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    lo   = float(np.percentile(boot, 100 * (1 - ci) / 2))
    hi   = float(np.percentile(boot, 100 * (1 + ci) / 2))
    return lo, hi


def ms_stats(df: pd.DataFrame, params: MSParams) -> dict:
    if df.empty:
        return {}

    gross = df['gross_usd']
    net   = df['net_Tight']
    wins  = gross[gross > 0]
    losses = gross[gross <= 0]
    mdd   = df['drawdown'].min()

    t_g, p_g = spstats.ttest_1samp(gross, 0.0)
    t_n, p_n = spstats.ttest_1samp(net,   0.0)
    ci_lo, ci_hi = _bootstrap_ci(net.values)

    return {
        'n':               len(df),
        'wr':              (gross > 0).mean() * 100,
        'pf':              wins.sum() / (abs(losses.sum()) or 1e-9),
        'avg_gross':       gross.mean(),
        'std_gross':       gross.std(),
        'avg_net':         net.mean(),
        'std_net':         net.std(),
        'tot_gross_scaled': gross.sum() * params.n_lots,
        'tot_net_scaled':  net.sum() * params.n_lots,
        'mdd_scaled':      mdd * params.n_lots,
        'recovery':        gross.sum() / abs(mdd) if mdd < 0 else float('inf'),
        'sharpe_pt':       net.mean() / net.std() if net.std() > 0 else 0.0,
        'p_gross':         p_g,
        'p_net':           p_n,
        'ci_lo':           ci_lo,
        'ci_hi':           ci_hi,
        'mr_pct':          (df['exit_type'] == 'MR').mean()   * 100,
        'sl_pct':          (df['exit_type'] == 'SL').mean()   * 100,
        'eod_pct':         (df['exit_type'] == 'EOD').mean()  * 100,
        'fomc_pct':        (df['exit_type'] == 'FOMC').mean() * 100,
        'avg_hold_min':    df['hold_min'].mean(),
        'med_hold_min':    df['hold_min'].median(),
        'long_n':          int((df['direction'] ==  1).sum()),
        'short_n':         int((df['direction'] == -1).sum()),
        'long_wr':         float((df.loc[df['direction'] ==  1, 'gross_usd'] > 0).mean() * 100)
                           if (df['direction'] ==  1).any() else 0.0,
        'short_wr':        float((df.loc[df['direction'] == -1, 'gross_usd'] > 0).mean() * 100)
                           if (df['direction'] == -1).any() else 0.0,
        'avg_phi':         df['entry_phi'].mean(),
    }


# ── Per-window runner ─────────────────────────────────────────────────────────

def run_window(wname: str, params: MSParams) -> pd.DataFrame:
    """Load data, build signal, and run ESTAR simulation for one roll window."""
    cfg = WINDOWS[wname]
    sp  = params.to_strategy_params()

    sofr     = load_sofr(DATA_DIR)
    dt_yr    = load_dt_years(cfg, DATA_DIR)
    vol_gate = load_volume_gate(cfg, sp, DATA_DIR)
    spread, fv, dev = load_rth_bars(cfg, sp, sofr, dt_yr, DATA_DIR)

    z    = _rolling_z(dev, params.window)
    phi  = estar_phi(z, params.gamma)
    mask = build_entry_mask(spread, vol_gate, cfg, sp)

    df = ms_simulate(spread, dev, z, phi, mask, cfg, params)
    if not df.empty:
        df['window'] = wname
    return df


# ── Reporting ─────────────────────────────────────────────────────────────────

def _stars(p: float) -> str:
    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'
    return ''


def _print_window(wname: str, df: pd.DataFrame, params: MSParams) -> None:
    if df.empty:
        print(f'  {wname}: no trades')
        return
    s = ms_stats(df, params)
    print(
        f"  {wname}  n={s['n']:4d}  WR={s['wr']:.1f}%  PF={s['pf']:.2f}"
        f"  avg_gross=${s['avg_gross']:+.2f}  avg_net=${s['avg_net']:+.2f}"
        f"  p={s['p_gross']:.4f}{_stars(s['p_gross'])}"
    )
    print(
        f"        tot_net(10-lot)=${s['tot_net_scaled']:+,.0f}"
        f"  MDD=${s['mdd_scaled']:+,.0f}  RF={s['recovery']:.2f}"
        f"  Sharpe={s['sharpe_pt']:+.4f}  avg_Φ={s['avg_phi']:.3f}"
    )
    print(
        f"        Exits: MR={s['mr_pct']:.0f}%  SL={s['sl_pct']:.0f}%"
        f"  EOD={s['eod_pct']:.0f}%"
        f"  hold_med={s['med_hold_min']:.1f}min"
    )
    print(
        f"        Long: n={s['long_n']} WR={s['long_wr']:.1f}%  "
        f"Short: n={s['short_n']} WR={s['short_wr']:.1f}%"
    )


def _print_pool(label: str, df: pd.DataFrame, params: MSParams) -> None:
    if df.empty:
        print(f'\n  {label}: no trades')
        return
    s   = ms_stats(df, params)
    sig = _stars(s['p_gross'])
    print(
        f"\n  {label}:"
        f"  n={s['n']}  avg_gross=${s['avg_gross']:+.2f}  avg_net=${s['avg_net']:+.2f}"
        f"  p={s['p_gross']:.4f}{sig}"
    )
    print(
        f"           95%CI=[${s['ci_lo']:+.2f}, ${s['ci_hi']:+.2f}]"
        f"  Sharpe(per-trade)={s['sharpe_pt']:+.4f}"
    )
    print(
        f"           tot_net(10-lot)=${s['tot_net_scaled']:+,.0f}"
        f"  MDD=${s['mdd_scaled']:+,.0f}  RF={s['recovery']:.2f}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Monoyios-Sarno (2002) ESTAR nonlinear baseline'
    )
    parser.add_argument('--gamma',   type=float, default=0.5,
                        help='ESTAR γ parameter (default 0.5)')
    parser.add_argument('--entry-z', type=float, default=2.0,
                        help='|z| entry crossing threshold (default 2.0)')
    parser.add_argument('--exit-z',  type=float, default=0.25,
                        help='Mean-reversion exit |z| threshold (default 0.25)')
    parser.add_argument('--sl',      type=float, default=0.50,
                        help='Hard stop loss in spread points (default 0.50)')
    parser.add_argument('--phi-min', type=float, default=0.0,
                        help='Minimum Φ to enter; 0=ESTAR gate off (default 0.0)')
    parser.add_argument('--windows', nargs='+', default=['W1', 'W2', 'W3', 'W4'],
                        choices=['W1', 'W2', 'W3', 'W4'],
                        help='Windows to run (default: all four)')
    args = parser.parse_args()

    params = MSParams(
        gamma   = args.gamma,
        entry_z = args.entry_z,
        exit_z  = args.exit_z,
        sl      = args.sl,
        phi_min = args.phi_min,
    )

    # Φ at entry_z (for display: shows how much mean-reversion ESTAR assigns at threshold)
    phi_at_entry = float(estar_phi(np.array([params.entry_z]), params.gamma)[0])

    print('=' * 72)
    print('Monoyios & Sarno (2002) — ESTAR Nonlinear Mean-Reversion Baseline')
    print(f'  γ={params.gamma}  Φ(γ,entry_z)={phi_at_entry:.3f}  '
          f'phi_min={params.phi_min}')
    print(f'  entry_z=±{params.entry_z}  exit_z=|z|<{params.exit_z}  '
          f'sl={params.sl}pt  window={params.window}')
    print(f'  TC=${params.tc_total:.2f}/lot (synthetic mid, zero slippage)  '
          f'n_lots={params.n_lots}')
    print('=' * 72)

    all_dfs: dict[str, pd.DataFrame] = {}
    for wname in args.windows:
        cfg = WINDOWS[wname]
        print(f'\n── {wname}  {cfg.front}/{cfg.back} ──')
        df = run_window(wname, params)
        all_dfs[wname] = df
        _print_window(wname, df, params)

    # Aggregate pools only if all four windows were run
    if set(args.windows) == {'W1', 'W2', 'W3', 'W4'}:
        is_parts  = [all_dfs[w] for w in ('W1', 'W2') if not all_dfs[w].empty]
        oos_parts = [all_dfs[w] for w in ('W3', 'W4') if not all_dfs[w].empty]
        is_df     = pd.concat(is_parts,  ignore_index=True) if is_parts  else pd.DataFrame()
        oos_df    = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()

        print('\n' + '=' * 72)
        print('AGGREGATE  (IS = W1+W2 in-sample  |  OOS = W3+W4 out-of-sample)')
        print('=' * 72)
        _print_pool('IS  (W1+W2)', is_df,  params)
        _print_pool('OOS (W3+W4)', oos_df, params)

        print('\n' + '─' * 72)
        print('Reference — strategy.py V1 (drift_4h gate, all sessions, n=670 OOS):')
        print('  avg_net=+$2.01/lot  p=0.027**  95%CI=[+$0.24, +$3.79]')
        print('  tot_net(10-lot)=+$10,488  MDD=-$3,115  RF=3.37')
        print('─' * 72)

    print()


if __name__ == '__main__':
    main()
