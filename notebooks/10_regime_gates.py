#!/usr/bin/env python3
"""
10_regime_gates.py — Feasibility and P&L impact study for 4 regime filter ideas.

Ideas:
  A — OU half-life gate      : entry only when spread is fast-reverting (half-life < 20 min)
  B — Kalman innovation gate : suppress entry after large Kalman surprise (|z| > 3)
  C — Session segmentation   : restrict entries to first/last 90 min of RTH
  F — OFI gate               : feasibility analysis (requires MBP-10 tick data)

For each of A/B/C:
  1. Pre-flight: which of the saved baseline trades would have been blocked, and were
     they winners or losers? This tells us gate direction quality without re-running.
  2. Backtest: full simulation with the gate; compared against no-gate baseline.
  3. Visualization: gate signal overlaid on timeseries + trade distribution charts.

Figures saved to results/gate_research/
Run: python notebooks/10_regime_gates.py
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from strategy import (
    WINDOWS, StrategyParams,
    _compute_half_life_gate, _compute_kalman_gate, _compute_session_gate,
)
from run_backtest import run_window

DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
RESULTS  = Path(__file__).parent.parent / 'results'
FIG_DIR  = RESULTS / 'gate_research'
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_PAIRS = {
    'W1': 'ESU4_ESZ4_20240912',
    'W2': 'ESZ4_ESH5_20241212',
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_saved(pair: str, gate: str = 'none'):
    base   = RESULTS / pair / gate.replace('+', '_')
    trades = pd.read_parquet(base / 'trades.parquet')
    ts     = pd.read_parquet(base / 'timeseries.parquet')
    with open(base / 'stats.json') as f:
        stats = json.load(f)
    return trades, ts, stats


def fmt_stats(label: str, stats: dict) -> str:
    return (f"  {label:<22} n={stats['n']:<4} WR={stats['wr']:.1f}%  "
            f"PF={stats['pf']:.2f}  avg=${stats['avg_gross']:>7.2f}  "
            f"tot=${stats['tot_gross']:>6.0f}  "
            f"MDD=${stats.get('mdd_scaled', 0):>7.0f}  "
            f"t={stats['t_gross']:>6.3f}  p={stats['p_gross']:.4f}")


def run_gate(wkey: str, gate: str, force: bool = True) -> dict:
    """Run backtest for a gate, suppress verbose output, return stats."""
    params = StrategyParams(n_lots=10, regime_gate=gate)
    print(f"    [{wkey}] {gate:<20} ... ", end='', flush=True)
    with _quiet():
        run_window(wkey, params, force=force)
    pair = WINDOW_PAIRS[wkey]
    _, _, stats = load_saved(pair, gate)
    print(f"n={stats['n']}  WR={stats['wr']:.1f}%  PF={stats['pf']:.2f}  "
          f"avg=${stats['avg_gross']:.2f}  MDD=${stats.get('mdd_scaled',0):.0f}")
    return stats


def preflight(gate_name: str, lb: np.ndarray, sb: np.ndarray,
              df_base: pd.DataFrame, ts_idx: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Check which baseline trades would be blocked by this gate.
    Uses signal bar (entry_time) as proxy — accurate to ±1 bar.
    """
    blocked_col = []
    for _, row in df_base.iterrows():
        et  = row['entry_time']
        pos = ts_idx.searchsorted(et, side='left')
        pos = min(pos, len(lb) - 1)
        bl  = lb[pos] if row['direction'] == 1 else sb[pos]
        blocked_col.append(bool(bl))

    df = df_base.copy()
    df['blocked'] = blocked_col
    blk   = df[df['blocked']]
    unblk = df[~df['blocked']]

    print(f"\n  Pre-flight '{gate_name}': "
          f"{df['blocked'].sum()}/{len(df)} baseline entries blocked "
          f"({100 * df['blocked'].mean():.0f}%)")

    for subset, label in [(blk, 'Blocked '), (unblk, 'Unblocked')]:
        if len(subset) == 0:
            continue
        wr  = 100 * (subset['gross_usd'] > 0).mean()
        avg = subset['gross_usd'].mean()
        tot = subset['gross_usd'].sum()
        print(f"    {label}: n={len(subset):>3}  WR={wr:>5.1f}%  "
              f"avg=${avg:>7.2f}  tot=${tot:>7.1f}")

    for dv, dl in [(1, 'Long'), (-1, 'Short')]:
        sub = df[df['direction'] == dv]
        n_bl = sub['blocked'].sum()
        n_ok = (~sub['blocked']).sum()
        if len(sub):
            print(f"    {dl:<6}: blocked={n_bl}  kept={n_ok}")

    # Also report gate activity rate across all RTH bars
    pct = 100 * (lb | sb).mean()
    print(f"    Gate active {pct:.1f}% of all RTH bars")

    return df


def rolling_half_life(dev: pd.Series, window: int = 1800) -> np.ndarray:
    """Return rolling OU half-life series in MINUTES (for plotting)."""
    s    = pd.Series(dev.values.astype(float), index=dev.index)
    s_lg = s.shift(1)
    mp   = window // 2
    rc   = ((s * s_lg).rolling(window, min_periods=mp).mean()
            - s.rolling(window, min_periods=mp).mean()
            * s_lg.rolling(window, min_periods=mp).mean())
    rv   = ((s_lg ** 2).rolling(window, min_periods=mp).mean()
            - s_lg.rolling(window, min_periods=mp).mean() ** 2)
    beta = (rc / rv).clip(lower=1e-9, upper=1 - 1e-9)
    hl   = (np.log(2) / (-np.log(beta.abs()))).fillna(np.inf)
    return (hl / 60.0).values   # minutes


def kalman_innovations(spread: pd.Series, Q: float = 1e-5, R: float = 0.01) -> np.ndarray:
    """Return normalized Kalman innovations for the spread (for plotting)."""
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
    return innov_norm


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def shade_gate(ax, times, blocked: np.ndarray, color='salmon', alpha=0.25, label=None):
    """Shade regions where blocked=True."""
    in_block   = False
    block_start = None
    drawn_label = False
    for i, (t, bl) in enumerate(zip(times, blocked)):
        if bl and not in_block:
            block_start = t
            in_block    = True
        elif not bl and in_block:
            lbl = label if not drawn_label else None
            ax.axvspan(block_start, t, color=color, alpha=alpha, label=lbl)
            drawn_label = True
            in_block    = False
    if in_block and block_start is not None:
        ax.axvspan(block_start, times[-1], color=color, alpha=alpha)


def plot_half_life(wkey: str, ts: pd.DataFrame, df_base: pd.DataFrame,
                   lb: np.ndarray, sb: np.ndarray, half_life_max_min: float):
    params = StrategyParams()
    hl     = rolling_half_life(ts['dev'], params.half_life_window)
    times  = ts.index

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{wkey}: OU Half-life Gate  (threshold = {half_life_max_min:.0f} min)',
                 fontsize=13)

    # Panel 1: Spread vs FV
    ax = axes[0]
    ax.plot(times, ts['spread'], lw=0.7, color='steelblue', label='Spread')
    ax.plot(times, ts['fv'],     lw=0.7, color='tomato',    label='FV', linestyle='--')
    shade_gate(ax, times, lb | sb, color='salmon', alpha=0.18, label='Gated (slow HL)')
    for _, row in df_base.iterrows():
        c = 'green' if row['gross_usd'] > 0 else 'red'
        ax.axvline(row['entry_time'], color=c, alpha=0.5, lw=0.8)
    ax.set_ylabel('Spread pts'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 2: Dev (spread - FV)
    ax = axes[1]
    ax.plot(times, ts['dev'], lw=0.7, color='purple', label='dev = spread−FV')
    shade_gate(ax, times, lb | sb, color='salmon', alpha=0.18)
    ax.axhline(0, color='black', lw=0.5, linestyle='--')
    ax.set_ylabel('Deviation pts'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # Panel 3: Rolling half-life
    ax = axes[2]
    hl_plot = np.where(hl > 120, 120, hl)   # cap display at 120 min
    ax.plot(times, hl_plot, lw=0.7, color='darkorange', label='Half-life (min, capped 120)')
    ax.axhline(half_life_max_min, color='red', lw=1.0, linestyle='--',
               label=f'Threshold {half_life_max_min:.0f} min')
    ax.fill_between(times, half_life_max_min, hl_plot,
                    where=hl_plot > half_life_max_min, color='salmon', alpha=0.3)
    ax.set_ylabel('Half-life (min)'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('UTC time')

    plt.tight_layout()
    fname = FIG_DIR / f'{wkey}_half_life_gate.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"    Figure saved → {fname.relative_to(RESULTS.parent)}")


def plot_kalman(wkey: str, ts: pd.DataFrame, df_base: pd.DataFrame,
                lb: np.ndarray, sb: np.ndarray, thresh: float):
    params = StrategyParams()
    innov  = kalman_innovations(ts['spread'], params.kalman_Q, params.kalman_R)
    times  = ts.index

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f'{wkey}: Kalman Innovation Gate  (|z| > {thresh:.1f})', fontsize=13)

    ax = axes[0]
    ax.plot(times, ts['spread'], lw=0.7, color='steelblue', label='Spread')
    shade_gate(ax, times, lb | sb, color='orange', alpha=0.22, label='Gated (post-surprise)')
    for _, row in df_base.iterrows():
        c = 'green' if row['gross_usd'] > 0 else 'red'
        ax.axvline(row['entry_time'], color=c, alpha=0.5, lw=0.8)
    ax.set_ylabel('Spread pts'); ax.legend(fontsize=8); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(times, np.clip(innov, -8, 8), lw=0.5, color='darkblue',
            label='Normalized innovation (clipped ±8)')
    ax.axhline( thresh, color='red', lw=1.0, linestyle='--', label=f'+{thresh:.0f}σ')
    ax.axhline(-thresh, color='red', lw=1.0, linestyle='--', label=f'−{thresh:.0f}σ')
    shade_gate(ax, times, lb | sb, color='orange', alpha=0.22)
    ax.set_ylabel('Innovation z-score'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('UTC time')

    plt.tight_layout()
    fname = FIG_DIR / f'{wkey}_kalman_gate.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"    Figure saved → {fname.relative_to(RESULTS.parent)}")


def plot_session(wkey: str, ts: pd.DataFrame, df_base: pd.DataFrame,
                 lb: np.ndarray, sb: np.ndarray):
    times = ts.index
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'{wkey}: Session Segmentation Gate', fontsize=13)

    # Panel 1: Timeseries with active/midday shading
    ax = axes[0]
    ax.plot(times, ts['spread'], lw=0.7, color='steelblue', label='Spread')
    # Shade midday in light blue, active in white (no shade)
    shade_gate(ax, times, lb | sb, color='lightyellow', alpha=0.6, label='Midday (gated)')
    for _, row in df_base.iterrows():
        c = 'green' if row['gross_usd'] > 0 else 'red'
        ax.axvline(row['entry_time'], color=c, alpha=0.5, lw=0.8)
    ax.set_ylabel('Spread pts'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_xlabel('UTC time')

    # Panel 2: Entry hour distribution (active vs gated)
    ax     = axes[1]
    hours  = df_base['entry_hour_utc'].values
    gated  = np.array([(lb | sb)[ts.index.searchsorted(et, side='left')]
                       for et in df_base['entry_time']])
    bins   = np.arange(int(hours.min()) - 0.5, int(hours.max()) + 1.5, 0.5)

    ax.hist(hours[~gated], bins=bins, color='steelblue', alpha=0.7, label='Active (allowed)')
    ax.hist(hours[gated],  bins=bins, color='salmon',    alpha=0.7, label='Midday (blocked)')
    ax.set_xlabel('Entry hour (UTC)'); ax.set_ylabel('Trade count')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    plt.tight_layout()
    fname = FIG_DIR / f'{wkey}_session_gate.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"    Figure saved → {fname.relative_to(RESULTS.parent)}")


# ─── OFI Feasibility ─────────────────────────────────────────────────────────

def ofi_feasibility():
    print(f"\n{'='*70}")
    print("  IDEA F: ORDER FLOW IMBALANCE (OFI) — FEASIBILITY ANALYSIS")
    print(f"{'='*70}")

    # Check what raw data is available
    raw_files = sorted(DATA_DIR.glob('mbp10_*.parquet')) if DATA_DIR.exists() else []
    print(f"\n  Raw MBP-10 files on SEAGATE: {len(raw_files)}")
    for f in raw_files[:6]:
        try:
            sz = f.stat().st_size / 1e9
            print(f"    {f.name}  ({sz:.2f} GB)")
        except Exception:
            print(f"    {f.name}")

    # Try loading a small sample from first available file
    sample_loaded = False
    for f in raw_files[:1]:
        try:
            print(f"\n  Sampling {f.name} (first 10,000 rows)...")
            df_raw = pd.read_parquet(f).head(10_000)
            print(f"  Columns: {list(df_raw.columns)}")
            print(f"  Shape: {df_raw.shape}")
            print(f"  dtypes:\n{df_raw.dtypes.to_string()}")

            # Check if OFI-relevant columns exist
            ofi_cols = [c for c in df_raw.columns if any(
                x in c.lower() for x in ['bid_sz', 'ask_sz', 'side', 'size', 'volume'])]
            if ofi_cols:
                print(f"\n  OFI-relevant columns found: {ofi_cols}")
                print("  → OFI COMPUTATION IS FEASIBLE with existing raw data.")
                print("  Implementation: aggregate (bid_sz_00 − ask_sz_00) per 1s bar")
                print("  across top 5 levels to get signed volume imbalance.")
            else:
                print("\n  No bid/ask size columns found in raw files.")
                print("  → Current parquet files are 1s OHLCV bars (pre-compressed).")
                print("  → Full MBP-10 tick data would need to be re-pulled from Databento.")
            sample_loaded = True
        except Exception as e:
            print(f"  Could not load {f.name}: {e}")

    if not sample_loaded:
        print("\n  SEAGATE drive not mounted or no MBP-10 files found.")
        print("  Drive path checked:", DATA_DIR)

    print("""
  OFI Gate Implementation Plan:
  ─────────────────────────────
  Signal:  OFI_t = SUM_k=0..4 (bid_sz_k[t] - bid_sz_k[t-1])
                             - (ask_sz_k[t] - ask_sz_k[t-1])
           Aggregated to 30s rolling sum per instrument.

  Gate logic (for a SHORT entry when z > +2.5):
    Check OFI in the front contract over the past 30s.
    If OFI < −threshold (sellers driving front down → spread elevated mechanically):
        Allow short (flow-driven dislocation, will revert).
    If OFI > +threshold (buyers in front → spread pushed up by demand):
        Block short (information-driven move, may continue).

  Why this is orthogonal to z-score:
    z-score measures LEVEL of spread vs rolling mean.
    OFI measures DIRECTION of order flow pressure RIGHT NOW.
    A large negative z-score can occur with either positive or negative OFI —
    only negative OFI (sell-side pressure) creates a revertible dislocation.

  Data requirement:
    ~25-35 GB per roll window (mbp10 tick data).
    Already in your Phase 2 data pull plan.
    Compute: ~2-5 min to aggregate MBP-10 to 1s OFI bars per window.

  Expected impact (from Cont, Kukanov & Stoikov JFE 2014):
    OFI explains 65% of short-term price variance.
    Conditioning on OFI direction should substantially improve
    signal precision, particularly for the short signal which
    has been structurally weak in both W1 and W2.
  """)


# ─── Main ────────────────────────────────────────────────────────────────────

def analyze_window(wkey: str):
    pair = WINDOW_PAIRS[wkey]
    cfg  = WINDOWS[wkey]
    params_base = StrategyParams(n_lots=10, regime_gate='none')

    print(f"\n{'='*70}")
    print(f"  WINDOW {wkey}: {cfg.front} → {cfg.back}  [{cfg.roll_start}]")
    print(f"{'='*70}")

    # Load baseline (already computed; re-run if missing)
    base_path = RESULTS / pair / 'none'
    if not (base_path / 'trades.parquet').exists():
        print("  Baseline missing — running now...")
        with _quiet():
            run_window(wkey, params_base, force=True)
    df_base, ts, stats_base = load_saved(pair, 'none')

    print(f"\n  Baseline (no gate):")
    print(fmt_stats('none', stats_base))
    print(f"  RTH window: {cfg.rth_open}–{cfg.rth_close} UTC")
    print(f"  Trades: {len(df_base)}  ({len(df_base[df_base['direction']==1])} long, "
          f"{len(df_base[df_base['direction']==-1])} short)")

    spread_s = ts['spread']
    dev_s    = ts['dev']
    ts_idx   = spread_s.index

    all_stats = {'none': stats_base}

    # ── IDEA A: OU Half-life gate ─────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  IDEA A: OU HALF-LIFE GATE")

    params_hl = StrategyParams(n_lots=10, regime_gate='half_life')
    lb_hl, sb_hl = _compute_half_life_gate(dev_s, params_hl)

    hl_arr   = rolling_half_life(dev_s, params_hl.half_life_window)
    hl_finite = hl_arr[np.isfinite(hl_arr) & (hl_arr < 1e6)]
    if len(hl_finite):
        print(f"  Half-life distribution (minutes):")
        for pct_label, pct_val in [('p25', 25), ('p50', 50), ('p75', 75), ('p90', 90)]:
            print(f"    {pct_label}: {np.percentile(hl_finite, pct_val):.1f} min")
        print(f"  Gate threshold: {params_hl.half_life_max/60:.0f} min")

    preflight('half_life', lb_hl, sb_hl, df_base, ts_idx)
    plot_half_life(wkey, ts, df_base, lb_hl, sb_hl, params_hl.half_life_max / 60)

    print(f"\n  Backtest results:")
    all_stats['half_life'] = run_gate(wkey, 'half_life')

    # ── IDEA B: Kalman innovation gate ────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  IDEA B: KALMAN INNOVATION GATE")

    params_kl = StrategyParams(n_lots=10, regime_gate='kalman')
    lb_kl, sb_kl = _compute_kalman_gate(spread_s, params_kl)

    innov = kalman_innovations(spread_s, params_kl.kalman_Q, params_kl.kalman_R)
    large = np.abs(innov) > params_kl.kalman_innov_thresh
    print(f"  Large innovation events (|z|>{params_kl.kalman_innov_thresh:.0f}): "
          f"{large.sum()} bars = {100*large.mean():.1f}% of RTH")
    q95 = np.percentile(np.abs(innov), 95)
    q99 = np.percentile(np.abs(innov), 99)
    print(f"  Innovation |z| p95={q95:.2f}  p99={q99:.2f}")
    print(f"  Cooldown={params_kl.kalman_cooldown}s — gate blocks "
          f"{100*(lb_kl|sb_kl).mean():.1f}% of RTH bars")

    preflight('kalman', lb_kl, sb_kl, df_base, ts_idx)
    plot_kalman(wkey, ts, df_base, lb_kl, sb_kl, params_kl.kalman_innov_thresh)

    print(f"\n  Backtest results:")
    all_stats['kalman'] = run_gate(wkey, 'kalman')

    # ── IDEA C: Session segmentation gate ────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  IDEA C: SESSION SEGMENTATION GATE")

    params_ss = StrategyParams(n_lots=10, regime_gate='session')
    lb_ss, sb_ss = _compute_session_gate(spread_s, cfg, params_ss)

    pct_midday = 100 * (lb_ss | sb_ss).mean()
    open_h,  open_m  = map(int, cfg.rth_open.split(':'))
    close_h, close_m = map(int, cfg.rth_close.split(':'))
    rth_total = (close_h * 60 + close_m) - (open_h * 60 + open_m)
    midday_mins = rth_total - params_ss.session_open_mins - params_ss.session_close_mins
    print(f"  RTH duration: {rth_total} min  |  "
          f"Active: {params_ss.session_open_mins}+{params_ss.session_close_mins} min  |  "
          f"Midday: {midday_mins} min ({pct_midday:.1f}% of bars)")
    print(f"  Active windows (UTC): {cfg.rth_open}–", end='')
    # compute morning_end
    morning_end_min = open_h * 60 + open_m + params_ss.session_open_mins
    print(f"{morning_end_min//60:02d}:{morning_end_min%60:02d}  and  ", end='')
    aftnoon_start_min = close_h * 60 + close_m - params_ss.session_close_mins
    print(f"{aftnoon_start_min//60:02d}:{aftnoon_start_min%60:02d}–{cfg.rth_close}")

    # Trade distribution by session
    entry_hours = df_base['entry_hour_utc'].values
    morning_end_h = morning_end_min / 60
    aftnoon_start_h = aftnoon_start_min / 60
    in_morning  = entry_hours < morning_end_h
    in_aftnoon  = entry_hours >= aftnoon_start_h
    in_midday   = ~in_morning & ~in_aftnoon

    print(f"\n  Baseline trade distribution by session:")
    for mask, lbl in [(in_morning, 'Morning active'), (in_midday, 'Midday (gated)'),
                      (in_aftnoon, 'Afternoon active')]:
        sub = df_base[mask]
        if len(sub) == 0:
            print(f"    {lbl:<22}: n=0")
            continue
        wr  = 100 * (sub['gross_usd'] > 0).mean()
        avg = sub['gross_usd'].mean()
        tot = sub['gross_usd'].sum()
        print(f"    {lbl:<22}: n={len(sub):>3}  WR={wr:>5.1f}%  "
              f"avg=${avg:>7.2f}  tot=${tot:>7.1f}")

    preflight('session', lb_ss, sb_ss, df_base, ts_idx)
    plot_session(wkey, ts, df_base, lb_ss, sb_ss)

    print(f"\n  Backtest results:")
    all_stats['session'] = run_gate(wkey, 'session')

    # ── Also test half_life+kalman combination ────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  COMBINATION: HALF_LIFE + KALMAN")
    all_stats['half_life+kalman'] = run_gate(wkey, 'half_life+kalman')

    # ── Comparative summary ───────────────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  COMPARATIVE SUMMARY — {wkey}")
    print(f"  {'─'*60}")
    print(f"  {'Gate':<22} {'n':>4}  {'WR%':>6}  {'PF':>5}  "
          f"{'AvgGross':>9}  {'TotGross':>9}  {'MDD(10L)':>9}  {'t':>7}")
    print(f"  {'─'*80}")
    for g, s in all_stats.items():
        mdd = s.get('mdd_scaled', 0)
        print(f"  {g:<22} {s['n']:>4}  {s['wr']:>5.1f}%  {s['pf']:>5.2f}  "
              f"{s['avg_gross']:>9.2f}  {s['tot_gross']:>9.0f}  "
              f"{mdd:>9.0f}  {s['t_gross']:>7.3f}")

    return all_stats


def plot_comparison(all_results: dict):
    """Bar chart comparing PF and MDD across gates and windows."""
    gates   = ['none', 'half_life', 'kalman', 'session', 'half_life+kalman']
    windows = ['W1', 'W2']
    colors  = {'W1': 'steelblue', 'W2': 'tomato'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Regime Gate Comparison — W1 and W2', fontsize=13)

    x = np.arange(len(gates))
    w = 0.35

    for col, metric, ylabel in [(0, 'pf', 'Profit Factor'),
                                  (1, 'avg_gross', 'Avg Gross per Lot ($)'),
                                  (2, 'mdd_scaled', 'Max Drawdown 10L ($)')]:
        ax = axes[col]
        for j, wkey in enumerate(windows):
            vals = [all_results[wkey].get(g, {}).get(metric, 0) for g in gates]
            bars = ax.bar(x + j * w, vals, w, label=wkey, color=colors[wkey], alpha=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.2f}' if abs(val) < 100 else f'{val:.0f}',
                        ha='center', va='bottom', fontsize=7)
        ax.axhline(1.0 if col == 0 else 0, color='black', lw=0.8, linestyle='--')
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(gates, rotation=20, ha='right', fontsize=8)
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fname = FIG_DIR / 'gate_comparison.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\n  Comparison figure saved → {fname.relative_to(RESULTS.parent)}")


def main():
    print("=" * 70)
    print("  10_REGIME_GATES.PY — Regime Filter Feasibility & P&L Study")
    print("=" * 70)

    all_results = {}
    for wkey in ['W1', 'W2']:
        all_results[wkey] = analyze_window(wkey)

    # Cross-window comparison figure
    plot_comparison(all_results)

    # OFI feasibility
    ofi_feasibility()

    # ── Final recommendation ──────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY AND RECOMMENDATIONS")
    print(f"{'='*70}")
    print("""
  Gate quality is assessed on two criteria:
    (1) Pre-flight: does the gate block more losers than winners?
    (2) Backtest:   does PF improve and MDD reduce after full simulation?

  Key interpretation notes:
  • All configurations remain net-negative after transaction costs ($10.25/lot).
    The TC hurdle requires avg_gross > $10.25/lot to break even — none reach this.
  • Focus is on PF direction (trending toward profitability?) and MDD reduction
    (risk management value), not absolute net P&L at this research stage.
  • Statistical tests (t, p) are unreliable at n=24–45. Use as directional signals only.

  See results/gate_research/ for per-gate visualizations.
  """)


if __name__ == '__main__':
    main()
