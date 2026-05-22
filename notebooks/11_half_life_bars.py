#!/usr/bin/env python3
"""
11_half_life_bars.py — Half-life gate sensitivity to bar resolution.

Tests the OU half-life gate at 1-min, 2-min, 5-min, 10-min bar resolutions
against the 1s baseline. At 1s, β ≈ 0 almost everywhere (noise-dominated),
so the gate is nearly inert. At coarser resolutions the AR(1) should capture
meaningful regime variation between fast-reverting and slow-drift periods.

For each resolution:
  - Half-life distribution stats
  - Gate activity rate (% of RTH bars gated)
  - Pre-flight: which baseline trades are blocked (winners vs losers)?
  - Inline backtest (no disk I/O overhead)

Also: threshold sweep at the best resolution.

Run: python notebooks/11_half_life_bars.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from strategy import (
    WINDOWS, StrategyParams,
    load_sofr, load_dt_years, load_volume_gate,
    load_rth_bars, compute_z, build_entry_mask, simulate, compute_stats,
    _compute_half_life_gate,
)

DATA_DIR = Path('/Volumes/SEAGATE/Databento_Futures')
RESULTS  = Path(__file__).parent.parent / 'results'
FIG_DIR  = RESULTS / 'gate_research'
FIG_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_PAIRS = {'W1': 'ESU4_ESZ4_20240912', 'W2': 'ESZ4_ESH5_20241212'}

# Bar resolutions to test (label, pandas freq, seconds-per-bar, n_bars window)
BAR_CONFIGS = [
    ('1s',   '1s',    1,   1800),   # 30-min window
    ('1min', '1min',  60,  30),     # 30-bar = 30 min
    ('2min', '2min',  120, 30),     # 30-bar = 60 min
    ('5min', '5min',  300, 30),     # 30-bar = 150 min
    ('10min','10min', 600, 30),     # 30-bar = 300 min
]

# ─── Data loading ─────────────────────────────────────────────────────────────

def load_window_data(wkey: str, cache: dict) -> tuple:
    if wkey in cache:
        return cache[wkey]
    cfg = WINDOWS[wkey]
    params_base = StrategyParams()
    print(f"  Loading data for {wkey}...", end=' ', flush=True)
    sofr_utc = load_sofr(DATA_DIR)
    dt_yr    = load_dt_years(cfg, DATA_DIR)
    spread, fv, dev = load_rth_bars(cfg, params_base, sofr_utc, dt_yr, DATA_DIR)
    vol_gate  = load_volume_gate(cfg, params_base, DATA_DIR)
    cache[wkey] = (spread, fv, dev, vol_gate, cfg)
    print(f"{len(spread):,} 1s bars")
    return cache[wkey]


def run_inline(wkey: str, params: StrategyParams, cache: dict) -> dict:
    """Run backtest using cached data; return stats dict."""
    spread, fv, dev, vol_gate, cfg = load_window_data(wkey, cache)
    z     = compute_z(dev, params)
    emask = build_entry_mask(spread, vol_gate, cfg, params)
    df    = simulate(spread, dev, z, emask, cfg, params)
    if df.empty:
        return {'n': 0, 'wr': 0, 'pf': 0, 'avg_gross': 0, 'tot_gross': 0,
                'mdd_scaled': 0, 't_gross': 0, 'p_gross': 1}
    return compute_stats(df, params)

# ─── Analysis helpers ──────────────────────────────────────────────────────────

def hl_series_minutes(dev: pd.Series, bar_res: str, n_bars: int,
                       bar_secs: int) -> np.ndarray:
    """Return rolling OU half-life in MINUTES for analysis/plotting."""
    if bar_res == '1s':
        s_work   = pd.Series(dev.values.astype(float), index=dev.index)
        win_bars = 1800
    else:
        s_work   = dev.resample(bar_res).last().dropna().astype(float)
        win_bars = n_bars

    s_lag = s_work.shift(1)
    mp    = max(win_bars // 2, 5)
    rc    = ((s_work * s_lag).rolling(win_bars, min_periods=mp).mean()
             - s_work.rolling(win_bars, min_periods=mp).mean()
             * s_lag.rolling(win_bars, min_periods=mp).mean())
    rv    = ((s_lag ** 2).rolling(win_bars, min_periods=mp).mean()
             - s_lag.rolling(win_bars, min_periods=mp).mean() ** 2)
    beta  = (rc / rv).clip(lower=1e-9, upper=1 - 1e-9)
    hl_m  = (np.log(2) / (-np.log(beta.abs()))).fillna(np.inf) * bar_secs / 60
    return hl_m.values


def preflight(lb, sb, df_base, ts_idx):
    """Return stats dict for blocked vs unblocked baseline trades."""
    blocked = []
    for _, row in df_base.iterrows():
        et  = row['entry_time']
        pos = ts_idx.searchsorted(et, side='left')
        pos = min(pos, len(lb) - 1)
        blocked.append(bool(lb[pos] if row['direction'] == 1 else sb[pos]))
    df = df_base.copy()
    df['blocked'] = blocked
    blk   = df[df['blocked']]
    unblk = df[~df['blocked']]
    return {
        'n_blocked':   int(df['blocked'].sum()),
        'pct_blocked': 100 * df['blocked'].mean(),
        'blk_wr':   100 * (blk['gross_usd'] > 0).mean()  if len(blk) else np.nan,
        'blk_avg':  blk['gross_usd'].mean()               if len(blk) else 0,
        'unblk_wr': 100 * (unblk['gross_usd'] > 0).mean() if len(unblk) else np.nan,
        'unblk_avg': unblk['gross_usd'].mean()             if len(unblk) else 0,
    }

# ─── Main analysis ────────────────────────────────────────────────────────────

def analyze_window(wkey: str, cache: dict, hl_threshold: float = 1200.0):
    pair  = WINDOW_PAIRS[wkey]
    cfg   = WINDOWS[wkey]
    spread, fv, dev, vol_gate, _ = load_window_data(wkey, cache)
    ts_idx = dev.index

    print(f"\n{'='*72}")
    print(f"  WINDOW {wkey}: {cfg.front} → {cfg.back}  [threshold={hl_threshold/60:.0f} min]")
    print(f"{'='*72}")

    # Ungated (no gate)
    params_base = StrategyParams(n_lots=10, regime_gate='none')
    stats_base  = run_inline(wkey, params_base, cache)
    df_base = None
    # Rerun to get trade-level df
    z_b   = compute_z(dev, params_base)
    em_b  = build_entry_mask(spread, vol_gate, cfg, params_base)
    df_base = simulate(spread, dev, z_b, em_b, cfg, params_base)

    print(f"\n  {'Bar res':<8} {'n_bars/win':>10}  "
          f"{'HL p50 (min)':>13}  {'HL p90 (min)':>13}  "
          f"{'Gated%':>7}  {'Blk':>4}  {'BlkWR':>6}  {'BlkAvg':>8}  "
          f"{'n':>4}  {'WR%':>5}  {'PF':>5}  {'AvgGross':>9}  {'MDD(10L)':>9}")
    print(f"  {'─'*120}")

    all_stats = {}
    hl_arrays = {}  # for plotting

    for (label, freq, bar_secs, n_bars) in BAR_CONFIGS:
        params_hl = StrategyParams(
            n_lots=10, regime_gate='half_life',
            half_life_bar_res=label,
            half_life_n_bars=n_bars,
            half_life_max=hl_threshold,
        )
        lb, sb = _compute_half_life_gate(dev, params_hl)

        # HL distribution
        hl_m = hl_series_minutes(dev, label if label != '1s' else '1s',
                                  n_bars, bar_secs)
        # Upsample hl_m to 1s if needed (for consistent stats)
        if label != '1s':
            hl_ser = dev.resample(freq).last().dropna()
            hl_m_1s = pd.Series(hl_m, index=hl_ser.index).reindex(
                dev.index, method='ffill').fillna(np.inf).values
        else:
            hl_m_1s = hl_m

        hl_finite = hl_m_1s[np.isfinite(hl_m_1s) & (hl_m_1s < 1e6)]
        hl_p50 = np.median(hl_finite) if len(hl_finite) else np.inf
        hl_p90 = np.percentile(hl_finite, 90) if len(hl_finite) else np.inf
        hl_arrays[label] = (hl_m, freq, bar_secs)

        gate_pct = 100 * (lb | sb).mean()

        pf_info  = preflight(lb, sb, df_base, ts_idx)
        stats_hl = run_inline(wkey, params_hl, cache)

        n_bwin   = f"w={n_bars if label!='1s' else '1800'}"
        blk_wr_s = f"{pf_info['blk_wr']:.0f}%" if not np.isnan(pf_info['blk_wr']) else "  —"
        mdd      = stats_hl.get('mdd_scaled', 0)

        print(f"  {label:<8} {n_bwin:>10}  "
              f"{hl_p50 if hl_p50<999 else '>999':>13.1f}  "
              f"{hl_p90 if hl_p90<999 else '>999':>13.1f}  "
              f"{gate_pct:>7.1f}%  "
              f"{pf_info['n_blocked']:>4}  "
              f"{blk_wr_s:>6}  "
              f"{pf_info['blk_avg']:>8.2f}  "
              f"{stats_hl['n']:>4}  "
              f"{stats_hl['wr']:>5.1f}%  "
              f"{stats_hl['pf']:>5.2f}  "
              f"{stats_hl['avg_gross']:>9.2f}  "
              f"{mdd:>9.0f}")

        all_stats[label] = stats_hl

    # ── Show baseline for comparison ──────────────────────────────────────────
    mdd_b = stats_base.get('mdd_scaled', 0)
    print(f"  {'─'*120}")
    print(f"  {'BASELINE':<8} {'(no gate)':>10}  "
          f"{'—':>13}  {'—':>13}  "
          f"{'0.0%':>7}  {'0':>4}  {'—':>6}  {'—':>8}  "
          f"{stats_base['n']:>4}  {stats_base['wr']:>5.1f}%  {stats_base['pf']:>5.2f}  "
          f"{stats_base['avg_gross']:>9.2f}  {mdd_b:>9.0f}")

    # ── Threshold sweep at 5-min resolution ───────────────────────────────────
    print(f"\n  Threshold sweep at 5-min resolution:")
    print(f"\n  {'Threshold':>12}  {'Gated%':>7}  {'Blk':>4}  {'BlkWR':>6}  "
          f"{'n':>4}  {'WR%':>5}  {'PF':>5}  {'AvgGross':>9}  {'MDD(10L)':>9}")
    print(f"  {'─'*85}")

    thresholds = [300, 600, 900, 1200, 1800, 3600]  # 5, 10, 15, 20, 30, 60 min

    for t in thresholds:
        p = StrategyParams(n_lots=10, regime_gate='half_life',
                           half_life_bar_res='5min', half_life_n_bars=30,
                           half_life_max=float(t))
        lb_t, sb_t = _compute_half_life_gate(dev, p)
        gate_pct   = 100 * (lb_t | sb_t).mean()
        pf_t       = preflight(lb_t, sb_t, df_base, ts_idx)
        s_t        = run_inline(wkey, p, cache)
        blk_wr_s   = f"{pf_t['blk_wr']:.0f}%" if not np.isnan(pf_t['blk_wr']) else "  —"
        mdd_t      = s_t.get('mdd_scaled', 0)
        print(f"  {t/60:>9.0f} min  "
              f"{gate_pct:>7.1f}%  "
              f"{pf_t['n_blocked']:>4}  "
              f"{blk_wr_s:>6}  "
              f"{s_t['n']:>4}  "
              f"{s_t['wr']:>5.1f}%  "
              f"{s_t['pf']:>5.2f}  "
              f"{s_t['avg_gross']:>9.2f}  "
              f"{mdd_t:>9.0f}")

    # ── Figure: HL distribution across bar resolutions ────────────────────────
    _plot_hl_distributions(wkey, dev, hl_arrays, hl_threshold)

    # ── Figure: timeseries with 5-min gate overlaid ───────────────────────────
    _plot_gate_overlay(wkey, dev, spread, df_base, hl_threshold)

    return all_stats


def _plot_hl_distributions(wkey: str, dev: pd.Series, hl_arrays: dict,
                            threshold: float):
    """Box-whisker / violin of HL distributions for each bar resolution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'{wkey}: Half-life Distribution by Bar Resolution', fontsize=13)

    labels, p50s, p75s, p90s, p99s, gate_pcts = [], [], [], [], [], []
    for label, (hl_m, freq, bar_secs) in hl_arrays.items():
        if label == '1s':
            hl_1s = hl_m
        else:
            hl_ser = dev.resample(freq).last().dropna()
            hl_1s  = pd.Series(hl_m, index=hl_ser.index).reindex(
                dev.index, method='ffill').fillna(np.inf).values
        finite = hl_1s[np.isfinite(hl_1s) & (hl_1s < 500)]
        if len(finite) == 0:
            continue
        labels.append(label)
        p50s.append(np.median(finite))
        p75s.append(np.percentile(finite, 75))
        p90s.append(np.percentile(finite, 90))
        p99s.append(np.percentile(finite, 99))
        gate_pcts.append(100 * np.mean(hl_1s > threshold / 60))  # threshold in min

    x = np.arange(len(labels))
    ax = axes[0]
    ax.bar(x - 0.3, p50s, 0.2, label='p50', color='steelblue')
    ax.bar(x - 0.1, p75s, 0.2, label='p75', color='royalblue')
    ax.bar(x + 0.1, p90s, 0.2, label='p90', color='navy')
    ax.bar(x + 0.3, p99s, 0.2, label='p99', color='black', alpha=0.6)
    ax.axhline(threshold / 60, color='red', lw=1.2, linestyle='--',
               label=f'Threshold {threshold/60:.0f} min')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('Half-life (minutes, clipped 500)')
    ax.set_title('HL percentiles by bar resolution')
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(x, gate_pcts, color='salmon', alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('% of RTH bars gated')
    ax.set_title(f'Gate activity rate (HL > {threshold/60:.0f} min)')
    ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    fname = FIG_DIR / f'{wkey}_hl_distributions.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\n  Figure saved → {fname.relative_to(RESULTS.parent)}")


def _plot_gate_overlay(wkey: str, dev: pd.Series, spread: pd.Series,
                       df_base: pd.DataFrame, threshold: float):
    """Timeseries with 5-min half-life gate shaded."""
    p5 = StrategyParams(n_lots=10, regime_gate='half_life',
                        half_life_bar_res='5min', half_life_n_bars=30,
                        half_life_max=threshold)
    p10 = StrategyParams(n_lots=10, regime_gate='half_life',
                         half_life_bar_res='10min', half_life_n_bars=30,
                         half_life_max=threshold)
    lb5, sb5   = _compute_half_life_gate(dev, p5)
    lb10, sb10 = _compute_half_life_gate(dev, p10)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f'{wkey}: Spread + Half-life Gate (5-min and 10-min)', fontsize=13)
    times = spread.index

    def shade(ax, blocked, color, alpha, label=None):
        in_b = False; t0 = None; labeled = False
        for t, b in zip(times, blocked):
            if b and not in_b:
                t0 = t; in_b = True
            elif not b and in_b:
                lbl = label if not labeled else None
                ax.axvspan(t0, t, color=color, alpha=alpha, label=lbl)
                labeled = True; in_b = False
        if in_b and t0 is not None:
            ax.axvspan(t0, times[-1], color=color, alpha=alpha)

    for ax, title, lb, sb, color in [
            (axes[0], 'Spread (no gate)', np.zeros(len(spread),bool), np.zeros(len(spread),bool), None),
            (axes[1], '5-min HL gate', lb5, sb5, 'salmon'),
            (axes[2], '10-min HL gate', lb10, sb10, 'orange'),
    ]:
        ax.plot(times, spread.values, lw=0.7, color='steelblue')
        if color:
            shade(ax, lb | sb, color, 0.3, 'Gated (slow HL)')
        for _, row in df_base.iterrows():
            c = 'green' if row['gross_usd'] > 0 else 'red'
            ax.axvline(row['entry_time'], color=c, alpha=0.5, lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Spread pts')
        if color:
            ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    axes[2].set_xlabel('UTC time')
    plt.tight_layout()
    fname = FIG_DIR / f'{wkey}_hl_gate_overlay.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"  Figure saved → {fname.relative_to(RESULTS.parent)}")


# ─── Cross-window comparison ──────────────────────────────────────────────────

def plot_cross_window(results: dict):
    """PF and MDD comparison across bar resolutions for both windows."""
    bar_labels = [c[0] for c in BAR_CONFIGS]
    windows    = ['W1', 'W2']
    colors     = {'W1': 'steelblue', 'W2': 'tomato'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Half-life Gate: Bar Resolution Sweep — W1 vs W2', fontsize=13)

    x = np.arange(len(bar_labels))
    w = 0.35

    for col, metric, ylabel in [
        (0, 'pf',       'Profit Factor'),
        (1, 'avg_gross','Avg Gross/lot ($)'),
        (2, 'mdd_scaled','MDD 10 lots ($)'),
    ]:
        ax = axes[col]
        for j, wk in enumerate(windows):
            vals = [results[wk].get(b, {}).get(metric, 0) for b in bar_labels]
            bars = ax.bar(x + j * w, vals, w, label=wk, color=colors[wk], alpha=0.8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.01 if col == 0 else 0),
                        f'{val:.2f}' if col <= 1 else f'{val:.0f}',
                        ha='center', va='bottom', fontsize=7)
        ax.axhline(1.0 if col == 0 else 0, color='black', lw=0.8, linestyle='--')
        ax.set_xticks(x + w / 2)
        ax.set_xticklabels(bar_labels, fontsize=9)
        ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y')
        ax.set_xlabel('Bar resolution')

    plt.tight_layout()
    fname = FIG_DIR / 'hl_bar_resolution_sweep.png'
    plt.savefig(fname, dpi=120)
    plt.close()
    print(f"\n  Cross-window comparison → {fname.relative_to(RESULTS.parent)}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print('=' * 72)
    print('  11_HALF_LIFE_BARS.PY — OU Half-life Gate: Bar Resolution Sweep')
    print('=' * 72)

    cache   = {}
    results = {}

    for wkey in ['W1', 'W2']:
        results[wkey] = analyze_window(wkey, cache)

    plot_cross_window(results)

    print(f"\n{'='*72}")
    print('  CROSS-WINDOW SUMMARY')
    print(f"{'='*72}")
    print(f"\n  {'Bar res':<8}  "
          f"{'W1 PF':>7}  {'W1 AvgG':>9}  {'W1 MDD':>9}  "
          f"{'W2 PF':>7}  {'W2 AvgG':>9}  {'W2 MDD':>9}")
    print(f"  {'─'*75}")
    for label, *_ in BAR_CONFIGS:
        s1 = results['W1'].get(label, {})
        s2 = results['W2'].get(label, {})
        print(f"  {label:<8}  "
              f"{s1.get('pf',0):>7.2f}  {s1.get('avg_gross',0):>9.2f}  "
              f"{s1.get('mdd_scaled',0):>9.0f}  "
              f"{s2.get('pf',0):>7.2f}  {s2.get('avg_gross',0):>9.2f}  "
              f"{s2.get('mdd_scaled',0):>9.0f}")

    print(f"""
  Interpretation guide:
  ─────────────────────
  At 1s: AR(1) β ≈ 0 (noise-dominated) → half-life ≈ 0 min everywhere → gate barely triggers.
  At coarser bars: β captures genuine multi-bar persistence → gate separates trending vs
    reverting regimes with meaningful frequency.

  Key metrics to evaluate a gate resolution:
    (1) HL p50/p90: are these above the threshold? If median HL < threshold, gate is
        mostly inactive (bars spend most time in fast-reversion regime = allow entries).
    (2) Gated%: what fraction of RTH is gated? 10-40% is informative; < 5% is inert;
        > 60% is too aggressive.
    (3) BlkWR: win-rate of blocked baseline trades. If BlkWR < 50%, the gate correctly
        removes bad trades. If BlkWR > 70%, the gate is removing good trades.
    (4) PF: post-gate profit factor. Should improve vs baseline.

  Figures saved to results/gate_research/
  """)


if __name__ == '__main__':
    main()
