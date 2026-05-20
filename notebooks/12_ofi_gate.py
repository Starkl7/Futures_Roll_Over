#!/usr/bin/env python3
"""
12_ofi_gate.py — Order Flow Imbalance (OFI) Gate

Hypothesis: Persistent LOB imbalance on the calendar spread signals directional
pressure that prevents mean-reversion from paying off. Gate entries when the
rolling spread OFI exceeds a threshold in the direction of the current deviation.

OFI definition (LOB imbalance):
    imb(t) = (Σ_k bid_sz_k - Σ_k ask_sz_k) / (Σ_k bid_sz_k + Σ_k ask_sz_k)
    spread_ofi(t) = imb_front(t) - imb_back(t)  ∈ [-2, +2]

Gate rule:
    block longs  when rolling_ofi(t) < -threshold   (net selling on spread)
    block shorts when rolling_ofi(t) > +threshold   (net buying  on spread)

Windows tested: 1, 5, 10, 30 minutes
Thresholds:     0.05, 0.10, 0.15, 0.20
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from strategy import (
    WINDOWS, StrategyParams,
    load_sofr, load_dt_years, load_volume_gate,
    load_rth_bars, compute_z, build_entry_mask, simulate, compute_stats,
)

DATA_DIR  = Path('/Volumes/SEAGATE/Databento_Futures')
RESULTS   = Path(__file__).parent.parent / 'results'
CACHE_DIR = RESULTS / 'ofi_cache'
FIG_DIR   = RESULTS / 'gate_research'
CACHE_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

BID_SZ = [f'bid_sz_{k:02d}' for k in range(10)]
ASK_SZ = [f'ask_sz_{k:02d}' for k in range(10)]

# ─── OFI cache ───────────────────────────────────────────────────────────────

def build_ofi_cache(cfg) -> pd.Series:
    front, back = cfg.front, cfg.back
    roll_pfx    = cfg.roll_start[:10]   # 'YYYY-MM-DD' as in filenames
    files       = sorted(DATA_DIR.glob(f'mbp10_{front}_{back}_{roll_pfx}_*.parquet'))
    needed      = ['symbol'] + BID_SZ + ASK_SZ

    parts = []
    for f in files:
        print(f'  {f.name}...', end='', flush=True)
        raw = pd.read_parquet(f, columns=needed)
        day_imb = {}
        for sym, label in [(front, 'front'), (back, 'back')]:
            sub     = raw[raw['symbol'] == sym]
            bid_tot = sub[BID_SZ].sum(axis=1).astype(float)
            ask_tot = sub[ASK_SZ].sum(axis=1).astype(float)
            total   = bid_tot + ask_tot
            imb     = (bid_tot - ask_tot) / total.where(total > 0, np.nan)
            day_imb[label] = imb.resample('1s').last()
        df_day        = pd.DataFrame(day_imb).ffill()
        df_day['ofi'] = df_day['front'] - df_day['back']
        rth           = df_day['ofi'].between_time(cfg.rth_open, cfg.rth_close)
        parts.append(rth)
        print(f' {len(rth):,} RTH 1s bars')

    ofi = pd.concat(parts).sort_index()
    ofi.to_frame('ofi').to_parquet(CACHE_DIR / f'{front}_{back}_ofi_1s.parquet')
    print(f'  → cached {len(ofi):,} bars')
    return ofi


def load_ofi(cfg) -> pd.Series:
    path = CACHE_DIR / f'{cfg.front}_{cfg.back}_ofi_1s.parquet'
    if path.exists():
        return pd.read_parquet(path)['ofi']
    print(f'  Cache miss — building from MBP-10...')
    return build_ofi_cache(cfg)


# ─── Window data cache (avoid re-reading from SEAGATE) ───────────────────────

_data_cache: dict = {}

def load_window_data(wkey: str):
    if wkey in _data_cache:
        return _data_cache[wkey]
    cfg    = WINDOWS[wkey]
    sofr   = load_sofr(DATA_DIR)
    dt_yr  = load_dt_years(cfg, DATA_DIR)
    p      = StrategyParams()
    spread, fv, dev = load_rth_bars(cfg, p, sofr, dt_yr, DATA_DIR)
    vgate  = load_volume_gate(cfg, p, DATA_DIR)
    _data_cache[wkey] = (cfg, spread, fv, dev, vgate)
    return _data_cache[wkey]


# ─── Inline backtest ─────────────────────────────────────────────────────────

def run_ofi(wkey: str, ofi_1s: pd.Series, window_min: int, threshold: float) -> dict:
    cfg, spread, fv, dev, vgate = load_window_data(wkey)
    params = StrategyParams(n_lots=10, regime_gate='ofi',
                            ofi_window_min=window_min, ofi_threshold=threshold)
    z     = compute_z(dev, params)
    emask = build_entry_mask(spread, vgate, cfg, params)
    df    = simulate(spread, dev, z, emask, cfg, params)
    if df.empty:
        return {}
    s = compute_stats(df, params)
    # gate activity
    ofi_aligned = ofi_1s.reindex(spread.index, method='ffill').fillna(0.0)
    rolling     = ofi_aligned.rolling(window_min * 60, min_periods=1).mean()
    gated       = ((rolling < -threshold) | (rolling > threshold))
    gated_pct   = gated.mean() * 100
    s['gated_pct'] = gated_pct
    return s


def run_baseline(wkey: str) -> tuple[dict, pd.DataFrame]:
    cfg, spread, fv, dev, vgate = load_window_data(wkey)
    params = StrategyParams(n_lots=10)
    z     = compute_z(dev, params)
    emask = build_entry_mask(spread, vgate, cfg, params)
    df    = simulate(spread, dev, z, emask, cfg, params)
    s     = compute_stats(df, params)
    return s, df


# ─── Pre-flight ──────────────────────────────────────────────────────────────

def preflight(wkey: str, ofi_1s: pd.Series, window_min: int, threshold: float):
    cfg, spread, fv, dev, vgate = load_window_data(wkey)
    _, base_df = run_baseline(wkey)
    if base_df.empty:
        return

    ofi_aligned = ofi_1s.reindex(spread.index, method='ffill').fillna(0.0)
    rolling     = ofi_aligned.rolling(window_min * 60, min_periods=1).mean()

    blocked, total, winners = [], 0, 0
    for _, row in base_df.iterrows():
        entry_t = row['entry_time']
        if entry_t not in rolling.index:
            loc = rolling.index.searchsorted(entry_t)
            if loc >= len(rolling):
                continue
            ofi_val = rolling.iloc[loc]
        else:
            ofi_val = rolling.loc[entry_t]

        is_long  = row['direction'] == 1
        is_short = row['direction'] == -1
        blk = (is_long and ofi_val < -threshold) or (is_short and ofi_val > threshold)
        if blk:
            blocked.append({'time': entry_t, 'dir': 'L' if is_long else 'S',
                            'ofi': ofi_val, 'gross': row['gross_usd']})
            if row['gross_usd'] > 0:
                winners += 1
        total += 1

    print(f'\n  Pre-flight ({wkey}, win={window_min}min, thr={threshold}):')
    print(f'  Baseline n={total}  |  Blocked {len(blocked)}/{total}  '
          f'(WR={winners/len(blocked)*100:.0f}%  '
          f'AvgG=${sum(r["gross"] for r in blocked)/max(len(blocked),1):.2f})')
    for r in blocked:
        print(f'    {r["time"].strftime("%m-%d %H:%M")}  {r["dir"]}  '
              f'OFI={r["ofi"]:+.3f}  gross=${r["gross"]:.2f}')


# ─── Sweep ───────────────────────────────────────────────────────────────────

WINDOWS_MINS = [1, 5, 10, 30]
THRESHOLDS   = [0.05, 0.10, 0.15, 0.20]

def run_sweep(wkey: str, ofi_1s: pd.Series) -> pd.DataFrame:
    rows = []
    for win in WINDOWS_MINS:
        for thr in THRESHOLDS:
            s = run_ofi(wkey, ofi_1s, win, thr)
            if not s:
                continue
            rows.append({
                'window_min': win,
                'threshold':  thr,
                'gated_pct':  s.get('gated_pct', np.nan),
                'n':          s['n'],
                'wr':         s['wr'],
                'pf':         s['pf'],
                'avg_gross':  s['avg_gross'],
                'mdd':        s.get('mdd_scaled', np.nan),
            })
    return pd.DataFrame(rows)


# ─── Figures ─────────────────────────────────────────────────────────────────

def plot_ofi_distribution(wkey: str, ofi_1s: pd.Series):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{wkey} — Spread OFI Distribution (1s RTH)')

    # histogram
    ax = axes[0]
    vals = ofi_1s.dropna()
    ax.hist(vals, bins=100, color='steelblue', alpha=0.8)
    for thr in [0.10, 0.20]:
        ax.axvline( thr, color='red',   lw=1.2, ls='--', label=f'±{thr}')
        ax.axvline(-thr, color='green', lw=1.2, ls='--')
    ax.set_xlabel('Spread OFI')
    ax.set_ylabel('Count')
    ax.set_title('OFI Histogram')
    ax.legend(fontsize=8)

    # rolling means
    ax = axes[1]
    for win, col in [(1, '#1f77b4'), (5, '#ff7f0e'), (10, '#2ca02c'), (30, '#d62728')]:
        r = ofi_1s.rolling(win * 60, min_periods=1).mean()
        ax.plot(ofi_1s.index, r, lw=0.6, alpha=0.8, color=col, label=f'{win}min')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title('Rolling OFI (sample day)')
    ax.set_xlabel('Time (UTC)')
    ax.set_ylabel('Rolling mean OFI')
    ax.legend(fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))

    fig.tight_layout()
    out = FIG_DIR / f'{wkey}_ofi_distribution.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Figure → {out}')


def plot_sweep_heatmap(wkey: str, sweep: pd.DataFrame, baseline_pf: float):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'{wkey} OFI Gate Sweep  (baseline PF={baseline_pf:.2f})')

    metrics = [('pf', 'Profit Factor', plt.cm.RdYlGn),
               ('avg_gross', 'Avg Gross ($/lot)', plt.cm.RdYlGn),
               ('gated_pct', 'Gated % of RTH', plt.cm.Blues)]

    for ax, (col, title, cmap) in zip(axes, metrics):
        mat = sweep.pivot(index='window_min', columns='threshold', values=col)
        im  = ax.imshow(mat.values, cmap=cmap, aspect='auto')
        ax.set_xticks(range(len(mat.columns)))
        ax.set_xticklabels([f'{v:.2f}' for v in mat.columns], fontsize=9)
        ax.set_yticks(range(len(mat.index)))
        ax.set_yticklabels([f'{v}min' for v in mat.index], fontsize=9)
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Window')
        ax.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            fontsize=8, color='black')
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out = FIG_DIR / f'{wkey}_ofi_sweep.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Figure → {out}')


def plot_cross_window(w1_sweep: pd.DataFrame, w2_sweep: pd.DataFrame,
                      b1: dict, b2: dict):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('OFI Gate Cross-Window Comparison', fontsize=13)

    for row_idx, (sweep, wkey, base) in enumerate(
            [(w1_sweep, 'W1', b1), (w2_sweep, 'W2', b2)]):
        for col_idx, metric in enumerate(['pf', 'avg_gross']):
            ax = axes[row_idx][col_idx]
            baseline_val = base['pf'] if metric == 'pf' else base['avg_gross']
            ax.axhline(baseline_val, color='k', lw=1.5, ls='--', label='Baseline')
            for win in WINDOWS_MINS:
                sub = sweep[sweep['window_min'] == win].sort_values('threshold')
                if sub.empty:
                    continue
                ax.plot(sub['threshold'], sub[metric], marker='o', ms=5,
                        label=f'{win}min')
            ax.set_title(f'{wkey} — {metric}')
            ax.set_xlabel('Threshold')
            ax.set_ylabel(metric)
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / 'ofi_cross_window.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'  Figure → {out}')


# ─── Main ────────────────────────────────────────────────────────────────────

def print_sweep_table(wkey: str, sweep: pd.DataFrame, base: dict):
    header = f'{"Win":>6}  {"Thr":>5}  {"Gated%":>7}  {"n":>4}  ' \
             f'{"WR%":>6}  {"PF":>5}  {"AvgGross":>9}  {"MDD(10L)":>9}'
    sep = '─' * len(header)
    print(f'\n  {wkey} OFI Gate Sweep:')
    print(f'  {header}')
    print(f'  {sep}')
    for _, r in sweep.iterrows():
        print(f'  {int(r.window_min):>4}min  {r.threshold:>5.2f}  '
              f'{r.gated_pct:>6.1f}%  {int(r.n):>4}  '
              f'{r.wr:>5.1f}%  {r.pf:>5.2f}  '
              f'${r.avg_gross:>8.2f}  ${r.mdd:>8.0f}')
    print(f'  {sep}')
    print(f'  BASELINE:   {"—":>7}  {base["n"]:>4}  '
          f'{base["wr"]:>5.1f}%  {base["pf"]:>5.2f}  '
          f'${base["avg_gross"]:>8.2f}  ${base.get("mdd_scaled", 0):>8.0f}')


def main():
    print('=' * 72)
    print('  12_OFI_GATE.PY — LOB Imbalance Regime Gate')
    print('=' * 72)

    baselines = {}
    ofi_data  = {}
    sweeps    = {}

    for wkey in ['W1', 'W2']:
        cfg = WINDOWS[wkey]
        print(f'\nLoading data for {wkey}...')
        load_window_data(wkey)

        print(f'Loading/building OFI cache for {wkey}...')
        ofi = load_ofi(cfg)
        ofi_data[wkey] = ofi
        print(f'  OFI series: {len(ofi):,} 1s bars  '
              f'({ofi.index[0].date()} → {ofi.index[-1].date()})')
        print(f'  OFI stats: mean={ofi.mean():.4f}  std={ofi.std():.4f}  '
              f'p10={ofi.quantile(0.1):.3f}  p90={ofi.quantile(0.9):.3f}')

        # Distribution figures
        plot_ofi_distribution(wkey, ofi)

        # Baseline
        base, _ = run_baseline(wkey)
        baselines[wkey] = base
        print(f'  Baseline: n={base["n"]}  WR={base["wr"]:.1f}%  '
              f'PF={base["pf"]:.2f}  AvgG=${base["avg_gross"]:.2f}')

        # Sweep
        print(f'\nRunning OFI gate sweep for {wkey} '
              f'({len(WINDOWS_MINS)}×{len(THRESHOLDS)} = '
              f'{len(WINDOWS_MINS)*len(THRESHOLDS)} combos)...')
        sweep = run_sweep(wkey, ofi)
        sweeps[wkey] = sweep
        print_sweep_table(wkey, sweep, base)

        # Heatmap
        plot_sweep_heatmap(wkey, sweep, base['pf'])

        # Pre-flight at best config (highest PF)
        best = sweep.loc[sweep['pf'].idxmax()]
        preflight(wkey, ofi, int(best['window_min']), best['threshold'])

    # Cross-window comparison
    print('\nCross-window comparison:')
    plot_cross_window(sweeps['W1'], sweeps['W2'], baselines['W1'], baselines['W2'])

    # Combined summary
    print('\n' + '=' * 72)
    print('  CROSS-WINDOW SUMMARY (best PF per window)')
    print('=' * 72)
    for wkey in ['W1', 'W2']:
        sweep = sweeps[wkey]
        best  = sweep.loc[sweep['pf'].idxmax()]
        base  = baselines[wkey]
        print(f'\n  {wkey}  baseline → PF={base["pf"]:.2f}  '
              f'AvgG=${base["avg_gross"]:.2f}')
        print(f'  Best OFI gate: {int(best["window_min"])}min / '
              f'thr={best["threshold"]:.2f} → '
              f'n={int(best["n"])}  PF={best["pf"]:.2f}  '
              f'AvgG=${best["avg_gross"]:.2f}  '
              f'MDD=${best["mdd"]:.0f} (10L)  '
              f'Gated={best["gated_pct"]:.1f}%')


if __name__ == '__main__':
    main()
