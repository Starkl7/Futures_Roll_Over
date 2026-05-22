"""Generate all 36 Supplementary_notebooks/*.ipynb files."""
import json
from pathlib import Path

HERE = Path(__file__).parent


def code_cell(src):
    return {"cell_type": "code", "execution_count": None,
            "metadata": {}, "outputs": [],
            "source": src if isinstance(src, list) else [src]}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src if isinstance(src, list) else [src]}


def nb(title, cells):
    return {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": [md_cell(f"# {title}")] + cells,
    }


def write_nb(name, title, cells):
    path = HERE / f"{name}.ipynb"
    with open(path, "w") as f:
        json.dump(nb(title, cells), f, indent=1)
    print(f"  wrote {name}.ipynb")


IMPORTS = """\
import sys, gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
sys.path.insert(0, '.')
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / 'src'))
from config import (RESULTS_DIR, SEAGATE_DIR, FIGS_DIR, WINDOWS_META,
                    BASELINE_STATS, UPDATED_STATS, WIN_COLORS,
                    TICK, MULT, save_fig)
Path('figures').mkdir(exist_ok=True)
"""

# ── A1 ─────────────────────────────────────────────────────────────────────
write_nb("A1_equity_curves_per_window", "A1 — Equity Curves per Window", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    trades = trades.sort_values('entry_time').reset_index(drop=True)
    trades['cum_gross'] = trades['gross_usd'].cumsum()
    ax.step(range(len(trades)), trades['cum_gross'], where='post',
            color=WIN_COLORS[wk], linewidth=1.5, label='Baseline')
    if 'fomc_date' in wm:
        # mark approximate fomc trade index
        fd = wm['fomc_date']
        fomc_mask = trades['entry_time'].dt.date.astype(str) == fd
        if fomc_mask.any():
            ax.axvline(fomc_mask.idxmax(), color='orange', ls='--', alpha=0.7, label='FOMC')
    ax.axhline(0, color='black', lw=0.5)
    bs = BASELINE_STATS[wk]
    ax.set_title(f\"{wk}: {wm['front']}→{wm['back']}  n={bs['n']}  WR={bs['wr']:.1%}  Gross=${bs['gross']:.2f}\")
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Gross P&L ($)')
    ax.legend(fontsize=8)
fig.suptitle('Equity Curves — All 4 Roll Windows (Baseline)', fontsize=13)
save_fig(fig, 'A1_equity_curves_per_window.png')
"""),
])

# ── A2 ─────────────────────────────────────────────────────────────────────
write_nb("A2_combined_equity_curve", "A2 — Combined Equity Curve (All 4 Windows)", [
    code_cell(IMPORTS),
    code_cell("""\
fig, ax = plt.subplots(figsize=(14, 5))
offset = 0
xticks, xlabels = [], []
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    trades = trades.sort_values('entry_time').reset_index(drop=True)
    cum = trades['gross_usd'].cumsum() + offset
    x = range(len(trades))
    ax.step([xi + len(xlabels_prev) if (len_prev := sum(len(t) for t in all_trades[:i])) else xi
             for xi in x], cum, where='post', color=WIN_COLORS[wk], label=wk, linewidth=1.5)
    offset = cum.iloc[-1]
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Trade # (sequential across windows)')
ax.set_ylabel('Cumulative Gross P&L ($)')
ax.set_title('Combined Equity Curve — W1 → W2 → W3 → W4')
ax.legend()
save_fig(fig, 'A2_combined_equity_curve.png')
"""),
    md_cell("**Note:** Rewritten below with correct offset logic."),
    code_cell("""\
fig, ax = plt.subplots(figsize=(14, 5))
offset = 0
x_start = 0
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    trades = trades.sort_values('entry_time').reset_index(drop=True)
    n = len(trades)
    cum = trades['gross_usd'].cumsum() + offset
    xs = list(range(x_start, x_start + n))
    ax.step(xs, cum.values, where='post', color=WIN_COLORS[wk], label=wk, linewidth=1.8)
    ax.axvline(x_start, color='grey', ls=':', lw=0.8)
    ax.text(x_start + n/2, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else -50,
            wk, ha='center', fontsize=9, color=WIN_COLORS[wk])
    offset = float(cum.iloc[-1])
    x_start += n
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('Trade # (sequential across windows)')
ax.set_ylabel('Cumulative Gross P&L ($)')
ax.set_title('Combined Equity Curve — W1 → W2 → W3 → W4')
ax.legend()
save_fig(fig, 'A2_combined_equity_curve.png')
"""),
])

# ── A3 ─────────────────────────────────────────────────────────────────────
write_nb("A3_performance_bar_chart", "A3 — Performance Bar Chart (All Windows)", [
    code_cell(IMPORTS),
    code_cell("""\
import json
windows = list(WINDOWS_META.keys())
x = np.arange(len(windows))
w = 0.35

# load PF from stats.json where available, else use BASELINE_STATS
pf_vals = []
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    sjson = RESULTS_DIR / rk / 'none' / 'stats.json'
    if sjson.exists():
        with open(sjson) as f:
            pf_vals.append(json.load(f).get('profit_factor', BASELINE_STATS[wk]['pf']))
    else:
        pf_vals.append(BASELINE_STATS[wk]['pf'])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Win rate
ax = axes[0]
base_wr = [BASELINE_STATS[wk]['wr'] for wk in windows]
upd_wr  = [UPDATED_STATS[wk]['wr']  for wk in windows]
ax.bar(x - w/2, base_wr, w, label='Baseline', color='steelblue', alpha=0.8)
ax.bar(x + w/2, upd_wr,  w, label='Updated',  color='darkorange', alpha=0.8)
ax.axhline(0.5, color='black', ls='--', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(windows)
ax.set_ylabel('Win Rate'); ax.set_title('Win Rate'); ax.legend()
ax.set_ylim(0, 1)

# Gross P&L
ax = axes[1]
base_g = [BASELINE_STATS[wk]['gross'] for wk in windows]
upd_g  = [UPDATED_STATS[wk]['gross']  for wk in windows]
ax.bar(x - w/2, base_g, w, label='Baseline', color='steelblue', alpha=0.8)
ax.bar(x + w/2, upd_g,  w, label='Updated',  color='darkorange', alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(windows)
ax.set_ylabel('Gross P&L ($)'); ax.set_title('Gross P&L'); ax.legend()

# Profit factor
ax = axes[2]
ax.bar(x, pf_vals, color=[WIN_COLORS[wk] for wk in windows], alpha=0.85)
ax.axhline(1.0, color='black', ls='--', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(windows)
ax.set_ylabel('Profit Factor'); ax.set_title('Profit Factor (Baseline)')

fig.suptitle('Performance Summary — All 4 Roll Windows', fontsize=13)
save_fig(fig, 'A3_performance_bar_chart.png')
"""),
])

# ── A4 ─────────────────────────────────────────────────────────────────────
write_nb("A4_bootstrap_ci_win_rate", "A4 — Bootstrap CI on Win Rate", [
    code_cell(IMPORTS),
    code_cell("import sys; from pathlib import Path; sys.path.insert(0, str(Path('__file__').resolve().parent.parent.parent / 'src'))"),
    code_cell("from strategy import bootstrap_ci"),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    win_flag = (trades['gross_usd'] > 0).astype(int).values
    lo, mid, hi = bootstrap_ci(win_flag, stat_fn=np.mean, n_boot=5000, ci=0.95)
    ax.errorbar([0], [mid], yerr=[[mid-lo],[hi-mid]], fmt='o', capsize=8,
                color=WIN_COLORS[wk], markersize=8)
    ax.axhline(0.5, color='black', ls='--', lw=0.8)
    ax.set_title(f'{wk}\\nn={len(win_flag)}, WR={mid:.3f}')
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
axes[0].set_ylabel('Win Rate')
fig.suptitle('Bootstrap 95% CI on Win Rate — All Windows', fontsize=13)
save_fig(fig, 'A4_bootstrap_ci_win_rate.png')
"""),
])

# ── A5 ─────────────────────────────────────────────────────────────────────
write_nb("A5_gross_pnl_distributions", "A5 — Gross P&L Distributions", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    pnl = trades['gross_usd'].values
    colors = np.where(pnl >= 0, '#2ecc71', '#e74c3c')
    bins = np.arange(-275, 125, 12.5)
    ax.hist(pnl[pnl >= 0], bins=bins, color='#2ecc71', alpha=0.7, label='Win')
    ax.hist(pnl[pnl <  0], bins=bins, color='#e74c3c', alpha=0.7, label='Loss')
    ax.axvline(0, color='black', lw=0.8)
    ax.axvline(pnl.mean(), color='navy', ls='--', lw=1, label=f'Mean ${pnl.mean():.1f}')
    ax.set_title(f'{wk}: n={len(pnl)}')
    ax.set_xlabel('Gross P&L ($)')
    ax.set_xlim(-275, 125)
    ax.legend(fontsize=7)
axes[0].set_ylabel('Count')
fig.suptitle('Gross P&L Distributions — All Windows', fontsize=13)
save_fig(fig, 'A5_gross_pnl_distributions.png')
"""),
])

# ── B1 ─────────────────────────────────────────────────────────────────────
write_nb("B1_daily_mean_deviation", "B1 — Daily Mean Deviation per Roll Day", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts['date'] = ts.index.date.astype(str)
    daily = ts.groupby('date')['dev'].mean().reindex(wm['days'])
    bars = ax.bar(range(len(daily)), daily.values, color=WIN_COLORS[wk], alpha=0.8)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(range(len(wm['day_labels'])))
    ax.set_xticklabels(wm['day_labels'], fontsize=8)
    ax.set_title(f'{wk}: {wm[\"front\"]}→{wm[\"back\"]}')
    ax.set_xlabel('Roll Day')
axes[0].set_ylabel('Mean Spread Deviation (pts)')
fig.suptitle('Daily Mean Deviation from Fair Value — All Windows', fontsize=13)
save_fig(fig, 'B1_daily_mean_deviation.png')
"""),
])

# ── B2 ─────────────────────────────────────────────────────────────────────
write_nb("B2_heatmap_hour_vs_rollday", "B2 — Heatmap: Hour vs Roll Day", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts['roll_day'] = ts.index.normalize().map(
        {pd.Timestamp(d): i for i, d in enumerate(wm['days'])})
    ts['hour_utc'] = ts.index.hour
    ts = ts.dropna(subset=['roll_day'])
    pivot = ts.pivot_table(values='dev', index='hour_utc', columns='roll_day', aggfunc='mean')
    absmax = np.abs(pivot.values).max()
    im = ax.imshow(pivot.values, aspect='auto', cmap='RdBu_r',
                   vmin=-absmax, vmax=absmax, origin='lower')
    ax.set_xticks(range(len(wm['day_labels'])))
    ax.set_xticklabels(wm['day_labels'], fontsize=7)
    ax.set_xlabel('Roll Day')
    ax.set_ylabel('Hour UTC')
    ax.set_title(wk)
    plt.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle('Mean Deviation Heatmap: Hour vs Roll Day', fontsize=13)
save_fig(fig, 'B2_heatmap_hour_vs_rollday.png')
"""),
])

# ── B3 ─────────────────────────────────────────────────────────────────────
write_nb("B3_w3_fv_vs_spread_dual_axis", "B3 — W3: Fair Value vs Spread (Dual Axis)", [
    code_cell(IMPORTS),
    code_cell("""\
wm = WINDOWS_META['W3']
rk = wm['result_key']
ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
ts.index = pd.to_datetime(ts.index)
ts['date'] = ts.index.date.astype(str)
daily = ts.groupby('date')[['fv','spread']].mean().reindex(wm['days'])

fig, ax1 = plt.subplots(figsize=(10, 5))
color_fv = '#3498db'
color_sp = '#e74c3c'
x = range(len(daily))
ax1.plot(x, daily['fv'].values, color=color_fv, marker='o', label='Fair Value (FV)')
ax1.set_ylabel('Fair Value (pts)', color=color_fv)
ax1.tick_params(axis='y', labelcolor=color_fv)
ax1.set_xticks(list(x))
ax1.set_xticklabels(wm['day_labels'])
ax1.set_xlabel('Roll Day')

ax2 = ax1.twinx()
ax2.plot(x, daily['spread'].values, color=color_sp, marker='s', ls='--', label='Spread')
ax2.set_ylabel('Calendar Spread (pts)', color=color_sp)
ax2.tick_params(axis='y', labelcolor=color_sp)

# shade the zone where FV > spread (carry compression)
fv_arr = daily['fv'].values
sp_arr = daily['spread'].values
for i in range(len(x)-1):
    if fv_arr[i] > sp_arr[i]:
        ax1.axvspan(x[i], x[i]+1, alpha=0.12, color='purple')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax1.set_title('W3 ESH5→ESM5: Fair Value Rising vs Spread Falling\\n(shaded = FV > Spread squeeze zone)')
save_fig(fig, 'B3_w3_fv_vs_spread_dual_axis.png')
"""),
])

# ── B4 ─────────────────────────────────────────────────────────────────────
write_nb("B4_entry_zscore_violin", "B4 — Entry Z-Score Violin by Direction", [
    code_cell(IMPORTS),
    code_cell("""\
import matplotlib.patches as mpatches
all_trades = []
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    t = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    t['window'] = wk
    all_trades.append(t)
df = pd.concat(all_trades, ignore_index=True)

fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
for ax, wk in zip(axes, WINDOWS_META.keys()):
    sub = df[df['window'] == wk]
    for j, direction in enumerate(['Long', 'Short']):
        data = sub[sub['dir_label'] == direction]['entry_z'].dropna().values
        if len(data) == 0:
            continue
        parts = ax.violinplot([data], positions=[j], showmedians=True, widths=0.6)
        color = '#2ecc71' if direction == 'Long' else '#e74c3c'
        for pc in parts['bodies']:
            pc.set_facecolor(color); pc.set_alpha(0.7)
    ax.axhline(2.5, color='grey', ls='--', lw=0.8)
    ax.axhline(-2.5, color='grey', ls='--', lw=0.8)
    ax.set_xticks([0,1]); ax.set_xticklabels(['Long','Short'])
    ax.set_title(wk)
axes[0].set_ylabel('Entry Z-Score')
fig.suptitle('Entry Z-Score Distribution by Direction — All Windows', fontsize=13)
save_fig(fig, 'B4_entry_zscore_violin.png')
"""),
])

# ── C1 ─────────────────────────────────────────────────────────────────────
write_nb("C1_ou_halflife_per_rollday", "C1 — OU Half-Life per Roll Day", [
    code_cell(IMPORTS),
    code_cell("""\
import importlib.util
spec = importlib.util.spec_from_file_location('hl_mod', '../notebooks/11_half_life_bars.py')
hl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hl_mod)
hl_series_minutes = hl_mod.hl_series_minutes
"""),
    code_cell("""\
fig, ax = plt.subplots(figsize=(12, 5))
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts['date'] = ts.index.date.astype(str)
    day_hl = []
    for d in wm['days']:
        sub = ts[ts['date'] == d]['dev'].dropna().values
        if len(sub) < 10:
            day_hl.append(np.nan); continue
        try:
            hl = hl_series_minutes(sub, bar_res=1, n_bars=len(sub), bar_secs=1)
            day_hl.append(np.nanmedian(hl))
        except Exception:
            day_hl.append(np.nan)
    ax.plot(range(len(wm['days'])), day_hl, marker='o', label=wk, color=WIN_COLORS[wk])
ax.set_xticks(range(7))
ax.set_xticklabels(['D1','D2','D3','D4','D5','D6','D7'])
ax.set_xlabel('Roll Day')
ax.set_ylabel('Median OU Half-Life (minutes)')
ax.set_title('OU Half-Life per Roll Day — All Windows')
ax.legend()
save_fig(fig, 'C1_ou_halflife_per_rollday.png')
"""),
])

# ── C2 ─────────────────────────────────────────────────────────────────────
write_nb("C2_acf_lag1_per_rollday", "C2 — ACF Lag-1 per Roll Day", [
    code_cell(IMPORTS),
    code_cell("""\
fig, ax = plt.subplots(figsize=(12, 5))
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts['date'] = ts.index.date.astype(str)
    rhos = []
    for d in wm['days']:
        sub = ts[ts['date'] == d]['dev'].dropna().values
        if len(sub) < 3:
            rhos.append(np.nan); continue
        rho = np.corrcoef(sub[1:], sub[:-1])[0,1]
        rhos.append(rho)
    ax.plot(range(len(wm['days'])), rhos, marker='o', label=wk, color=WIN_COLORS[wk])
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(range(7))
ax.set_xticklabels(['D1','D2','D3','D4','D5','D6','D7'])
ax.set_xlabel('Roll Day')
ax.set_ylabel('ACF Lag-1 (rho)')
ax.set_title('ACF Lag-1 per Roll Day — All Windows (negative = mean-reverting)')
ax.legend()
save_fig(fig, 'C2_acf_lag1_per_rollday.png')
"""),
])

# ── C3 ─────────────────────────────────────────────────────────────────────
write_nb("C3_zscore_path_with_trades", "C3 — Z-Score Path with Trade Markers", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts_1min = ts['zscore'].resample('1min').last().dropna()
    ax.plot(ts_1min.index, ts_1min.values, lw=0.6, color='grey', alpha=0.8)

    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    trades['entry_time'] = pd.to_datetime(trades['entry_time'])
    longs  = trades[trades['dir_label'] == 'Long']
    shorts = trades[trades['dir_label'] == 'Short']
    entry_z_long  = trades.loc[longs.index,  'entry_z']
    entry_z_short = trades.loc[shorts.index, 'entry_z']
    ax.scatter(longs['entry_time'],  entry_z_long,  marker='^', color='#2ecc71', s=30, zorder=5, label='Long')
    ax.scatter(shorts['entry_time'], entry_z_short, marker='v', color='#e74c3c', s=30, zorder=5, label='Short')
    ax.axhline(0, color='black', lw=0.5)
    ax.axhline(2.5, color='grey', ls='--', lw=0.5)
    ax.axhline(-2.5, color='grey', ls='--', lw=0.5)
    ax.set_title(f'{wk}: {wm[\"front\"]}→{wm[\"back\"]}')
    ax.set_ylabel('Z-Score')
    ax.legend(fontsize=8)
fig.suptitle('Z-Score Path with Trade Entry Markers — All Windows', fontsize=13)
save_fig(fig, 'C3_zscore_path_with_trades.png')
"""),
])

# ── D1 ─────────────────────────────────────────────────────────────────────
write_nb("D1_mae_mfe_scatter", "D1 — MAE vs MFE Scatter by Exit Type", [
    code_cell(IMPORTS),
    code_cell("""\
EXIT_COLORS = {'TP': '#2ecc71', 'SL': '#e74c3c', 'EOD': '#95a5a6', 'BE': '#f39c12'}
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    for etype, grp in trades.groupby('exit_type'):
        c = EXIT_COLORS.get(etype, 'grey')
        ax.scatter(grp['mae_pts'], grp['mfe_pts'], color=c, alpha=0.7, s=30, label=etype)
    ax.set_xlabel('MAE (pts)')
    ax.set_ylabel('MFE (pts)')
    ax.set_title(wk)
    ax.axhline(0, color='black', lw=0.4)
    ax.axvline(0, color='black', lw=0.4)
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=c, markersize=8, label=l)
               for l, c in EXIT_COLORS.items()]
    ax.legend(handles=handles, fontsize=7)
# annotate W3 trade 27 (worst spike loss)
ax_w3 = axes[2]
wm3 = WINDOWS_META['W3']
t3 = pd.read_parquet(RESULTS_DIR / wm3['result_key'] / 'none' / 'trades.parquet')
worst = t3.loc[t3['mae_pts'].idxmax()]
ax_w3.annotate('Trade 27\\n−$218.75', xy=(worst['mae_pts'], worst['mfe_pts']),
               xytext=(worst['mae_pts']-0.3, worst['mfe_pts']+0.5),
               fontsize=7, arrowprops=dict(arrowstyle='->', color='black'))
fig.suptitle('MAE vs MFE by Exit Type — All Windows', fontsize=13)
save_fig(fig, 'D1_mae_mfe_scatter.png')
"""),
])

# ── D2 ─────────────────────────────────────────────────────────────────────
write_nb("D2_exit_type_stacked_bar", "D2 — Exit Type Stacked Bar (Direction × Window)", [
    code_cell(IMPORTS),
    code_cell("""\
windows = list(WINDOWS_META.keys())
directions = ['Long', 'Short']
exit_types = ['TP', 'SL', 'EOD', 'BE']
ECOLS = {'TP':'#2ecc71','SL':'#e74c3c','EOD':'#95a5a6','BE':'#f39c12'}

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for row, direction in enumerate(directions):
    for col, wk in enumerate(windows):
        ax = axes[row][col]
        wm = WINDOWS_META[wk]
        rk = wm['result_key']
        trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
        sub = trades[trades['dir_label'] == direction]
        counts = sub['exit_type'].value_counts().reindex(exit_types, fill_value=0)
        bottom = 0
        for et in exit_types:
            ax.bar([0], counts[et], bottom=bottom, color=ECOLS[et], label=et if row==0 and col==0 else '')
            bottom += counts[et]
        ax.set_title(f'{direction} / {wk}', fontsize=9)
        ax.set_xticks([])
        if col == 0:
            ax.set_ylabel('# Trades')

handles = [plt.Rectangle((0,0),1,1, color=ECOLS[et]) for et in exit_types]
fig.legend(handles, exit_types, loc='lower center', ncol=4, fontsize=9)
fig.suptitle('Exit Type Distribution by Direction and Window', fontsize=13)
save_fig(fig, 'D2_exit_type_stacked_bar.png')
"""),
])

# ── D3 ─────────────────────────────────────────────────────────────────────
write_nb("D3_entry_hour_histogram", "D3 — Entry Hour Histogram", [
    code_cell(IMPORTS),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=False)
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    hours = trades['entry_hour_utc'].dropna()
    hour_vals = np.arange(12, 22)
    counts = hours.value_counts().reindex(hour_vals, fill_value=0)
    bar_colors = ['orange' if h == 14 else WIN_COLORS[wk] for h in hour_vals]
    ax.bar(hour_vals, counts.values, color=bar_colors, alpha=0.85)
    ax.set_xlabel('Hour (UTC)')
    ax.set_title(wk)
    ax.set_xticks(hour_vals)
axes[0].set_ylabel('# Entries')
fig.suptitle('Entry Hour Distribution (14:00 UTC highlighted) — All Windows', fontsize=13)
save_fig(fig, 'D3_entry_hour_histogram.png')
"""),
])

# ── D4 ─────────────────────────────────────────────────────────────────────
write_nb("D4_long_short_pnl_decomposition", "D4 — Long vs Short P&L Decomposition", [
    code_cell(IMPORTS),
    code_cell("""\
windows = list(WINDOWS_META.keys())
x = np.arange(len(windows))
w = 0.35
long_pnl  = []
short_pnl = []
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
    long_pnl.append(trades[trades['dir_label']=='Long']['gross_usd'].sum())
    short_pnl.append(trades[trades['dir_label']=='Short']['gross_usd'].sum())

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w/2, long_pnl,  w, label='Long',  color='#2ecc71', alpha=0.85)
ax.bar(x + w/2, short_pnl, w, label='Short', color='#e74c3c', alpha=0.85)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(windows)
ax.set_ylabel('Gross P&L ($)')
ax.set_title('Long vs Short P&L Decomposition — All Windows')
ax.legend()
save_fig(fig, 'D4_long_short_pnl_decomp.png')
"""),
])

# ── D5 ─────────────────────────────────────────────────────────────────────
write_nb("D5_wr_by_zscore_bucket", "D5 — Win Rate by Z-Score Bucket", [
    code_cell(IMPORTS),
    code_cell("""\
bins = [0, 1, 2, 3, 4, np.inf]
labels = ['0-1','1-2','2-3','3-4','4+']

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, direction in zip(axes, ['Long', 'Short']):
    for wk, wm in WINDOWS_META.items():
        rk = wm['result_key']
        trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
        sub = trades[trades['dir_label'] == direction].copy()
        sub['z_bucket'] = pd.cut(sub['entry_z'].abs(), bins=bins, labels=labels, right=False)
        wr = sub.groupby('z_bucket', observed=True).apply(lambda g: (g['gross_usd']>0).mean())
        ax.plot(range(len(labels)), wr.reindex(labels).values,
                marker='o', label=wk, color=WIN_COLORS[wk])
    ax.axhline(0.5, color='black', ls='--', lw=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel('|Entry Z| Bucket')
    ax.set_title(f'Win Rate by Z-Score Bucket — {direction}')
    ax.legend(fontsize=8)
axes[0].set_ylabel('Win Rate')
fig.suptitle('Win Rate by Entry Z-Score Magnitude', fontsize=13)
save_fig(fig, 'D5_wr_by_zscore_bucket.png')
"""),
])

# ── E1 ─────────────────────────────────────────────────────────────────────
_e_imports = IMPORTS + "\n# SEAGATE required for E notebooks\n"

def _e_ba_cells(wk, days_key):
    wm_str = f"WINDOWS_META['{wk}']"
    return [
        code_cell(_e_imports),
        code_cell(f"""\
wm = {wm_str}
summary_rows = []
for day in wm['days']:
    for sym in [wm['front'], wm['back']]:
        fpath = SEAGATE_DIR / f'mbp10_{{sym}}_{{wm[\"front\"]}}_{{wm[\"back\"]}}_{{day}}.parquet'
        if not fpath.exists():
            # try alternate naming convention
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{{sym}}*{{day}}*.parquet'))
            fpath = fpath[0] if fpath else None
        if fpath is None:
            print(f'  MISSING: {{sym}} {{day}}'); continue
        df = pd.read_parquet(fpath, columns=['ts_event','symbol','bid_px_00','ask_px_00'])
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        # RTH filter
        rth_mask = ((df['ts_event'].dt.hour * 60 + df['ts_event'].dt.minute) >= wm['rth_start_min']) & \\
                   ((df['ts_event'].dt.hour * 60 + df['ts_event'].dt.minute) <= wm['rth_end_min'])
        df = df[rth_mask]
        df['ba'] = (df['ask_px_00'] - df['bid_px_00']) / TICK
        summary_rows.append(dict(
            day=day, symbol=sym,
            mean=df['ba'].mean(), median=df['ba'].median(),
            pct_1t=(df['ba'] == 1).mean(), pct_2t=(df['ba'] == 2).mean(),
            pct_3t=(df['ba'] >= 3).mean(),
        ))
        del df; gc.collect()

ba_df = pd.DataFrame(summary_rows)
print(ba_df.to_string(index=False))
"""),
        code_cell(f"""\
# Bar chart: mean BA per day (front vs back)
wm = {wm_str}
fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(wm['days']))
w = 0.35
front_means = ba_df[ba_df['symbol']==wm['front']].set_index('day')['mean'].reindex(wm['days'])
back_means  = ba_df[ba_df['symbol']==wm['back']].set_index('day')['mean'].reindex(wm['days'])
ax.bar(x-w/2, front_means.values, w, label=wm['front'], color='steelblue', alpha=0.85)
ax.bar(x+w/2, back_means.values,  w, label=wm['back'],  color='darkorange', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(wm['day_labels'])
ax.set_ylabel('Mean BA Spread (ticks)')
ax.set_title(f'{wk}: Mean Bid-Ask Spread per Day')
ax.legend()
save_fig(fig, 'E{wk[-1]}_ba_spread_{wk.lower()}.png')
"""),
        code_cell(f"""\
# Stacked bar: tick distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, sym in zip(axes, [wm['front'], wm['back']]):
    sub = ba_df[ba_df['symbol']==sym].set_index('day').reindex(wm['days'])
    x = np.arange(len(wm['days']))
    ax.bar(x, sub['pct_1t'].values, label='1-tick', color='#2ecc71', alpha=0.85)
    ax.bar(x, sub['pct_2t'].values, bottom=sub['pct_1t'].values, label='2-tick', color='#3498db', alpha=0.85)
    ax.bar(x, sub['pct_3t'].values, bottom=(sub['pct_1t']+sub['pct_2t']).values, label='3+tick', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(wm['day_labels'])
    ax.set_title(f'{{sym}} BA width distribution')
    ax.set_ylabel('Fraction of quotes')
    ax.legend(fontsize=8)
fig.suptitle(f'{wk} Bid-Ask Width Distribution', fontsize=12)
save_fig(fig, 'E{wk[-1]}_ba_width_dist_{wk.lower()}.png')
"""),
    ]

write_nb("E1_ba_analysis_w2", "E1 — Bid-Ask Analysis W2 (ESZ4/ESH5)", _e_ba_cells("W2", "W2"))
write_nb("E2_ba_analysis_w3", "E2 — Bid-Ask Analysis W3 (ESH5/ESM5)", _e_ba_cells("W3", "W3"))
write_nb("E3_ba_analysis_w4", "E3 — Bid-Ask Analysis W4 (ESM5/ESU5)", _e_ba_cells("W4", "W4"))

# ── E4 ─────────────────────────────────────────────────────────────────────
write_nb("E4_crosswindow_ba_grid", "E4 — Cross-Window BA Grid", [
    code_cell(_e_imports),
    code_cell("""\
rows = []
for wk, wm in WINDOWS_META.items():
    if wk == 'W1':
        continue  # W1 already analysed in notebook 13
    for day in wm['days']:
        for sym in [wm['front'], wm['back']]:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath:
                continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth]
            ba = (df['ask_px_00'] - df['bid_px_00']) / TICK
            rows.append(dict(window=wk, symbol=sym, leg='front' if sym==wm['front'] else 'back',
                             day=day, mean_ba=ba.mean()))
            del df; gc.collect()

grid = pd.DataFrame(rows)
fig, axes = plt.subplots(4, 2, figsize=(12, 14), sharey=False)
for row_i, wk in enumerate(['W2','W3','W4','W1']):
    for col_i, leg in enumerate(['front','back']):
        ax = axes[row_i][col_i]
        sub = grid[(grid['window']==wk) & (grid['leg']==leg)]
        if sub.empty:
            ax.set_visible(False); continue
        ax.bar(range(len(sub)), sub['mean_ba'].values, color=WIN_COLORS.get(wk,'grey'), alpha=0.85)
        ax.set_title(f'{wk} {leg}', fontsize=9)
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels([d[-5:] for d in sub['day'].values], rotation=30, fontsize=7)
        ax.set_ylabel('Mean BA (ticks)', fontsize=8)
fig.suptitle('Cross-Window Mean BA Spread Grid', fontsize=13)
save_fig(fig, 'E4_crosswindow_ba_grid.png')
"""),
])

# ── E5 ─────────────────────────────────────────────────────────────────────
write_nb("E5_width_distribution_comparison", "E5 — Width Distribution Comparison (All Windows)", [
    code_cell(_e_imports),
    code_cell("""\
rows = []
for wk, wm in WINDOWS_META.items():
    if wk == 'W1':
        continue
    for sym in [wm['front'], wm['back']]:
        p1, p2, p3 = [], [], []
        for day in wm['days']:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath: continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth]
            ba = (df['ask_px_00']-df['bid_px_00'])/TICK
            p1.append((ba==1).mean()); p2.append((ba==2).mean()); p3.append((ba>=3).mean())
            del df; gc.collect()
        rows.append(dict(window=wk, leg='front' if sym==wm['front'] else 'back',
                         pct_1t=np.mean(p1), pct_2t=np.mean(p2), pct_3t=np.mean(p3)))
df_wid = pd.DataFrame(rows)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, leg in zip(axes, ['front', 'back']):
    sub = df_wid[df_wid['leg']==leg]
    x = np.arange(len(sub))
    ax.bar(x, sub['pct_1t'], label='1-tick', color='#2ecc71', alpha=0.85)
    ax.bar(x, sub['pct_2t'], bottom=sub['pct_1t'].values, label='2-tick', color='#3498db', alpha=0.85)
    ax.bar(x, sub['pct_3t'], bottom=(sub['pct_1t']+sub['pct_2t']).values, label='3+tick', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sub['window'].values)
    ax.set_title(f'{leg.capitalize()} Contract BA Width')
    ax.set_ylabel('Avg fraction of quotes')
    ax.legend()
fig.suptitle('Bid-Ask Width Distribution Comparison — W2/W3/W4', fontsize=13)
save_fig(fig, 'E5_width_distribution_comparison.png')
"""),
])

# ── E6 ─────────────────────────────────────────────────────────────────────
write_nb("E6_calendar_spread_intraday_w2w3w4", "E6 — Calendar Spread Intraday W2/W3/W4", [
    code_cell(_e_imports),
    code_cell("""\
from matplotlib import cm
plasma = cm.get_cmap('plasma')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, wk in zip(axes, ['W2','W3','W4']):
    wm = WINDOWS_META[wk]
    colors_day = [plasma(i/6) for i in range(7)]
    for di, (day, lbl, dc) in enumerate(zip(wm['days'], wm['day_labels'], colors_day)):
        mids = {}
        for sym in [wm['front'], wm['back']]:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath: continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth].set_index('ts_event')
            mids[sym] = ((df['bid_px_00']+df['ask_px_00'])/2).resample('1min').last().dropna()
            del df; gc.collect()
        if wm['front'] not in mids or wm['back'] not in mids:
            continue
        cal = mids[wm['back']].subtract(mids[wm['front']], fill_value=np.nan).dropna()
        t0 = cal.index.normalize()
        mins = (cal.index - t0).seconds / 60
        ax.plot(mins, cal.values, color=dc, lw=0.8, alpha=0.8, label=lbl)
    ax.set_xlabel('Minutes from midnight UTC')
    ax.set_ylabel('Calendar Spread (pts)')
    ax.set_title(wk)
    ax.legend(fontsize=6)
fig.suptitle('Calendar Spread Intraday — W2/W3/W4 (D1→D7, plasma colormap)', fontsize=13)
save_fig(fig, 'E6_cal_spread_intraday_w2w3w4.png')
"""),
])

# ── E7 ─────────────────────────────────────────────────────────────────────
write_nb("E7_calendar_spread_daily_w2w3w4", "E7 — Calendar Spread Daily Stats W2/W3/W4", [
    code_cell(_e_imports),
    code_cell("""\
daily_rows = []
for wk in ['W2','W3','W4']:
    wm = WINDOWS_META[wk]
    for day in wm['days']:
        mids = {}
        for sym in [wm['front'], wm['back']]:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath: continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth].set_index('ts_event')
            mids[sym] = ((df['bid_px_00']+df['ask_px_00'])/2).resample('1min').last().dropna()
            del df; gc.collect()
        if len(mids) < 2: continue
        cal = mids[wm['back']].subtract(mids[wm['front']], fill_value=np.nan).dropna()
        daily_rows.append(dict(window=wk, day=day,
                               cal_open=cal.iloc[0], cal_close=cal.iloc[-1],
                               cal_drift=cal.iloc[-1]-cal.iloc[0],
                               cal_ba_mean=np.nan))  # ba needs separate load
daily_df = pd.DataFrame(daily_rows)

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
for row, wk in enumerate(['W2','W3','W4']):
    sub = daily_df[daily_df['window']==wk].reset_index(drop=True)
    x = range(len(sub))
    wm = WINDOWS_META[wk]
    # levels
    axes[row][0].plot(x, sub['cal_open'],  marker='o', label='Open',  color='steelblue')
    axes[row][0].plot(x, sub['cal_close'], marker='s', label='Close', color='darkorange')
    axes[row][0].set_title(f'{wk} Spread Levels')
    axes[row][0].legend(fontsize=8)
    # drift
    colors = ['#2ecc71' if d >= 0 else '#e74c3c' for d in sub['cal_drift']]
    axes[row][1].bar(x, sub['cal_drift'], color=colors, alpha=0.85)
    axes[row][1].axhline(0, color='black', lw=0.5)
    axes[row][1].set_title(f'{wk} Daily Drift')
    # xticks
    for col in range(3):
        axes[row][col].set_xticks(list(x))
        axes[row][col].set_xticklabels(wm['day_labels'], fontsize=7)
    axes[row][2].set_visible(False)  # cal_ba placeholder
fig.suptitle('Calendar Spread Daily Stats — W2/W3/W4', fontsize=13)
save_fig(fig, 'E7_cal_spread_daily_w2w3w4.png')
"""),
])

# ── E8 ─────────────────────────────────────────────────────────────────────
write_nb("E8_liquidity_migration_w2w3w4", "E8 — Liquidity Migration W2/W3/W4", [
    code_cell(_e_imports),
    code_cell("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, wk in zip(axes, ['W2','W3','W4']):
    wm = WINDOWS_META[wk]
    front_sz, back_sz = [], []
    for day in wm['days']:
        row_f, row_b = np.nan, np.nan
        for sym, store in [(wm['front'], front_sz), (wm['back'], back_sz)]:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath: store.append(np.nan); continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_sz_00','ask_sz_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth]
            sz = ((df['bid_sz_00']+df['ask_sz_00'])/2).median()
            store.append(sz)
            del df; gc.collect()
    x = range(len(wm['days']))
    ax.plot(x, front_sz, marker='o', label=wm['front'], color='steelblue')
    ax.plot(x, back_sz,  marker='s', label=wm['back'],  color='darkorange')
    ax.set_xticks(list(x)); ax.set_xticklabels(wm['day_labels'], fontsize=8)
    ax.set_title(wk)
    ax.set_ylabel('Median Top-of-Book Size (contracts)')
    ax.legend()
fig.suptitle('Liquidity Migration: Median Bid/Ask Size — W2/W3/W4', fontsize=13)
save_fig(fig, 'E8_liq_migration_w2w3w4.png')
"""),
])

# ── E9 ─────────────────────────────────────────────────────────────────────
write_nb("E9_summary_4panel_w2w3w4", "E9 — Summary 4-Panel (W2/W3/W4)", [
    code_cell(_e_imports),
    md_cell("Reproduces the 4-panel layout from `notebooks/figures/13_summary_4panel_w1.png` for W2, W3, W4."),
    code_cell("""\
def build_summary_panel(wk):
    wm = WINDOWS_META[wk]
    rows = []
    for day in wm['days']:
        for sym in [wm['front'], wm['back']]:
            fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
            if not fpath: continue
            df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00','bid_sz_00','ask_sz_00'])
            df['ts_event'] = pd.to_datetime(df['ts_event'])
            rth = ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) >= wm['rth_start_min']) & \
                  ((df['ts_event'].dt.hour*60+df['ts_event'].dt.minute) <= wm['rth_end_min'])
            df = df[rth]
            ba = (df['ask_px_00']-df['bid_px_00'])/TICK
            sz = (df['bid_sz_00']+df['ask_sz_00'])/2
            rows.append(dict(day=day, symbol=sym,
                             mean_ba=ba.mean(), median_ba=ba.median(),
                             pct_1t=(ba==1).mean(), pct_2t=(ba==2).mean(), pct_3t=(ba>=3).mean(),
                             med_sz=sz.median()))
            del df; gc.collect()
    return pd.DataFrame(rows)

for wk in ['W2','W3','W4']:
    wm = WINDOWS_META[wk]
    df = build_summary_panel(wk)
    if df.empty:
        print(f'{wk}: no data found'); continue
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    x = range(len(wm['days']))
    for sym, style in [(wm['front'],'o-'),(wm['back'],'s--')]:
        sub = df[df['symbol']==sym].set_index('day').reindex(wm['days'])
        axes[0][0].plot(x, sub['mean_ba'],  style, label=sym)
        axes[0][1].plot(x, sub['median_ba'],style, label=sym)
        axes[1][0].plot(x, sub['pct_1t'],   style, label=sym)
        axes[1][1].plot(x, sub['med_sz'],   style, label=sym)
    for ax in axes.flatten():
        ax.set_xticks(list(x)); ax.set_xticklabels(wm['day_labels'], fontsize=8)
        ax.legend(fontsize=8)
    axes[0][0].set_title('Mean BA (ticks)'); axes[0][1].set_title('Median BA (ticks)')
    axes[1][0].set_title('Pct 1-tick quotes'); axes[1][1].set_title('Median Size (cts)')
    fig.suptitle(f'{wk}: {wm[\"front\"]}→{wm[\"back\"]} Bid-Ask Summary', fontsize=13)
    save_fig(fig, f'E9_summary_4panel_{wk.lower()}.png')
"""),
])

# ── F1 ─────────────────────────────────────────────────────────────────────
write_nb("F1_w3_mar14_spike", "F1 — W3 Mar 14 Flash Spike at 14:00 UTC", [
    code_cell(_e_imports),
    code_cell("""\
wm = WINDOWS_META['W3']
day = '2025-03-14'
mids = {}
for sym in [wm['front'], wm['back']]:
    fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
    assert fpath, f'Missing {sym} {day}'
    df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    mask = (df['ts_event'].dt.hour == 13) & (df['ts_event'].dt.minute >= 55) | \
           (df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute <= 5)
    df = df[mask].set_index('ts_event')
    mids[sym] = ((df['bid_px_00']+df['ask_px_00'])/2).resample('1ms').last().ffill()
    del df; gc.collect()

cal = mids[wm['back']].subtract(mids[wm['front']]).dropna()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(cal.index, cal.values, lw=0.8, color='steelblue')
ax.axvline(pd.Timestamp('2025-03-14 14:00:00'), color='red', ls='--', lw=1.2, label='14:00 UTC')
# Annotate trade 27 (entry ~14:00:01, Short SL -$218.75)
ax.annotate('Trade 27\\nShort SL\\n−$218.75',
            xy=(pd.Timestamp('2025-03-14 14:00:01'), cal.reindex([pd.Timestamp('2025-03-14 14:00:01')], method='nearest').values[0]),
            xytext=(pd.Timestamp('2025-03-14 13:58:00'), cal.max()*0.9),
            fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red'))
# Annotate trade 26 (entry ~13:45, running into spike, -$93.75)
ax.annotate('Trade 26\\nShort SL\\n−$93.75',
            xy=(pd.Timestamp('2025-03-14 14:00:00'), cal.reindex([pd.Timestamp('2025-03-14 14:00:00')], method='nearest').values[0]),
            xytext=(pd.Timestamp('2025-03-14 13:56:00'), cal.max()*0.7),
            fontsize=8, color='darkorange', arrowprops=dict(arrowstyle='->', color='darkorange'))
ax.set_xlabel('Time (UTC)'); ax.set_ylabel('Calendar Spread (pts)')
ax.set_title('W3 ESH5→ESM5: 14:00 UTC Flash Spike — 2025-03-14 (1ms resolution)')
ax.legend()
save_fig(fig, 'F1_w3_mar14_spike.png')
"""),
])

# ── F2 ─────────────────────────────────────────────────────────────────────
write_nb("F2_w4_jun13_spike_cluster", "F2 — W4 Jun 13 Spike Cluster at 14:00 UTC", [
    code_cell(_e_imports),
    code_cell("""\
wm = WINDOWS_META['W4']
day = '2025-06-13'
mids = {}
for sym in [wm['front'], wm['back']]:
    fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
    assert fpath, f'Missing {sym} {day}'
    df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    mask = (df['ts_event'].dt.hour == 13) & (df['ts_event'].dt.minute >= 55) | \
           (df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute <= 5)
    df = df[mask].set_index('ts_event')
    mids[sym] = ((df['bid_px_00']+df['ask_px_00'])/2).resample('1ms').last().ffill()
    del df; gc.collect()

cal = mids[wm['back']].subtract(mids[wm['front']]).dropna()

# load trade annotations from results
trades = pd.read_parquet(RESULTS_DIR / wm['result_key'] / 'none' / 'trades.parquet')
trades['entry_time'] = pd.to_datetime(trades['entry_time'])
spike_trades = trades[trades['entry_time'].dt.date.astype(str) == day].copy()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(cal.index, cal.values, lw=0.8, color='steelblue')
ax.axvline(pd.Timestamp('2025-06-13 14:00:00'), color='red', ls='--', lw=1.2, label='14:00 UTC')
trade_labels = {6: 'T6 Long SL\\n−$12.50', 7: 'T7 Long SL\\n−$43.75', 8: 'T8 Long TP\\n+$43.75'}
offsets = {6: 0.3, 7: 0.6, 8: 0.9}
for idx, row in spike_trades.iterrows():
    ti = row['entry_time']
    lbl = trade_labels.get(idx, f'T{idx}')
    cal_val = cal.reindex([ti], method='nearest').values[0]
    color = '#2ecc71' if row['gross_usd'] > 0 else '#e74c3c'
    ax.annotate(lbl, xy=(ti, cal_val), xytext=(ti, cal_val + offsets.get(idx, 0.3)),
                fontsize=8, color=color, arrowprops=dict(arrowstyle='->', color=color))
ax.set_xlabel('Time (UTC)'); ax.set_ylabel('Calendar Spread (pts)')
ax.set_title('W4 ESM5→ESU5: 14:00 UTC Spike Cluster — 2025-06-13 (1ms resolution)')
ax.legend()
save_fig(fig, 'F2_w4_jun13_spike_cluster.png')
"""),
])

# ── F3 ─────────────────────────────────────────────────────────────────────
write_nb("F3_spike_comparison", "F3 — Spike Comparison W3 vs W4", [
    code_cell(_e_imports),
    code_cell("""\
def load_spike_cal(wm, day, t_center=pd.Timestamp('14:00:00').time()):
    mids = {}
    for sym in [wm['front'], wm['back']]:
        fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
        if not fpath: return None
        df = pd.read_parquet(fpath[0], columns=['ts_event','bid_px_00','ask_px_00'])
        df['ts_event'] = pd.to_datetime(df['ts_event'])
        mask = (df['ts_event'].dt.hour == 13) & (df['ts_event'].dt.minute >= 55) | \
               (df['ts_event'].dt.hour == 14) & (df['ts_event'].dt.minute <= 5)
        df = df[mask].set_index('ts_event')
        mids[sym] = ((df['bid_px_00']+df['ask_px_00'])/2).resample('1ms').last().ffill()
        del df; gc.collect()
    if len(mids) < 2: return None
    cal = mids[wm['back']].subtract(mids[wm['front']]).dropna()
    return cal

w3_cal = load_spike_cal(WINDOWS_META['W3'], '2025-03-14')
w4_cal = load_spike_cal(WINDOWS_META['W4'], '2025-06-13')

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, cal, wk, day in [(axes[0],w3_cal,'W3','2025-03-14'),(axes[1],w4_cal,'W4','2025-06-13')]:
    if cal is None:
        ax.text(0.5,0.5,'Data missing',ha='center',transform=ax.transAxes); continue
    t0 = pd.Timestamp(f'{day} 14:00:00')
    secs = (cal.index - t0).total_seconds()
    ax.plot(secs, cal.values, lw=0.8, color=WIN_COLORS[wk])
    ax.axvline(0, color='red', ls='--', lw=1.2, label='t=14:00:00')
    ax.set_xlim(-300, 300)
    ax.set_xlabel('Seconds relative to 14:00 UTC')
    ax.set_ylabel('Calendar Spread (pts)')
    ax.set_title(f'{wk}: {day}')
    ax.legend()
fig.suptitle('Flash Spike Comparison: W3 Mar-14 vs W4 Jun-13 (±5 min)', fontsize=13)
save_fig(fig, 'F3_spike_comparison.png')
"""),
])

# ── F4 ─────────────────────────────────────────────────────────────────────
write_nb("F4_orderbook_depth_at_spike", "F4 — Order Book Depth at Spike (W3/W4)", [
    code_cell(_e_imports),
    code_cell("""\
LEVELS = 10
bid_px_cols = [f'bid_px_0{i}' if i < 10 else f'bid_px_{i}' for i in range(LEVELS)]
ask_px_cols = [f'ask_px_0{i}' if i < 10 else f'ask_px_{i}' for i in range(LEVELS)]
bid_sz_cols = [f'bid_sz_0{i}' if i < 10 else f'bid_sz_{i}' for i in range(LEVELS)]
ask_sz_cols = [f'ask_sz_0{i}' if i < 10 else f'ask_sz_{i}' for i in range(LEVELS)]
ALL_COLS = ['ts_event','symbol'] + bid_px_cols + ask_px_cols + bid_sz_cols + ask_sz_cols

# fix column names: they use 2 digits like bid_px_00..09
bid_px_cols = [f'bid_px_{i:02d}' for i in range(LEVELS)]
ask_px_cols = [f'ask_px_{i:02d}' for i in range(LEVELS)]
bid_sz_cols = [f'bid_sz_{i:02d}' for i in range(LEVELS)]
ask_sz_cols = [f'ask_sz_{i:02d}' for i in range(LEVELS)]
ALL_COLS = ['ts_event','symbol'] + bid_px_cols + ask_px_cols + bid_sz_cols + ask_sz_cols
"""),
    code_cell("""\
def get_book_snapshot(wm, day, sym, t_snap):
    fpath = list(SEAGATE_DIR.glob(f'mbp10_*{sym}*{day}*.parquet'))
    if not fpath: return None
    df = pd.read_parquet(fpath[0], columns=ALL_COLS)
    df['ts_event'] = pd.to_datetime(df['ts_event'])
    snap_mask = df['ts_event'].sub(t_snap).abs() < pd.Timedelta('5s')
    snap = df[snap_mask]
    if snap.empty: return None
    snap_row = snap.iloc[snap['ts_event'].sub(t_snap).abs().argmin()]
    del df; gc.collect()
    return snap_row

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
row_configs = [
    (WINDOWS_META['W3'], '2025-03-14', 'W3 Mar-14'),
    (WINDOWS_META['W4'], '2025-06-13', 'W4 Jun-13'),
]
for row_i, (wm, day, label) in enumerate(row_configs):
    t14 = pd.Timestamp(f'{day} 14:00:00')
    for col_i, (sym, offset_s) in enumerate([(wm['front'], -5),(wm['back'], -5),
                                              (wm['front'], +5),(wm['back'], +5)]):
        ax = axes[row_i][col_i]
        t_snap = t14 + pd.Timedelta(seconds=offset_s)
        snap = get_book_snapshot(wm, day, sym, t_snap)
        if snap is None:
            ax.text(0.5,0.5,'Missing', ha='center', transform=ax.transAxes)
            continue
        bid_prices = [snap[c] for c in bid_px_cols]
        ask_prices = [snap[c] for c in ask_px_cols]
        bid_sizes  = [snap[c] for c in bid_sz_cols]
        ask_sizes  = [snap[c] for c in ask_sz_cols]
        ax.barh(range(LEVELS), bid_sizes, color='#2ecc71', alpha=0.8, label='Bid')
        ax.barh(range(LEVELS), [-s for s in ask_sizes], color='#e74c3c', alpha=0.8, label='Ask')
        ax.axvline(0, color='black', lw=0.5)
        title = f'{label} {sym}\\n{"−5s" if offset_s<0 else "+5s"} @ 14:00'
        ax.set_title(title, fontsize=8)
        ax.set_yticks(range(LEVELS))
        ax.set_yticklabels([f'L{i}' for i in range(LEVELS)], fontsize=7)
        if col_i == 0:
            ax.set_ylabel('Book Level')
        if row_i == 0 and col_i == 0:
            ax.legend(fontsize=7)
fig.suptitle('Order Book Depth at 14:00 UTC Spike: Before (−5s) vs After (+5s)', fontsize=13)
save_fig(fig, 'F4_orderbook_depth_at_spike.png')
"""),
])

# ── G1 ─────────────────────────────────────────────────────────────────────
write_nb("G1_volume_migration_curves", "G1 — Volume Migration Curves", [
    code_cell(IMPORTS),
    code_cell("""\
fig, ax = plt.subplots(figsize=(12, 5))
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    varc = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'volume_arc.parquet')
    bs = varc['back_share'].values
    x = range(len(bs))
    ax.plot(x, bs, marker='o', label=wk, color=WIN_COLORS[wk], lw=1.8)
    # mark crossover
    crossover = next((i for i, v in enumerate(bs) if v >= 0.5), None)
    if crossover is not None:
        ax.scatter([crossover], [bs[crossover]], s=100, color=WIN_COLORS[wk],
                   edgecolors='black', zorder=5)
ax.axhline(0.5, color='black', ls='--', lw=0.8, label='50% crossover')
ax.set_xticks(range(7))
ax.set_xticklabels(['D1','D2','D3','D4','D5','D6','D7'])
ax.set_xlabel('Roll Day')
ax.set_ylabel('Back Contract Volume Share')
ax.set_title('Volume Migration Curves — All 4 Windows (circle = crossover day)')
ax.legend()
save_fig(fig, 'G1_volume_migration_curves.png')
"""),
])

# ── G2 ─────────────────────────────────────────────────────────────────────
write_nb("G2_crossover_day_comparison", "G2 — Volume Crossover Day Comparison", [
    code_cell(IMPORTS),
    code_cell("""\
crossover_days = {}
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    varc = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'volume_arc.parquet')
    bs = varc['back_share'].values
    crossover = next((i+1 for i, v in enumerate(bs) if v >= 0.5), None)
    crossover_days[wk] = crossover if crossover is not None else 8

fig, ax = plt.subplots(figsize=(8, 5))
windows = list(crossover_days.keys())
days = [crossover_days[wk] for wk in windows]
colors = [WIN_COLORS[wk] for wk in windows]
bars = ax.bar(windows, days, color=colors, alpha=0.85)
for bar, d in zip(bars, days):
    ax.text(bar.get_x()+bar.get_width()/2, d+0.05, f'D{d}', ha='center', fontsize=10)
ax.set_ylabel('Crossover Day (D1=1)')
ax.set_ylim(0, 9)
ax.set_title('Volume Crossover Day Comparison — All Windows\\n(first day back_share ≥ 50%)')
save_fig(fig, 'G2_crossover_day_comparison.png')
"""),
])

# ── G3 ─────────────────────────────────────────────────────────────────────
write_nb("G3_oi_proxy_migration", "G3 — OI Proxy Migration (OHLCV1D Volume)", [
    code_cell(_e_imports),
    code_cell("""\
fig, axes = plt.subplots(1, 4, figsize=(18, 5))
for ax, (wk, wm) in zip(axes, WINDOWS_META.items()):
    # load local volume_arc
    rk = wm['result_key']
    varc = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'volume_arc.parquet')
    bs_local = varc['back_share'].values

    # load OHLCV1D from SEAGATE as proxy
    ohlcv_files = list(SEAGATE_DIR.glob(f'ohlcv1d_{wm[\"front\"]}_{wm[\"back\"]}*.parquet'))
    if not ohlcv_files:
        ohlcv_files = list(SEAGATE_DIR.glob(f'ohlcv*{wm[\"front\"]}*{wm[\"back\"]}*.parquet'))
    if ohlcv_files:
        ohlcv = pd.read_parquet(ohlcv_files[0])
        ohlcv['date'] = pd.to_datetime(ohlcv.index if ohlcv.index.name == 'ts_event'
                                        else ohlcv.get('ts_event', ohlcv.index)).dt.date.astype(str)
        days_in_roll = wm['days']
        front_vols, back_vols = [], []
        for d in days_in_roll:
            day_data = ohlcv[ohlcv['date'] == d]
            fv = day_data[day_data.get('symbol', day_data.index) == wm['front']]['volume'].sum() if 'symbol' in ohlcv.columns else np.nan
            bv = day_data[day_data.get('symbol', day_data.index) == wm['back']]['volume'].sum()  if 'symbol' in ohlcv.columns else np.nan
            front_vols.append(fv); back_vols.append(bv)
        fv_arr = np.array(front_vols, dtype=float)
        bv_arr = np.array(back_vols,  dtype=float)
        total = fv_arr + bv_arr
        bs_ohlcv = np.where(total > 0, bv_arr / total, np.nan)
        ax.plot(range(7), bs_ohlcv, ls='--', marker='s', color=WIN_COLORS[wk],
                alpha=0.6, label='OHLCV1D proxy')
        del ohlcv; gc.collect()

    ax.plot(range(len(bs_local)), bs_local, marker='o', color=WIN_COLORS[wk],
            label='Results arc', lw=1.8)
    ax.axhline(0.5, color='black', ls=':', lw=0.8)
    ax.set_xticks(range(7)); ax.set_xticklabels(['D1','D2','D3','D4','D5','D6','D7'])
    ax.set_title(wk); ax.legend(fontsize=7)
    ax.set_ylabel('Back Share')
    ax.set_xlabel('Roll Day')
    ax.annotate('* volume proxy,\\nnot true OI', xy=(0.02,0.02), xycoords='axes fraction', fontsize=7)
fig.suptitle('Volume Migration: Results Arc vs OHLCV1D Proxy\\n(note: OHLCV1D volume ≠ open interest)', fontsize=12)
save_fig(fig, 'G3_oi_proxy_migration.png')
"""),
])

# ── H1 ─────────────────────────────────────────────────────────────────────
write_nb("H1_preroll_deviation_scatter", "H1 — Pre-Roll Deviation vs P&L Scatter", [
    code_cell(IMPORTS),
    code_cell("""\
import numpy as np

pre_roll_dev = {}
baseline_gross = {}
for wk, wm in WINDOWS_META.items():
    rk = wm['result_key']
    ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    d1 = wm['days'][0]
    d1_data = ts[ts.index.date.astype(str) == d1]['dev']
    pre_roll_dev[wk] = d1_data.mean()  # D1 mean as proxy
    baseline_gross[wk] = BASELINE_STATS[wk]['gross']

windows = list(WINDOWS_META.keys())
x_dev = [pre_roll_dev[wk] for wk in windows]
y_pnl = [baseline_gross[wk] for wk in windows]

m, b = np.polyfit(x_dev, y_pnl, 1)
x_fit = np.linspace(min(x_dev), max(x_dev), 100)

fig, ax = plt.subplots(figsize=(8, 6))
for wk, xv, yv in zip(windows, x_dev, y_pnl):
    ax.scatter(xv, yv, color=WIN_COLORS[wk], s=120, zorder=5)
    ax.annotate(wk, (xv, yv), textcoords='offset points', xytext=(6, 4), fontsize=10)
ax.plot(x_fit, m*x_fit+b, ls='--', color='grey', lw=1.2, label=f'OLS: y={m:.1f}x+{b:.1f}')
ax.axhline(0, color='black', lw=0.5)
ax.set_xlabel('D1 Mean Deviation (pts) — pre-roll regime proxy')
ax.set_ylabel('Baseline Gross P&L ($)')
ax.set_title('Pre-Roll Deviation vs Strategy P&L\\n(caveat: D1 is first trade day, not true pre-roll)')
ax.legend()
ax.annotate('* D1 used as pre-roll proxy; true pre-roll data not available',
            xy=(0.01, 0.01), xycoords='axes fraction', fontsize=8, color='grey')
save_fig(fig, 'H1_preroll_deviation_scatter.png')
"""),
])

# ── H2 ─────────────────────────────────────────────────────────────────────
write_nb("H2_sofr_trajectory", "H2 — SOFR Rate Trajectory per Roll Window", [
    code_cell(_e_imports),
    code_cell("""\
sofr_candidates = list(SEAGATE_DIR.glob('SOFR*.csv')) + list(SEAGATE_DIR.glob('sofr*.csv'))
if not sofr_candidates:
    # fallback: try to fetch from FRED (public, no key)
    import urllib.request, io
    url = 'https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR'
    with urllib.request.urlopen(url) as r:
        sofr = pd.read_csv(io.BytesIO(r.read()), parse_dates=['DATE'], index_col='DATE')
    sofr.columns = ['rate']
    print('Fetched SOFR from FRED')
else:
    sofr = pd.read_csv(sofr_candidates[0], parse_dates=[0], index_col=0)
    sofr.columns = ['rate']
    print(f'Loaded {sofr_candidates[0].name}')

fig, ax = plt.subplots(figsize=(12, 5))
for wk, wm in WINDOWS_META.items():
    roll_start = pd.Timestamp(wm['roll_start'])
    window = sofr.loc[roll_start - pd.Timedelta(days=10) : roll_start + pd.Timedelta(days=1), 'rate']
    window = window.dropna()
    if window.empty: continue
    x = range(len(window))
    ax.plot(x, window.values, marker='o', label=wk, color=WIN_COLORS[wk], lw=1.5)
    delta = window.iloc[-1] - window.iloc[0]
    ax.annotate(f'Δ={delta:+.3f}%', xy=(len(window)-1, window.iloc[-1]),
                xytext=(len(window)-1+0.2, window.iloc[-1]), fontsize=8)
ax.set_xlabel('Days before roll week start')
ax.set_ylabel('SOFR Rate (%)')
ax.set_title('SOFR Rate 10 Days Pre-Roll per Window')
ax.legend()
save_fig(fig, 'H2_sofr_trajectory.png')
"""),
])

# ── H3 ─────────────────────────────────────────────────────────────────────
write_nb("H3_session_open_hl_gate_w3", "H3 — Session-Open HL Gate vs P&L (W3)", [
    code_cell(IMPORTS),
    code_cell("""\
import importlib.util
spec = importlib.util.spec_from_file_location('hl_mod', '../notebooks/11_half_life_bars.py')
hl_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hl_mod)
hl_series_minutes = hl_mod.hl_series_minutes
"""),
    code_cell("""\
wm = WINDOWS_META['W3']
rk = wm['result_key']
ts = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'timeseries.parquet')
ts.index = pd.to_datetime(ts.index)
trades = pd.read_parquet(RESULTS_DIR / rk / 'none' / 'trades.parquet')
trades = trades.sort_values('entry_time').reset_index(drop=True)

day_hl, day_cum_pnl = [], []
for i, d in enumerate(wm['days']):
    sub = ts[ts.index.date.astype(str) == d]
    # first 30 min of RTH
    rth_start = sub.index.min() + pd.Timedelta(minutes=0)
    thirty_min_mask = (sub.index >= rth_start) & (sub.index < rth_start + pd.Timedelta(minutes=30))
    open_dev = sub[thirty_min_mask]['dev'].dropna().values
    if len(open_dev) < 5:
        day_hl.append(np.nan); day_cum_pnl.append(np.nan); continue
    try:
        hl = hl_series_minutes(open_dev, bar_res=1, n_bars=len(open_dev), bar_secs=1)
        day_hl.append(np.nanmedian(hl))
    except Exception:
        day_hl.append(np.nan)
    # cumulative PnL through end of this day
    day_trades = trades[trades['entry_time'].dt.date.astype(str) == d]
    day_cum_pnl.append(day_trades['gross_usd'].sum())

fig, ax1 = plt.subplots(figsize=(12, 5))
x = range(len(wm['days']))
color_hl = '#3498db'
color_pnl = '#2ecc71'
bars = ax1.bar(x, day_hl, color=color_hl, alpha=0.6, label='Session-open HL (min)')
for i, (b, h) in enumerate(zip(bars, day_hl)):
    if h is not None and not np.isnan(h) and h > 120:
        b.set_edgecolor('red'); b.set_linewidth(2)
ax1.set_ylabel('OU Half-Life (minutes)', color=color_hl)
ax1.tick_params(axis='y', labelcolor=color_hl)
ax1.set_xticks(list(x)); ax1.set_xticklabels(wm['day_labels'])
ax1.axhline(120, color='red', ls='--', lw=0.8, label='HL=120 min threshold')

ax2 = ax1.twinx()
cum_pnl_cum = np.cumsum([v if v is not None and not np.isnan(v) else 0 for v in day_cum_pnl])
ax2.step(x, cum_pnl_cum, where='post', color=color_pnl, lw=2, label='Cumulative P&L')
ax2.axhline(0, color='black', lw=0.5)
ax2.set_ylabel('Cumulative P&L ($)', color=color_pnl)
ax2.tick_params(axis='y', labelcolor=color_pnl)

lines1, lbls1 = ax1.get_legend_handles_labels()
lines2, lbls2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, lbls1+lbls2, loc='upper left')
ax1.set_title('W3: Session-Open HL Gate vs Daily P&L\\n(red border = HL > 120 min, spread not mean-reverting at open)')
save_fig(fig, 'H3_session_open_hl_gate_w3.png')
"""),
])

print("\nDone — all 36 notebooks written.")
