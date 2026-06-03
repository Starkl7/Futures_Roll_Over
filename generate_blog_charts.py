"""Generate 4 blog post charts for the Substack post."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT        = Path('/Users/stark/Desktop/Projects/Futures_RollOver')
RESULTS_DIR = ROOT / 'results'
OUT_DIR     = ROOT / 'blog_figures'
OUT_DIR.mkdir(exist_ok=True)

TICK = 0.25
MULT = 50

WIN_COLORS = {'W1': '#2ecc71', 'W2': '#3498db', 'W3': '#e74c3c', 'W4': '#f39c12'}

WINDOWS = {
    'W1': dict(key='ESU4_ESZ4_20240912',  label='W1  ESU4→ESZ4\nSep 2024'),
    'W2': dict(key='ESZ4_ESH5_20241212',  label='W2  ESZ4→ESH5\nDec 2024'),
    'W3': dict(key='ESH5_ESM5_20250313',  label='W3  ESH5→ESM5\nMar 2025'),
    'W4': dict(key='ESM5_ESU5_20250612',  label='W4  ESM5→ESU5\nJun 2025'),
}

V1_SESSIONS      = ['European_V1', 'US_RTH_V1', 'Post_close_V1']
UNGATED_SESSIONS = ['European_Ungated', 'US_RTH_Ungated', 'Post_close_Ungated']

plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
})


def load_trades(result_key, sessions):
    dfs = []
    for sess in sessions:
        p = RESULTS_DIR / result_key / sess / 'trades.parquet'
        if p.exists():
            df = pd.read_parquet(p)
            df['_sess'] = sess
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    out['entry_time'] = pd.to_datetime(out['entry_time'])
    out = (out.drop_duplicates(subset=['entry_time', 'direction'])
              .sort_values('entry_time')
              .reset_index(drop=True))
    return out


# ════════════════════════════════════════════════════════════════════════════
# CHART 1 — Fair Value vs Actual Spread + Back Volume Share (dual axis)
#   Shows the inconsistent roll premium across quarters
# ════════════════════════════════════════════════════════════════════════════
print('Generating Chart 1: Fair Value vs Actual Spread + Volume Share …')

fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharey=False)
fig.suptitle(
    'Theoretical Fair Value vs Actual Spread — Roll Premium Is Not Consistent Across Quarters',
    fontsize=12, fontweight='bold', y=1.02
)

for ax, (wk, wm) in zip(axes, WINDOWS.items()):
    ts = pd.read_parquet(RESULTS_DIR / wm['key'] / 'none' / 'timeseries.parquet')
    ts.index = pd.to_datetime(ts.index)
    ts['date'] = ts.index.date
    daily = ts.groupby('date')[['fv', 'spread']].mean()
    daily['premium'] = daily['spread'] - daily['fv']

    varc = pd.read_parquet(RESULTS_DIR / wm['key'] / 'none' / 'volume_arc.parquet')
    varc.index = pd.to_datetime(varc.index)
    varc.index = varc.index.date
    daily = daily.join(varc[['back_share']], how='left')

    x = np.arange(len(daily))
    labels = [d.strftime('%b %d') for d in daily.index]

    # ── left axis: spread & FV ───────────────────────────────────────────
    ax.fill_between(x, daily['fv'].values, daily['spread'].values,
                    alpha=0.18, color=WIN_COLORS[wk])
    ax.plot(x, daily['fv'].values,     color='#5dade2', marker='o', lw=2,
            markersize=5, label='Fair Value (FV)')
    ax.plot(x, daily['spread'].values, color=WIN_COLORS[wk], marker='s', lw=2,
            markersize=5, ls='--', label='Actual Spread')

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=7.5)
    ax.set_ylabel('Spread / FV (pts)' if wk == 'W1' else '', fontsize=8)
    ax.set_title(wm['label'], fontsize=9, fontweight='bold', color=WIN_COLORS[wk])

    avg_prem = daily['premium'].mean()
    ax.text(0.04, 0.97, f'Avg premium: {avg_prem:+.2f} pts',
            transform=ax.transAxes, va='top', fontsize=8, color='#7f8c8d',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))

    # ── right axis: back volume share (bars only) ────────────────────────
    ax2 = ax.twinx()
    ax2.bar(x, daily['back_share'].values * 100, color='#8e44ad',
            alpha=0.22, width=0.6, zorder=1, label='Back vol share %')
    ax2.set_ylim(0, 105)
    ax2.set_ylabel('Back vol share (%)' if wk == 'W4' else '', fontsize=8,
                   color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad', labelsize=7)
    ax2.spines['right'].set_visible(True)

    # ── shaded valid trading window (5% < back_share < 80%) ──────────────
    valid_idx = [i for i, v in enumerate(daily['back_share'].values)
                 if 0.05 < v < 0.80]
    if valid_idx:
        x_start = valid_idx[0]  - 0.5
        x_end   = valid_idx[-1] + 0.5
        ax.axvspan(x_start, x_end, alpha=0.10, color='#27ae60', zorder=0)
        ax.axvline(x_start, color='#27ae60', ls='--', lw=1.1, alpha=0.7)
        ax.axvline(x_end,   color='#e74c3c', ls='--', lw=1.1, alpha=0.7)

    if wk == 'W1':
        ax.legend(fontsize=7.5, loc='lower right')
        ax2.legend(fontsize=7.5, loc='upper right')

fig.text(0.5, -0.04,
         'Shaded area between curves = roll premium (spread − FV). '
         'Green shaded region = valid trading days (5% < back vol share < 80%). '
         'Green/red dashed lines = entry and exit of tradeable window. '
         'D6–D7 always breach 80% as back-month dominates into expiry.',
         ha='center', fontsize=8.5, color='#555')
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart1_fv_vs_spread.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  → chart1_fv_vs_spread.png')


# ════════════════════════════════════════════════════════════════════════════
# CHART 2 — Z-Score Bucket Performance
#   Win rate + avg net/lot for low / mid / high-z buckets, V1 trades
# ════════════════════════════════════════════════════════════════════════════
print('Generating Chart 2: Z-Score Bucket Performance …')

all_parts = []
for wk, wm in WINDOWS.items():
    df = load_trades(wm['key'], V1_SESSIONS)
    if not df.empty:
        df['window'] = wk
        all_parts.append(df)

all_df = pd.concat(all_parts, ignore_index=True)
all_df['z_abs']    = all_df['entry_z'].abs()
all_df['z_bucket'] = pd.cut(
    all_df['z_abs'],
    bins=[0, 2.0, 3.0, np.inf],
    labels=['Low-z\n|z| < 2.0', 'Mid-z\n2.0 ≤ |z| < 3.0', 'High-z\n|z| ≥ 3.0']
)
all_df['win'] = all_df['gross_usd'] > 0

stats = (all_df.groupby('z_bucket', observed=True)
               .agg(n=('gross_usd', 'count'),
                    wr=('win', 'mean'),
                    avg_net=('net_Tight', 'mean'))
               .reset_index())

BUCKET_COLORS = ['#95a5a6', '#3498db', '#e74c3c']
xlabels = stats['z_bucket'].astype(str).tolist()
x = np.arange(len(xlabels))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(
    'Performance by Entry Z-Score Magnitude — V1 Strategy, All 4 Windows Combined',
    fontsize=12, fontweight='bold'
)

# Win rate
ax = axes[0]
bars = ax.bar(x, stats['wr'] * 100, color=BUCKET_COLORS, alpha=0.88,
              edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(50, color='#aaa', ls=':', lw=0.8)
for bar, row in zip(bars, stats.itertuples()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f'n={row.n}\n{row.wr*100:.1f}%', ha='center', va='bottom',
            fontsize=8.5, fontweight='bold')
ax.set_xticks(x);  ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel('Win Rate (%)')
ax.set_title('Win Rate', fontsize=10)
ax.set_ylim(0, 102)

# Avg net per lot
ax = axes[1]
bars = ax.bar(x, stats['avg_net'], color=BUCKET_COLORS, alpha=0.88,
              edgecolor='white', linewidth=1.5, zorder=3)
ax.axhline(0, color='black', lw=1.2)
for bar, val in zip(bars, stats['avg_net']):
    offset = 0.25 if val >= 0 else -1.2
    ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
            f'${val:+.2f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold')
ax.set_xticks(x);  ax.set_xticklabels(xlabels, fontsize=9)
ax.set_ylabel('Avg Net P&L per Lot ($)')
ax.set_title('Avg Net P&L per Lot', fontsize=10)

fig.text(0.5, -0.03,
         '75% of signals are Low-z — just barely above the ±2.5σ entry threshold. '
         'Transaction costs ($8.04/lot) consume the entire gross edge at this z-level.',
         ha='center', fontsize=8.5, color='#555')
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart2_zscore_buckets.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  → chart2_zscore_buckets.png')


# ════════════════════════════════════════════════════════════════════════════
# CHART 3 — V1 (Gated) vs Benchmark, 2×2 panels, time-based x-axis
# ════════════════════════════════════════════════════════════════════════════
print('Generating Chart 3: V1 vs Benchmark Equity Curve (time axis) …')

LOT_SCALE = 10

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
fig.suptitle(
    'V1 (drift gate) vs Benchmark — Cumulative Net P&L per Window\n'
    'Ungated OOS: p = 0.264 (not significant)   ·   V1 OOS: p = 0.006***',
    fontsize=12, fontweight='bold'
)

IS_WINS  = ('W1', 'W2')
AX_MAP   = {'W1': axes[0,0], 'W2': axes[0,1], 'W3': axes[1,0], 'W4': axes[1,1]}
BG_COLOR = {'W1': '#eafaf1', 'W2': '#eaf4fb', 'W3': '#fdf2f8', 'W4': '#fef9e7'}

for wk, wm in WINDOWS.items():
    ax = AX_MAP[wk]
    is_oos = 'In-sample' if wk in IS_WINS else 'Out-of-sample'

    # light window-colour background
    ax.set_facecolor(BG_COLOR[wk])

    for label, sessions, color, ls, lw in [
        ('V1 (drift gate)', V1_SESSIONS,      '#2c3e50', '-',  2.2),
        ('Benchmark',       UNGATED_SESSIONS,  '#95a5a6', '--', 1.5),
    ]:
        df = load_trades(wm['key'], sessions)
        if df.empty:
            continue
        df = df.sort_values('entry_time').reset_index(drop=True)
        times = df['entry_time'].dt.tz_localize(None) if df['entry_time'].dt.tz is not None \
                else df['entry_time']
        cum = df['net_Tight'].cumsum() * LOT_SCALE

        # step plot against actual entry times
        ax.step(times, cum.values, where='post', color=color, ls=ls, lw=lw,
                label=label, alpha=0.92)

    ax.axhline(0, color='black', lw=0.8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    is_oos_tag = f'[{is_oos}]'
    ax.set_title(f'{wk}  {wm["label"].split(chr(10))[1].strip()}  {is_oos_tag}',
                 fontsize=10, fontweight='bold', color=WIN_COLORS[wk])
    ax.set_ylabel('Cum. Net P&L — 10 lots ($)', fontsize=8)
    ax.legend(fontsize=8, loc='upper left')

    # annotate UMich event on W3
    if wk == 'W3':
        t_umich = pd.Timestamp('2025-03-14 14:00:00')
        ax.axvline(t_umich, color='#e74c3c', ls=':', lw=1.2, alpha=0.8)
        ylims = ax.get_ylim()
        ax.text(t_umich, ylims[0] + (ylims[1]-ylims[0])*0.05,
                ' UMich\n spike', fontsize=7.5, color='#c0392b')

fig.tight_layout()
fig.savefig(OUT_DIR / 'chart3_equity_curve.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  → chart3_equity_curve.png')


# ════════════════════════════════════════════════════════════════════════════
# CHART 4 — W3 Mar 14 Flash Spike
# ════════════════════════════════════════════════════════════════════════════
print('Generating Chart 4: W3 Mar 14 Flash Spike …')

ts = pd.read_parquet(RESULTS_DIR / 'ESH5_ESM5_20250313' / 'none' / 'timeseries.parquet')
ts.index = pd.to_datetime(ts.index)

mask   = (ts.index >= '2025-03-14 13:58:00') & (ts.index <= '2025-03-14 14:02:00')
window = ts[mask][['spread']].resample('1s').last().ffill().dropna()

t_release = pd.Timestamp('2025-03-14 14:00:00', tz='UTC')
t_entry   = pd.Timestamp('2025-03-14 14:00:01', tz='UTC')
t_peak    = pd.Timestamp('2025-03-14 14:00:03', tz='UTC')
t_revert  = pd.Timestamp('2025-03-14 14:00:07', tz='UTC')

pre   = window[window.index <  t_release]
spike = window[(window.index >= t_release) & (window.index <= t_revert)]
post  = window[window.index >  t_revert]

fig, ax = plt.subplots(figsize=(13, 5))

ax.plot(pre.index,   pre['spread'],   color='#2c3e50', lw=2.2)
ax.plot(spike.index, spike['spread'], color='#e74c3c', lw=2.8, zorder=5)
ax.plot(post.index,  post['spread'],  color='#2c3e50', lw=2.2)

# Normal range band
ax.axhspan(51.25, 52.25, alpha=0.07, color='steelblue', label='Normal trading band')

# Release line
ax.axvline(t_release, color='#e74c3c', ls='--', lw=1.3, alpha=0.75)

# ── Annotations ──────────────────────────────────────────────────────────
entry_y = float(window.loc[window.index >= t_entry].iloc[0]['spread'])
peak_y  = float(window.loc[t_peak, 'spread'])
rev_y   = float(window.loc[window.index >= t_revert].iloc[0]['spread'])

ax.annotate(
    'Short entry  z = +3.32\n(HC trigger: 2× lots)',
    xy=(t_entry, entry_y),
    xytext=(pd.Timestamp('2025-03-14 13:58:30', tz='UTC'), 55.2),
    fontsize=8.5, color='#c0392b', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.5)
)
ax.annotate(
    f'Peak: 56.625 pts  z = +19.76\n4.375 pts through stop in 2 sec',
    xy=(t_peak, peak_y),
    xytext=(pd.Timestamp('2025-03-14 14:00:40', tz='UTC'), 56.0),
    fontsize=8.5, color='#922b21', fontweight='bold',
    arrowprops=dict(arrowstyle='->', color='#922b21', lw=1.5)
)
ax.annotate(
    'Fully reverted\n(7 seconds total)',
    xy=(t_revert, rev_y),
    xytext=(pd.Timestamp('2025-03-14 14:01:10', tz='UTC'), 54.2),
    fontsize=8.5, color='#27ae60',
    arrowprops=dict(arrowstyle='->', color='#27ae60', lw=1.5)
)

# UMich label
ax.text(t_release + pd.Timedelta('4s'), 51.5,
        '14:00 UTC — UMich Consumer Sentiment\n(preliminary, 2.5-yr low)',
        fontsize=8, color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.3', fc='#fdf2f2', ec='#e74c3c', alpha=0.85))

ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
ax.set_xlabel('Time (UTC)  ·  March 14, 2025', fontsize=10)
ax.set_ylabel('ESH5 → ESM5 Calendar Spread (pts)', fontsize=10)
ax.set_title(
    'W3: Flash Spike at 14:00 UTC — UMich Consumer Sentiment Release\n'
    '5 points · 20 ticks · 7 seconds · fully reverted. Net loss: $4,312.',
    fontsize=11, fontweight='bold'
)
ax.legend(fontsize=9, loc='upper left')
fig.tight_layout()
fig.savefig(OUT_DIR / 'chart4_mar14_spike.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('  → chart4_mar14_spike.png')

print(f'\nAll 4 charts saved to {OUT_DIR}')
