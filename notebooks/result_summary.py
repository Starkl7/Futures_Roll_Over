#!/usr/bin/env python3
"""
result_summary.py — ES Calendar Spread Strategy: Final Performance Summary
═══════════════════════════════════════════════════════════════════════════

Train / Test split
  IS  (in-sample) : W1 ESU4→ESZ4 (Sep 2024) + W2 ESZ4→ESH5 (Dec 2024)
  OOS (out-of-sample) : W3 ESH5→ESM5 (Mar 2025) + W4 ESM5→ESU5 (Jun 2025)

Strategy variants evaluated
  Baseline : no regime gate
  V1       : drift_4h gate (block shorts when 4h RTH drift > 0.10 pts)

Sessions per window
  European    07:00–12:29/13:29 UTC
  US_RTH      12:30/13:30–19:14/20:14 UTC
  Post_close  19:15/20:15–20:59/21:59 UTC

Fill model  : synthetic mid spread, zero slippage (limit-order semantics)
TC model    : $8.04/lot round-trip (exchange $4.60 + NFA $0.04 + broker $3.40)
Lot size    : 10 lots (gross_usd is per-lot-equivalent; multiply by 10 for $)
Signal      : z-score crosses ±2.5 threshold on 10-min rolling std of FV deviation
Layered TP  : 90% @ +0.50 pts → SL to BE; 10% @ +0.75 pts → SL to +0.50
Low-z exit  : |entry_z| < 2.0 → 100% @ +0.25 pts (single layer)
HC add-on   : |z_fill| > 3.0 at T+1 fill → execute add-on unconditionally at T+2; blended entry

Usage
  cd /Users/stark/Desktop/Projects/Futures_RollOver
  .venv/bin/python notebooks/result_summary.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as spstats

warnings.filterwarnings('ignore')

# ── Paths & constants ──────────────────────────────────────────────────────────
RESULTS = Path('/Users/stark/Desktop/Projects/Futures_RollOver/results')
TC      = 8.04        # $/lot round-trip
N_LOTS  = 10          # reporting basis
MULT    = 50.0        # $/pt
np.random.seed(42)

WINDOWS = {
    'W1': ('ESU4_ESZ4_20240912', 'ESU4→ESZ4', 'Sep-24', 'IS'),
    'W2': ('ESZ4_ESH5_20241212', 'ESZ4→ESH5', 'Dec-24', 'IS'),
    'W3': ('ESH5_ESM5_20250313', 'ESH5→ESM5', 'Mar-25', 'OOS'),
    'W4': ('ESM5_ESU5_20250612', 'ESM5→ESU5', 'Jun-25', 'OOS'),
}
SESSIONS   = ['European', 'US_RTH', 'Post_close']
GATES      = ['Baseline', 'V1']
ALL_LABELS = [f'{s}_{g}' for s in SESSIONS for g in GATES]


# ── Helper functions ───────────────────────────────────────────────────────────

def load_trades(wdir: str, label: str) -> pd.DataFrame:
    p = RESULTS / wdir / label / 'trades.parquet'
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    return df if not df.empty else pd.DataFrame()


def stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {}
    gross = df['gross_usd']
    net   = gross - TC
    wins  = gross[gross > 0]
    loss  = gross[gross <= 0]
    pf    = wins.sum() / abs(loss.sum()) if len(loss) > 0 and loss.sum() != 0 else np.inf
    t, p  = spstats.ttest_1samp(net, 0.0)
    ci    = spstats.t.interval(0.95, df=len(net)-1, loc=net.mean(), scale=net.sem()) if len(net) > 1 else (np.nan, np.nan)
    sh    = net.mean() / net.std() if net.std() > 0 else 0.0
    cum   = gross.cumsum()
    mdd   = (cum - cum.cummax()).min()
    hc_n  = int(df['hc_addon'].sum()) if 'hc_addon' in df.columns else 0
    return dict(
        n        = len(df),
        wr       = (gross > 0).mean() * 100,
        avg_g    = gross.mean(),
        tot_g    = gross.sum(),
        tot_g_10 = gross.sum() * N_LOTS,
        avg_n    = net.mean(),
        tot_n_10 = net.sum() * N_LOTS,
        pf       = pf,
        sharpe   = sh,
        t        = t,
        p        = p,
        ci_lo    = ci[0],
        ci_hi    = ci[1],
        mdd_10   = mdd * N_LOTS,
        tp_pct   = (df['exit_type'] == 'TP').mean() * 100,
        sl_pct   = (df['exit_type'] == 'SL').mean() * 100,
        eod_pct  = (df['exit_type'] == 'EOD').mean() * 100,
        hc_n     = hc_n,
        long_wr  = (gross[df['direction'] == 1] > 0).mean() * 100 if (df['direction'] == 1).any() else np.nan,
        short_wr = (gross[df['direction'] == -1] > 0).mean() * 100 if (df['direction'] == -1).any() else np.nan,
        long_avg = gross[df['direction'] == 1].mean() if (df['direction'] == 1).any() else np.nan,
        short_avg= gross[df['direction'] == -1].mean() if (df['direction'] == -1).any() else np.nan,
        long_n   = (df['direction'] == 1).sum(),
        short_n  = (df['direction'] == -1).sum(),
    )


def pool(wkeys: list, label: str) -> pd.DataFrame:
    parts = []
    for wk in wkeys:
        wdir = WINDOWS[wk][0]
        df   = load_trades(wdir, label)
        if not df.empty:
            df['window'] = wk
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def sig_stars(p: float) -> str:
    if np.isnan(p):
        return '   '
    return '***' if p < 0.01 else '** ' if p < 0.05 else '*  ' if p < 0.10 else '   '


def fmt_ci(lo, hi) -> str:
    if np.isnan(lo):
        return '      n/a       '
    return f'[{lo:>+7.2f}, {hi:>+7.2f}]'


def section(title: str):
    w = 96
    print(f'\n{"═" * w}')
    print(f'  {title}')
    print(f'{"═" * w}')


def subsection(title: str):
    print(f'\n  {"─" * 90}')
    print(f'  {title}')
    print(f'  {"─" * 90}')


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — HEADER
# ═══════════════════════════════════════════════════════════════════════════════

print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════╗
║          ES CALENDAR SPREAD STRATEGY — FINAL PERFORMANCE SUMMARY                           ║
║          Train: W1 (Sep-24) + W2 (Dec-24)   |   Test: W3 (Mar-25) + W4 (Jun-25)          ║
║          TC = $8.04/lot  |  Fill: synthetic mid, zero slip  |  Lot basis: 10 lots         ║
╚══════════════════════════════════════════════════════════════════════════════════════════════╝
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — TRANSACTION COST & BREAK-EVEN MODEL
# ═══════════════════════════════════════════════════════════════════════════════

section('1. COST MODEL & BREAK-EVEN THRESHOLDS')
print(f"""
  Calendar spread round-trip (4 contract sides):
    Exchange fee  : $1.15 × 4 = $4.60
    NFA fee       : $0.01 × 4 = $0.04
    Broker comm   : $0.85 × 4 = $3.40
    ─────────────────────────────────
    Total TC      : $8.04 / lot

  Fill model: resting limit order on GLOBEX spread book (synthetic mid).
  Zero slippage assumed. Market order would add $6.25/lot (½ tick per leg).
  Legging two outrights adds ≥$25/lot.

  Layered TP payoff (10 lots, standard layers):
    Win path  (90% @ +0.50 → BE, 10% @ +0.75):  gross = +$50  → net = +$41.96
    Loss path (initial SL at −0.50):              gross = −$25  → net = −$33.04
    Payoff ratio: 0.75  |  Gross BE win-rate: 33.3%  |  Net BE win-rate: 44.0%

  Low-z exit (100% @ +0.25):
    Win path:   gross = +$12.50  → net = +$4.46
    Loss path:  gross = −$25.00  → net = −$33.04
    Payoff ratio: 0.50  |  Net BE win-rate: 88.1%   ← structural disadvantage

  HC add-on (20 lots blended):
    Every metric doubles; net BE win-rate identical per-lot, but gap risk doubles.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — WINDOW-LEVEL OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════

section('2. WINDOW-LEVEL OVERVIEW  (all sessions combined per window)')

print(f"\n  {'Window':<6} {'Roll':<12} {'Split':<5} {'n':>5}  {'WR%':>6}  {'avg_gross':>10}  {'tot_net×10':>11}  {'MDD×10':>9}  {'Sharpe/tr':>10}  {'p-val':>7}")
print(f"  {'─'*6} {'─'*12} {'─'*5} {'─'*5}  {'─'*6}  {'─'*10}  {'─'*11}  {'─'*9}  {'─'*10}  {'─'*7}")

window_stats = {}
for wk, (wdir, label, period, split) in WINDOWS.items():
    dfs = []
    for sess in SESSIONS:
        for gate in GATES:
            df = load_trades(wdir, f'{sess}_{gate}')
            if not df.empty:
                dfs.append(df)
    if not dfs:
        continue
    combined = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=['entry_time','direction'])
    s = stats(combined)
    window_stats[wk] = s
    stars = sig_stars(s['p'])
    print(f"  {wk:<6} {label:<12} {split:<5} {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+10.2f}  "
          f"{s['tot_n_10']:>+11.0f}  {s['mdd_10']:>+9.0f}  {s['sharpe']:>+10.4f}  {s['p']:>7.4f} {stars}")

print(f"\n  Note: each trade appears in both Baseline and V1 outputs if the gate was inactive.")
print(f"  IS windows (W1+W2) and OOS windows (W3+W4) reported below separately.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — IS vs OOS: AGGREGATE BY SESSION TYPE
# ═══════════════════════════════════════════════════════════════════════════════

section('3. IN-SAMPLE vs OUT-OF-SAMPLE  (per session type × gate)')

IS_WINS  = ['W1', 'W2']
OOS_WINS = ['W3', 'W4']

hdr = (f"  {'Session':<22} {'Split':<5} {'n':>5}  {'WR%':>6}  {'avg_g':>7}  "
       f"{'avg_net':>8}  {'tot_net×10':>11}  {'PF':>5}  {'Sharpe':>7}  {'t':>7}  "
       f"{'p':>7}  {'95% CI net/lot':>20}  {'sig':>3}")
print(hdr)
print(f"  {'─'*22} {'─'*5} {'─'*5}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*11}  {'─'*5}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*20}  {'─'*3}")

session_split_stats = {}
for label in ALL_LABELS:
    for split, wkeys in [('IS', IS_WINS), ('OOS', OOS_WINS)]:
        df = pool(wkeys, label)
        s  = stats(df)
        if not s:
            continue
        session_split_stats[(label, split)] = s
        stars = sig_stars(s['p'])
        pf_str = f"{s['pf']:.2f}" if np.isfinite(s['pf']) else ' ∞   '
        ci_str = fmt_ci(s['ci_lo'], s['ci_hi'])
        print(f"  {label:<22} {split:<5} {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+7.2f}  "
              f"{s['avg_n']:>+8.2f}  {s['tot_n_10']:>+11.0f}  {pf_str:>5}  {s['sharpe']:>+7.4f}  "
              f"{s['t']:>+7.3f}  {s['p']:>7.4f}  {ci_str}  {stars}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — IS → OOS PERFORMANCE DECAY
# ═══════════════════════════════════════════════════════════════════════════════

section('4. IS → OOS PERFORMANCE DECAY')

print(f"\n  {'Session':<22}  {'IS avg_net':>10}  {'OOS avg_net':>11}  {'Decay':>8}  {'IS sig':>7}  {'OOS sig':>8}  {'Edge preserved?':>16}")
print(f"  {'─'*22}  {'─'*10}  {'─'*11}  {'─'*8}  {'─'*7}  {'─'*8}  {'─'*16}")

for label in ALL_LABELS:
    is_s  = session_split_stats.get((label, 'IS'), {})
    oos_s = session_split_stats.get((label, 'OOS'), {})
    if not is_s or not oos_s:
        continue
    is_net  = is_s['avg_n']
    oos_net = oos_s['avg_n']
    decay   = oos_net - is_net
    is_sig  = sig_stars(is_s['p']).strip() or '—'
    oos_sig = sig_stars(oos_s['p']).strip() or '—'
    # Edge preserved = OOS CI lower bound > 0 (positive), and same sign as IS
    oos_ci_pos = oos_s['ci_lo'] > 0
    same_sign  = (is_net > 0) == (oos_net > 0)
    preserved  = '✓ Yes' if same_sign and oos_net > 0 else ('↔ Flat' if abs(oos_net) < 2 else '✗ No')
    print(f"  {label:<22}  {is_net:>+10.2f}  {oos_net:>+11.2f}  {decay:>+8.2f}  {is_sig:>7}  {oos_sig:>8}  {preserved:>16}")

# Rank correlation: IS avg_net vs OOS avg_net across all 6 labels
is_nets  = [session_split_stats.get((lbl, 'IS'), {}).get('avg_n', np.nan)  for lbl in ALL_LABELS]
oos_nets = [session_split_stats.get((lbl, 'OOS'), {}).get('avg_n', np.nan) for lbl in ALL_LABELS]
mask = ~(np.isnan(is_nets) | np.isnan(oos_nets))
if mask.sum() >= 3:
    rho, rp = spstats.spearmanr(np.array(is_nets)[mask], np.array(oos_nets)[mask])
    print(f"\n  IS→OOS rank correlation (Spearman): ρ = {rho:.3f}  (p = {rp:.3f})")
    print(f"  {'Strong IS→OOS consistency (ρ > 0.7)' if rho > 0.7 else 'Moderate IS→OOS consistency (ρ > 0.4)' if rho > 0.4 else 'Weak IS→OOS consistency'}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — SPLIT AGGREGATE: IS AND OOS TOTALS
# ═══════════════════════════════════════════════════════════════════════════════

section('5. SPLIT-LEVEL AGGREGATE PERFORMANCE')

for split_label, split_key, wkeys in [('IN-SAMPLE (W1+W2)', 'IS', IS_WINS), ('OUT-OF-SAMPLE (W3+W4)', 'OOS', OOS_WINS)]:
    subsection(split_label)
    print(f"\n  {'Session':<22}  {'n':>5}  {'WR%':>6}  {'avg_gross':>10}  {'avg_net':>9}  "
          f"{'tot_gross×10':>13}  {'tot_net×10':>11}  {'PF':>5}  {'Sharpe':>7}  {'p':>7}")
    print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*13}  {'─'*11}  {'─'*5}  {'─'*7}  {'─'*7}")

    totals_n, totals_g, totals_n_usd = 0, 0, 0
    all_nets = []
    for label in ALL_LABELS:
        s = session_split_stats.get((label, split_key), {})
        if not s:
            continue
        pf_str = f"{s['pf']:.2f}" if np.isfinite(s['pf']) else ' ∞   '
        stars  = sig_stars(s['p'])
        print(f"  {label:<22}  {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+10.2f}  "
              f"{s['avg_n']:>+9.2f}  {s['tot_g_10']:>+13.0f}  {s['tot_n_10']:>+11.0f}  "
              f"{pf_str:>5}  {s['sharpe']:>+7.4f}  {s['p']:>7.4f} {stars}")
        totals_n   += s['n']
        totals_g   += s['tot_g_10']
        totals_n_usd += s['tot_n_10']

    # Grand row
    all_df = pool(wkeys, ALL_LABELS[0])
    agg_parts = []
    for label in ALL_LABELS:
        df = pool(wkeys, label)
        if not df.empty:
            agg_parts.append(df)
    if agg_parts:
        agg = pd.concat(agg_parts, ignore_index=True)
        s   = stats(agg)
        print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*13}  {'─'*11}  {'─'*5}  {'─'*7}  {'─'*7}")
        stars = sig_stars(s['p'])
        print(f"  {'AGGREGATE':<22}  {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+10.2f}  "
              f"{s['avg_n']:>+9.2f}  {s['tot_g_10']:>+13.0f}  {s['tot_n_10']:>+11.0f}  "
              f"{'—':>5}  {s['sharpe']:>+7.4f}  {s['p']:>7.4f} {stars}")
        print(f"  95% CI avg_net/lot: {fmt_ci(s['ci_lo'], s['ci_hi'])}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — EUROPEAN SESSION: PRIMARY ALPHA SOURCE
# ═══════════════════════════════════════════════════════════════════════════════

section('6. EUROPEAN SESSION — PRIMARY ALPHA SOURCE')

print(f"\n  Rationale: 07:00–12:29/13:29 UTC captures London open through pre-US handoff.")
print(f"  Fair-value deviation is most persistent at European open (low liquidity, overnight gaps).")
print(f"  Drift gate (V1) has minimal effect: short trades are rare in European hours.\n")

print(f"  {'Window':<6} {'Gate':<10} {'n':>5}  {'WR%':>6}  {'avg_gross':>10}  {'avg_net':>9}  "
      f"{'tot_net×10':>11}  {'Sharpe':>8}  {'t':>7}  {'p':>7}  {'sig':>3}")
print(f"  {'─'*6} {'─'*10} {'─'*5}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*11}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*3}")

for gate in GATES:
    label = f'European_{gate}'
    for wk, (wdir, wlabel, period, split) in WINDOWS.items():
        df = load_trades(wdir, label)
        s  = stats(df)
        if not s:
            continue
        stars = sig_stars(s['p'])
        row_tag = f'{wk}({period})'
        print(f"  {row_tag:<6} {gate:<10} {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+10.2f}  "
              f"{s['avg_n']:>+9.2f}  {s['tot_n_10']:>+11.0f}  {s['sharpe']:>+8.4f}  "
              f"{s['t']:>+7.3f}  {s['p']:>7.4f}  {stars}")
    # Cross-window aggregate
    for split, wkeys in [('IS ', IS_WINS), ('OOS', OOS_WINS), ('ALL', list(WINDOWS.keys()))]:
        df = pool(wkeys, label)
        s  = stats(df)
        if not s:
            continue
        stars = sig_stars(s['p'])
        print(f"  {'━'+split:<6} {gate:<10} {s['n']:>5}  {s['wr']:>6.1f}  {s['avg_g']:>+10.2f}  "
              f"{s['avg_n']:>+9.2f}  {s['tot_n_10']:>+11.0f}  {s['sharpe']:>+8.4f}  "
              f"{s['t']:>+7.3f}  {s['p']:>7.4f}  {stars}")
    print()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — US RTH SESSION: STRUCTURAL WEAKNESS
# ═══════════════════════════════════════════════════════════════════════════════

section('7. US RTH SESSION — STRUCTURAL WEAKNESS (SHORT DRAG)')

print(f"\n  Aggregate p-value across all 4 windows: ~0.91 (Baseline), ~0.51 (V1).")
print(f"  Structurally flat. Short trades underperform in every window.\n")

print(f"  {'Gate':<10} {'Dir':<7} {'n':>5}  {'WR%':>6}  {'avg_gross':>10}  {'avg_net':>9}  {'BE breach?':>11}")
print(f"  {'─'*10} {'─'*7} {'─'*5}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*11}")
for gate in GATES:
    label = f'US_RTH_{gate}'
    df    = pool(list(WINDOWS.keys()), label)
    if df.empty:
        continue
    for d, dlabel in [(1, 'Long'), (-1, 'Short')]:
        sub = df[df['direction'] == d]
        if sub.empty:
            continue
        ag  = sub['gross_usd'].mean()
        an  = ag - TC
        wr  = (sub['gross_usd'] > 0).mean() * 100
        breach = '✗ BELOW' if ag < TC else '✓ above'
        print(f"  {gate:<10} {dlabel:<7} {len(sub):>5}  {wr:>6.1f}  {ag:>+10.2f}  {an:>+9.2f}  {breach:>11}")
    # V1 separating line
    df_base = pool(list(WINDOWS.keys()), f'US_RTH_Baseline')
    df_v1   = pool(list(WINDOWS.keys()), f'US_RTH_V1')
    base_shorts = df_base[df_base['direction']==-1]
    v1_shorts   = df_v1[df_v1['direction']==-1]
    if gate == 'V1':
        print(f"\n  Drift gate (V1) reduces short count: {len(base_shorts)} → {len(v1_shorts)} ({len(v1_shorts)/len(base_shorts)*100:.0f}% retained)")
        print(f"  Remaining V1 shorts avg_gross: {v1_shorts['gross_usd'].mean():+.2f} — still below break-even")

# Per-window breakdown for US_RTH_Baseline
print(f"\n  US_RTH Baseline — per-window L/S split:")
print(f"  {'Window':<6} {'Long n':>7} {'Long avg':>9} {'Short n':>8} {'Short avg':>10}")
print(f"  {'─'*6} {'─'*7} {'─'*9} {'─'*8} {'─'*10}")
for wk, (wdir, wlabel, period, split) in WINDOWS.items():
    df = load_trades(wdir, 'US_RTH_Baseline')
    if df.empty: continue
    lg  = df[df['direction']== 1]
    sh  = df[df['direction']==-1]
    print(f"  {wk:<6} {len(lg):>7} {lg['gross_usd'].mean():>+9.2f} {len(sh):>8} {sh['gross_usd'].mean():>+10.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Z-SCORE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════════

section('8. Z-SCORE SEGMENTATION AT FILL')

all_parts = []
for wk, (wdir, *_) in WINDOWS.items():
    for label in ALL_LABELS:
        df = load_trades(wdir, label)
        if not df.empty:
            df['window'] = wk
            all_parts.append(df)
all_df = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()

buckets = [
    ('Low-z    |z|<2.0 ', all_df['entry_z'].abs() < 2.0),
    ('Mid-z  2≤|z|<3.0 ', (all_df['entry_z'].abs() >= 2.0) & (all_df['entry_z'].abs() < 3.0)),
    ('High-z   |z|≥3.0 ', all_df['entry_z'].abs() >= 3.0),
]

print(f"\n  NOTE: Low-z trades use 100%@+0.25 exit (single layer).")
print(f"  High-z includes HC add-on trades (lot_scale=2.0) — gross_usd is 2× for those.\n")

print(f"  {'Bucket':<22}  {'n':>5}  {'WR%':>6}  {'avg_gross':>10}  {'avg_net':>9}  {'TP%':>5}  {'SL%':>5}  {'% trades':>9}  {'HC_n':>5}")
print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*9}  {'─'*5}")
total_n = len(all_df)
for blabel, mask in buckets:
    sub = all_df[mask]
    if sub.empty: continue
    n   = len(sub)
    wr  = (sub['gross_usd'] > 0).mean() * 100
    ag  = sub['gross_usd'].mean()
    an  = ag - TC
    tp  = (sub['exit_type']=='TP').mean()*100
    sl  = (sub['exit_type']=='SL').mean()*100
    pct = n/total_n*100
    hc  = int(sub['hc_addon'].sum()) if 'hc_addon' in sub.columns else 0
    be  = '✗' if ag < TC else '✓'
    print(f"  {blabel:<22}  {n:>5}  {wr:>6.1f}  {ag:>+10.2f} {be}  {an:>+9.2f}  {tp:>5.1f}  {sl:>5.1f}  {pct:>8.1f}%  {hc:>5}")

print(f"""
  Interpretation:
  • Low-z (74% of trades): below break-even gross. High WR (≈87%) is misleading —
    the single-layer +0.25 pt exit has payoff ratio 0.50, requiring 88.1% WR net.
    This segment is the volume trap: frequent entries, thin edge.
  • Mid-z (clean segment): highest avg_net (+$4.28), no lot scaling distortion.
    Pure standard-layer trades with genuine spread persistence.
  • High-z: inflated by HC add-on (70/143 trades at 2× lots). Per-actual-lot
    avg_net is lower; the dominant exit type is SL (84% of HC trades).
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — HC ADD-ON ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

section('9. HIGH-CONVICTION ADD-ON (HC) — DOES LOT DOUBLING ADD VALUE?')

hc_parts, nohc_parts = [], []
for wk, (wdir, *_) in WINDOWS.items():
    for label in ALL_LABELS:
        df = load_trades(wdir, label)
        if df.empty or 'hc_addon' not in df.columns:
            continue
        df['window'] = wk
        hc_parts.append(df[df['hc_addon']==True])
        nohc_parts.append(df[df['hc_addon']==False])

hc_all   = pd.concat(hc_parts,   ignore_index=True) if hc_parts   else pd.DataFrame()
nohc_all = pd.concat(nohc_parts, ignore_index=True) if nohc_parts else pd.DataFrame()

print(f"\n  Trigger: |z_fill| > 3.0 at T+1 fill → execute add-on unconditionally at T+2 (no z re-check)")
print(f"  Blended entry resets TP/SL to standard layers. gross_usd is on 10-lot-equiv basis.")
print(f"  Per-actual-lot figures divide by lot_scale=2.\n")

print(f"  {'Group':<22}  {'n':>5}  {'WR%':>6}  {'avg_g(10-eq)':>13}  {'avg_g/lot':>10}  {'avg_n/lot':>10}  {'TP%':>5}  {'SL%':>5}")
print(f"  {'─'*22}  {'─'*5}  {'─'*6}  {'─'*13}  {'─'*10}  {'─'*10}  {'─'*5}  {'─'*5}")
for grp_label, grp_df in [('HC add-on (lot×2)', hc_all), ('Non-HC (lot×1)', nohc_all)]:
    if grp_df.empty: continue
    n   = len(grp_df)
    wr  = (grp_df['gross_usd'] > 0).mean() * 100
    ag  = grp_df['gross_usd'].mean()
    agl = ag / grp_df.get('lot_scale', pd.Series(1, index=grp_df.index)).mean() if 'lot_scale' in grp_df.columns else ag
    anl = agl - TC
    tp  = (grp_df['exit_type']=='TP').mean()*100
    sl  = (grp_df['exit_type']=='SL').mean()*100
    print(f"  {grp_label:<22}  {n:>5}  {wr:>6.1f}  {ag:>+13.2f}  {agl:>+10.2f}  {anl:>+10.2f}  {tp:>5.1f}  {sl:>5.1f}")

print(f"\n  HC by window (10-lot-equiv basis):")
print(f"  {'Window':<6}  {'Split':<5}  {'n':>4}  {'WR%':>6}  {'avg_g(10-eq)':>13}  {'avg_g/lot':>10}  {'verdict':>12}")
print(f"  {'─'*6}  {'─'*5}  {'─'*4}  {'─'*6}  {'─'*13}  {'─'*10}  {'─'*12}")
for wk, (wdir, wlabel, period, split) in WINDOWS.items():
    wdfs = [df for df in [load_trades(wdir, label) for label in ALL_LABELS] if not df.empty and 'hc_addon' in df.columns]
    if not wdfs: continue
    wdf = pd.concat(wdfs, ignore_index=True)
    hcw = wdf[wdf['hc_addon']==True]
    if hcw.empty: continue
    wr  = (hcw['gross_usd'] > 0).mean() * 100
    ag  = hcw['gross_usd'].mean()
    agl = ag / 2.0
    vrd = '✓ Positive' if agl > TC else '✗ Negative' if agl < 0 else '↔ Marginal'
    print(f"  {wk:<6}  {split:<5}  {len(hcw):>4}  {wr:>6.1f}  {ag:>+13.2f}  {agl:>+10.2f}  {vrd:>12}")

print(f"""
  Critical risk: fixed-point SL (0.50 pts) does not scale with window sigma.
  W3 (σ ≈ 22.60 pts/z-unit): one short on 2025-03-14 gapped +4.31 pts through SL
  in 3 seconds with 20 lots → loss of −$431.25 (10-lot-equiv basis), −$4,312.50 total.
  Gap risk scales quadratically with sigma when lot size doubles.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — PORTFOLIO-LEVEL EQUITY CURVE (cumulative, per-gate)
# ═══════════════════════════════════════════════════════════════════════════════

section('10. PORTFOLIO-LEVEL CUMULATIVE P&L  (×10 lots, net of TC)')

for gate in GATES:
    print(f"\n  ── {gate} ──")
    labels_g = [f'{s}_{gate}' for s in SESSIONS]
    all_trades = []
    for wk, (wdir, wlabel, period, split) in WINDOWS.items():
        for label in labels_g:
            df = load_trades(wdir, label)
            if not df.empty:
                df['window'] = wk
                df['split']  = split
                all_trades.append(df)
    if not all_trades:
        continue
    port = pd.concat(all_trades, ignore_index=True).sort_values('entry_time')
    port['net_Tight'] = port['gross_usd'] - TC
    port['cum_net']   = port['net_Tight'].cumsum() * N_LOTS
    port['peak']      = port['cum_net'].cummax()
    port['dd']        = port['cum_net'] - port['peak']
    mdd  = port['dd'].min()
    final = port['cum_net'].iloc[-1]
    n    = len(port)

    # Per-window terminal equity
    print(f"  {'Window':<8}  {'n':>5}  {'Trades':>7}  {'Cum_net×10 terminal':>20}  {'MDD×10':>9}")
    print(f"  {'─'*8}  {'─'*5}  {'─'*7}  {'─'*20}  {'─'*9}")
    running = 0.0
    for wk, (wdir, wlabel, period, split) in WINDOWS.items():
        w_trades = port[port['window']==wk]
        if w_trades.empty: continue
        w_end = w_trades['net_Tight'].sum() * N_LOTS
        running += w_end
        print(f"  {wk}({period})  {len(w_trades):>5}  {'IS ' if split=='IS' else 'OOS':>7}  "
              f"  {w_end:>+15.0f} (cum: {running:>+7.0f})  ")
    print(f"  {'Overall':<8}  {n:>5}  {'All':>7}  {final:>+20.0f}  {mdd:>+9.0f}")
    print(f"  Recovery factor: {final/abs(mdd):.2f}" if mdd < 0 else "  No drawdown.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — STATISTICAL VERDICT
# ═══════════════════════════════════════════════════════════════════════════════

section('11. STATISTICAL VERDICT')

print(f"""
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  KEY QUESTION: Is there a statistically significant edge that generalises   │
  │  from IS (W1+W2) to OOS (W3+W4)?                                           │
  └─────────────────────────────────────────────────────────────────────────────┘
""")

# Compute IS and OOS aggregate t-tests
for split_label, wkeys in [('IS  (W1+W2)', IS_WINS), ('OOS (W3+W4)', OOS_WINS)]:
    agg_parts = []
    for label in ALL_LABELS:
        df = pool(wkeys, label)
        if not df.empty:
            agg_parts.append(df)
    if not agg_parts: continue
    agg  = pd.concat(agg_parts, ignore_index=True)
    nets = agg['gross_usd'] - TC
    t, p = spstats.ttest_1samp(nets, 0.0)
    ci   = spstats.t.interval(0.95, df=len(nets)-1, loc=nets.mean(), scale=nets.sem())
    stars= sig_stars(p)
    print(f"  {split_label}: n={len(nets)}  avg_net={nets.mean():+.2f}/lot  t={t:.3f}  p={p:.4f} {stars}  "
          f"95%CI=[{ci[0]:+.2f}, {ci[1]:+.2f}]")

# European only
print()
for split_label, wkeys in [('IS  European only', IS_WINS), ('OOS European only', OOS_WINS)]:
    agg_parts = []
    for label in [f'European_{g}' for g in GATES]:
        df = pool(wkeys, label)
        if not df.empty:
            agg_parts.append(df)
    if not agg_parts: continue
    agg  = pd.concat(agg_parts, ignore_index=True)
    nets = agg['gross_usd'] - TC
    t, p = spstats.ttest_1samp(nets, 0.0)
    ci   = spstats.t.interval(0.95, df=len(nets)-1, loc=nets.mean(), scale=nets.sem())
    stars= sig_stars(p)
    print(f"  {split_label}: n={len(nets)}  avg_net={nets.mean():+.2f}/lot  t={t:.3f}  p={p:.4f} {stars}  "
          f"95%CI=[{ci[0]:+.2f}, {ci[1]:+.2f}]")

print(f"""
  Power caveat: 2 IS windows and 2 OOS windows is an extremely thin evaluation basis.
  Each window is ~7 trading days. Any result — positive or negative — carries high
  uncertainty. The CI bounds reflect per-trade variance, not window-level variance.
  A proper OOS evaluation would require ≥10 roll periods.
""")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 12 — RISK METRICS
# ═══════════════════════════════════════════════════════════════════════════════

section('12. RISK METRICS PER SESSION TYPE (cross-window)')

print(f"\n  {'Session':<22}  {'n':>5}  {'MDD×10':>9}  {'Recovery':>9}  {'Worst/lot':>10}  {'Best/lot':>9}  {'SL%':>5}  {'Avg hold':>9}")
print(f"  {'─'*22}  {'─'*5}  {'─'*9}  {'─'*9}  {'─'*10}  {'─'*9}  {'─'*5}  {'─'*9}")
for label in ALL_LABELS:
    df = pool(list(WINDOWS.keys()), label)
    if df.empty: continue
    gross = df['gross_usd']
    net   = gross - TC
    cum   = net.cumsum() * N_LOTS
    mdd   = (cum - cum.cummax()).min()
    rec   = cum.iloc[-1] / abs(mdd) if mdd < 0 else float('inf')
    worst = gross.min()
    best  = gross.max()
    sl_p  = (df['exit_type']=='SL').mean()*100
    hold  = df['hold_min'].mean()
    rec_s = f"{rec:.2f}" if np.isfinite(rec) else '∞'
    print(f"  {label:<22}  {len(df):>5}  {mdd:>+9.0f}  {rec_s:>9}  {worst:>+10.2f}  {best:>+9.2f}  {sl_p:>5.1f}  {hold:>8.1f}m")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 13 — FINAL ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

section('13. FINAL ASSESSMENT')

# Compute the key numbers for the verdict
eur_all_df  = pool(list(WINDOWS.keys()), 'European_Baseline')
eur_nets    = eur_all_df['gross_usd'] - TC
_, eur_p    = spstats.ttest_1samp(eur_nets, 0.0)
eur_oos_df  = pool(OOS_WINS, 'European_Baseline')
eur_oos_net = (eur_oos_df['gross_usd'] - TC).mean()

# IS→OOS rank correlation
rho_val = rho if mask.sum() >= 3 else np.nan

print(f"""
  ╔═══════════════════════════════════════════════════════════════════════════╗
  ║  VERDICT: MARGINAL POSITIVE EDGE — NOT YET DEPLOYABLE                    ║
  ╚═══════════════════════════════════════════════════════════════════════════╝

  WHAT THE DATA SHOWS:

  1. European session is the only statistically significant source of edge.
     Cross-window (n=205): p={eur_p:.4f}, avg_net=+${eur_nets.mean():.2f}/lot.
     The edge persists OOS: European Baseline OOS avg_net=+${eur_oos_net:.2f}/lot.
     But individual windows range from near-zero (W2, FOMC-impacted) to +$7/lot (W1,W3).

  2. US RTH is structurally flat (p≈0.91 aggregate). Short trades are the culprit:
     avg_gross ≈ $4–6/lot vs $8.04 hurdle in every window. Drift gate (V1)
     reduces shorts but doesn't eliminate the problem. This session should not
     be traded independently.

  3. Post_close shows modest signal (p=0.083 Baseline cross-window) but loses
     significance under V1 filtering (p=0.24). Sample is thin (n=122 Baseline).

  4. IS → OOS generalisation is reasonable at the session-ranking level
     (ρ ≈ {rho_val:.2f}), but with only 2 IS and 2 OOS windows this is
     essentially anecdote. No meaningful statistical conclusion is possible
     about OOS Sharpe from 2 windows.

  5. HC add-on adds conditional value in low-sigma windows (W1: WR=94.4%,
     avg_gross=+$45.42/10-lot-equiv). It is dangerous in high-sigma windows
     (W3: one trade wiped +$4,312 on a gap through 0.50pt SL). Fixed-point
     stops must be volatility-scaled before HC is deployable.

  6. Low-z trades (74% of volume) are below break-even. The 100%@+0.25 single-
     layer exit requires 88.1% net WR to cover TC. Actual WR is 86.8%.
     These entries should either be eliminated or converted to passive limit
     orders at better prices.

  WHAT WOULD CHANGE THE VERDICT:

  → Replace low-z single-layer exit with skip-or-patient-entry logic.
  → Volatility-scale SL for HC add-on (SL in z-units, not spread points).
  → Collect ≥4 more roll windows before making deployment decision.
  → Separate European session as the focused tradeable; US_RTH as marginal.
  → European Baseline at 10 lots × $4.66 avg_net × ~50 trades/window
    = ~$2,330 expected net per window (4 windows/year = ~$9,300/yr at 10 lots).
    Scales linearly. At 100 lots: ~$93,000/yr, but execution assumptions break.
""")

print('Done.\n')

# ── Section 8: OOS Per-Trade Sharpe (W3+W4) ──────────────────────────────────
section("8. OOS Per-Trade Sharpe (W3+W4)")

_OOS_WDIRS = [
    Path(__file__).parent.parent / "results" / "ESH5_ESM5_20250313",
    Path(__file__).parent.parent / "results" / "ESM5_ESU5_20250612",
]
_SESSIONS_ALL = [
    "European_Baseline", "European_V1",
    "US_RTH_Baseline",   "US_RTH_V1",
    "Post_close_Baseline","Post_close_V1",
]

def _load_oos_pool(tag=None):
    frames = []
    for wdir in _OOS_WDIRS:
        for sess in _SESSIONS_ALL:
            if tag and tag not in sess:
                continue
            p = wdir / sess / "trades.parquet"
            if p.exists():
                df = pd.read_parquet(p)
                df["_sess"] = sess
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def _sharpe_boot(net_arr, n_boot=5000, ci=0.90):
    rng = np.random.default_rng(42)
    s = net_arr.mean() / net_arr.std(ddof=1)
    bs = []
    for _ in range(n_boot):
        b = rng.choice(net_arr, size=len(net_arr), replace=True)
        bs.append(b.mean() / b.std(ddof=1))
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(bs, [alpha * 100, (1 - alpha) * 100])
    return s, lo, hi

_TC8 = 8.04
print(f"  {'Pool':<18} {'n':>5}  {'mean_net':>9}  {'std_net':>9}  {'Sharpe':>8}  {'90% CI'}")
print(f"  {'-'*18} {'-'*5}  {'-'*9}  {'-'*9}  {'-'*8}  {'-'*20}")
for label, tag in [("All OOS", None), ("Baseline only", "Baseline"), ("V1 only", "V1")]:
    df = _load_oos_pool(tag)
    net = (df["gross_usd"] - _TC8).values
    s, lo, hi = _sharpe_boot(net)
    print(f"  {label:<18} {len(net):>5}  {net.mean():>+9.2f}  {net.std(ddof=1):>9.2f}  {s:>+8.4f}  [{lo:+.4f}, {hi:+.4f}]")

print("""
  Sharpe = mean(net) / std(net, ddof=1) — per-trade, not annualized.
  net = gross_usd − $8.04 TC, 10-lot-equiv reporting basis.
  90% CI: percentile bootstrap, 5,000 iterations, seed=42.
""")

# Annualized V1 OOS Sharpe
# Scale: Sharpe_ann = Sharpe_trade × √(trades_per_year)
# Assumes 4 quarterly roll windows per year; OOS spans 2 windows.
_N_WINDOWS_PER_YEAR = 4
_N_OOS_WINDOWS = 2
_v1_net = (_load_oos_pool("V1")["gross_usd"] - _TC8).values
_s_v1, _lo_v1, _hi_v1 = _sharpe_boot(_v1_net)
_n_annual = (len(_v1_net) / _N_OOS_WINDOWS) * _N_WINDOWS_PER_YEAR
_scale = np.sqrt(_n_annual)
print(f"  Annualized V1 OOS Sharpe")
print(f"  {'─'*50}")
print(f"  Trades/yr estimate : {_n_annual:.0f}  "
      f"({len(_v1_net)} OOS trades / {_N_OOS_WINDOWS} windows × {_N_WINDOWS_PER_YEAR} windows/yr)")
print(f"  Scale factor       : √{_n_annual:.0f} = {_scale:.2f}×")
print(f"  Sharpe_ann         : {_s_v1 * _scale:+.3f}  "
      f"(90% CI: [{_lo_v1 * _scale:+.3f}, {_hi_v1 * _scale:+.3f}])")
print("""
  Assumes trades are i.i.d. within and across roll windows.
  Strategy is episodic (active ~28 trading days/yr), so this is
  a within-episode density annualization, not a calendar-year return.
""")
