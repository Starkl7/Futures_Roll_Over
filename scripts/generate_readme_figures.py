"""
Generate 3 figures for README Results section:
  Fig 1 — Combined net equity curve: V1 vs Z-score Ungated (10-lot, all 4 windows)
  Fig 2 — Per-window net P&L bar chart: V1 vs Ungated (with IS/OOS split)
  Fig 3 — Strategy comparison: avg_net/lot with 95% CI (ESTAR vs Ungated vs V1)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT    = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUTDIR  = ROOT / "notebooks" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

N_LOTS = 10
TC     = 8.04  # $/lot round-trip, same as result_summary.py

WINDOWS = [
    ("W1", "ESU4_ESZ4_20240912", "IS"),
    ("W2", "ESZ4_ESH5_20241212", "IS"),
    ("W3", "ESH5_ESM5_20250313", "OOS"),
    ("W4", "ESM5_ESU5_20250612", "OOS"),
]
SESSIONS = ["European", "US_RTH", "Post_close"]


def load_combined(window_dir: Path, gate: str) -> pd.DataFrame:
    """Concatenate session trades for one window/gate and sort by entry_time.
    Computes net_lot = gross_usd - TC, matching result_summary.py convention."""
    parts = []
    for sess in SESSIONS:
        p = window_dir / f"{sess}_{gate}" / "trades.parquet"
        if p.exists():
            df_s = pd.read_parquet(p, columns=["entry_time", "gross_usd"])
            df_s["net_lot"] = df_s["gross_usd"] - TC
            parts.append(df_s[["entry_time", "net_lot"]])
    if not parts:
        return pd.DataFrame(columns=["entry_time", "net_lot"])
    df = pd.concat(parts, ignore_index=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    return df.sort_values("entry_time").reset_index(drop=True)


# ── load all data ──────────────────────────────────────────────────────────────
data = {}
for label, wdir, role in WINDOWS:
    wd = RESULTS / wdir
    data[label] = {
        "V1":      load_combined(wd, "V1"),
        "Ungated": load_combined(wd, "Ungated"),
        "role":    role,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Combined net equity curve
# ══════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(11, 4.5))

colors = {"V1": "#2563EB", "Ungated": "#9CA3AF"}

for gate, color in colors.items():
    cum_pnl  = 0.0
    trade_no = 0
    xs, ys   = [0], [0]
    for label, wdir, role in WINDOWS:
        df = data[label][gate]
        for _, row in df.iterrows():
            trade_no += 1
            cum_pnl  += row["net_lot"] * N_LOTS
            xs.append(trade_no)
            ys.append(cum_pnl)
    ax1.plot(xs, ys, color=color, lw=1.8, label=gate)

# IS/OOS divider — after last IS trade
is_trades = sum(len(data[lbl][g]) for lbl, _, role in WINDOWS if role == "IS"
                for g in ["V1"])
ax1.axvline(is_trades, color="#6B7280", lw=1.0, ls="--", alpha=0.7)
ax1.text(is_trades + 2, ax1.get_ylim()[0] * 0.85, "OOS →",
         fontsize=8, color="#6B7280", va="top")
ax1.text(is_trades - 4, ax1.get_ylim()[0] * 0.85, "← IS",
         fontsize=8, color="#6B7280", va="top", ha="right")

# window boundaries
n = 0
for label, wdir, role in WINDOWS:
    n += len(data[label]["V1"])
    ax1.axvline(n, color="#E5E7EB", lw=0.8, ls=":")

# window labels
n = 0
for label, wdir, role in WINDOWS:
    n_w = len(data[label]["V1"])
    mid = n + n_w / 2
    ax1.text(mid, ax1.get_ylim()[1] * 0.95, label,
             ha="center", va="top", fontsize=8, color="#6B7280")
    n += n_w

ax1.axhline(0, color="black", lw=0.7, alpha=0.4)
ax1.set_xlabel("Trade # (sequential, W1 → W4)", fontsize=9)
ax1.set_ylabel("Cumulative Net P&L — 10-lot ($)", fontsize=9)
ax1.set_title("Combined Net Equity Curve — V1 vs Z-Score Ungated", fontsize=11, fontweight="bold")
ax1.legend(fontsize=9, framealpha=0.8)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
fig1.tight_layout()
fig1.savefig(OUTDIR / "readme_equity_curve.png", dpi=150, bbox_inches="tight")
plt.close(fig1)
print("Saved readme_equity_curve.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Per-window net P&L bar chart
# ══════════════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(8, 4))

bar_w = 0.35
win_labels = [lbl for lbl, _, _ in WINDOWS]
x = np.arange(len(win_labels))

v1_pnl      = [data[lbl]["V1"]["net_lot"].sum()      * N_LOTS for lbl, _, _ in WINDOWS]
ungated_pnl = [data[lbl]["Ungated"]["net_lot"].sum() * N_LOTS for lbl, _, _ in WINDOWS]

bars_u = ax2.bar(x - bar_w/2, ungated_pnl, bar_w, label="Z-Score Ungated",
                 color="#9CA3AF", edgecolor="white")
bars_v = ax2.bar(x + bar_w/2, v1_pnl,      bar_w, label="V1",
                 color="#2563EB", edgecolor="white")

# colour-code IS vs OOS
for i, (lbl, _, role) in enumerate(WINDOWS):
    alpha = 1.0 if role == "OOS" else 0.65
    bars_u[i].set_alpha(alpha)
    bars_v[i].set_alpha(alpha)

ax2.axhline(0, color="black", lw=0.7, alpha=0.4)
ax2.set_xticks(x)
ax2.set_xticklabels(win_labels, fontsize=10)
ax2.set_ylabel("Total Net P&L — 10-lot ($)", fontsize=9)
ax2.set_title("Per-Window Net P&L: V1 vs Z-Score Ungated\n(faded = IS, solid = OOS)", fontsize=10, fontweight="bold")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:,.0f}"))
ax2.legend(fontsize=9, framealpha=0.8)
fig2.tight_layout()
fig2.savefig(OUTDIR / "readme_window_pnl.png", dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Saved readme_window_pnl.png")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — avg_net/lot with 95% CI: ESTAR vs Ungated vs V1
# ══════════════════════════════════════════════════════════════════════════════
# ESTAR: hardcoded from ms_baseline.py output (no saved trade files)
ESTAR_STATS = {
    "IS":  {"mean": -4.36, "ci_lo": -4.58, "ci_hi": -4.15},
    "OOS": {"mean": -4.00, "ci_lo": -4.27, "ci_hi": -3.74},
}

def pool_stats(gate: str, windows: list) -> dict:
    """Compute mean and 95% t-CI for per-lot net_lot across a pool of windows."""
    parts = [data[lbl][gate] for lbl, _, _ in windows if len(data[lbl][gate]) > 0]
    if not parts:
        return {"mean": 0, "ci_lo": 0, "ci_hi": 0}
    vals = pd.concat(parts, ignore_index=True)["net_lot"].values
    n    = len(vals)
    mu   = vals.mean()
    se   = vals.std(ddof=1) / np.sqrt(n)
    from scipy import stats as sst
    t    = sst.t.ppf(0.975, df=n - 1)
    return {"mean": mu, "ci_lo": mu - t * se, "ci_hi": mu + t * se}

IS_wins  = [(lbl, w, r) for lbl, w, r in WINDOWS if r == "IS"]
OOS_wins = [(lbl, w, r) for lbl, w, r in WINDOWS if r == "OOS"]

stats_table = {
    "IS": {
        "ESTAR":   ESTAR_STATS["IS"],
        "Ungated": pool_stats("Ungated", IS_wins),
        "V1":      pool_stats("V1",      IS_wins),
    },
    "OOS": {
        "ESTAR":   ESTAR_STATS["OOS"],
        "Ungated": pool_stats("Ungated", OOS_wins),
        "V1":      pool_stats("V1",      OOS_wins),
    },
}

fig3, axes = plt.subplots(1, 2, figsize=(9, 4.5), sharey=True)
strats    = ["ESTAR", "Ungated", "V1"]
pal       = {"ESTAR": "#EF4444", "Ungated": "#9CA3AF", "V1": "#2563EB"}
x3        = np.arange(len(strats))

for ax, pool in zip(axes, ["IS", "OOS"]):
    means  = [stats_table[pool][s]["mean"]  for s in strats]
    ci_lo  = [stats_table[pool][s]["ci_lo"] for s in strats]
    ci_hi  = [stats_table[pool][s]["ci_hi"] for s in strats]
    yerr   = [[m - lo for m, lo in zip(means, ci_lo)],
              [hi - m for m, hi in zip(means, ci_hi)]]
    colors3 = [pal[s] for s in strats]
    bars3   = ax.bar(x3, means, 0.5, color=colors3, edgecolor="white",
                     yerr=yerr, capsize=5, error_kw={"elinewidth": 1.4,
                     "ecolor": "#374151", "capthick": 1.4})
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.set_xticks(x3)
    ax.set_xticklabels(strats, fontsize=10)
    ax.set_title(f"{pool} Pool\n(W{'1–2' if pool == 'IS' else '3–4'})", fontsize=10, fontweight="bold")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${v:.1f}"))

axes[0].set_ylabel("Avg Net P&L / lot ($)", fontsize=9)
fig3.suptitle("Strategy Comparison — Avg Net P&L per Lot with 95% CI",
              fontsize=11, fontweight="bold", y=1.02)
patches = [mpatches.Patch(color=pal[s], label=s) for s in strats]
fig3.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
            bbox_to_anchor=(0.5, -0.08), framealpha=0.8)
fig3.tight_layout()
fig3.savefig(OUTDIR / "readme_strategy_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig3)
print("Saved readme_strategy_comparison.png")
