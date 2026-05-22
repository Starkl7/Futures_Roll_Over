# ES Futures Roll Analysis — Project Workflow

This document describes the **as-built** research pipeline for the ES Futures Calendar Spread Mean-Reversion project. It supersedes the original planning spec.

---

## Repository Structure (as built)

```
src/
  strategy.py              # Core engine: StrategyParams, simulate(), compute_stats(), bootstrap_ci()

scripts/
  ms_baseline.py           # Monoyios & Sarno (2002) ESTAR external baseline
  run_sessions.py          # Multi-window, multi-session backtest runner (all gates)
  run_backtest.py          # Single-window backtest entrypoint
  result_summary.py        # IS/OOS performance report with bootstrap CIs
  databento_cost_estimate.py  # Phase 2a: cost estimation before data pull
  databento_datapull.py    # Phase 2b: tick data pull (idempotent)

notebooks/
  01_eda.ipynb             # W1 exploratory data analysis
  02_signal_analysis.ipynb # Five-signal evaluation
  03_zscore_simulation.py  # Z-score parameter sweep
  04_threshold_analysis.py # Entry/exit threshold optimisation
  05_robustness_globex.py  # Globex session robustness check
  07_tearsheet.py          # W1 performance tearsheet
  08_fomc_trend.py         # FOMC event-day analysis
  09-13_...                # Additional window and signal analyses
  results_dashboard.ipynb  # Portfolio equity curves and MDD visualisation
  supplementary/           # 36 publication-quality analysis notebooks
    config.py              # Shared paths and window definitions
    generate_notebooks.py  # Script to regenerate all supplementary notebooks
    A1-H3                  # Performance, FV deviation, OU dynamics, trade analytics,
                           # bid-ask, spikes, volume migration, pre-roll context

data/                      # Gitignored; on /Volumes/SEAGATE/Databento_Futures/
results/                   # Gitignored; per-window subdirectories
reports/                   # Gitignored; PDF and markdown research outputs
```

---

## Data

- **Source:** Databento `GLBX.MDP3`, `mbp-10` schema
- **Size:** ~25–35 GB per roll window
- **Storage:** `/Volumes/SEAGATE/Databento_Futures/`
- **Processing:** Resampled to 1-second RTH bars; sessions defined by UTC timestamps
- **Expiry timestamps:** Pulled from Databento `definition` schema (free, cached permanently)
- **Interest rate:** SOFR pulled from FRED API (free, no key required)

---

## Roll Windows

| Window | Front | Back | Roll Date | Role |
|--------|-------|------|-----------|------|
| W1 | ESU4 | ESZ4 | Sep 16, 2024 | In-sample |
| W2 | ESZ4 | ESH5 | Dec 16, 2024 | In-sample |
| W3 | ESH5 | ESM5 | Mar 17, 2025 | Out-of-sample |
| W4 | ESM5 | ESU5 | Jun 16, 2025 | Out-of-sample |

Gate opening: ≥5% back-month volume share. Gate high (excluded): ≥80% back-month share.

---

## Signal Architecture

### Fair Value

```
FV = S × (r_f − q) × ΔT
```

- `S`: front-month mid price
- `r_f`: SOFR (FRED, daily scalar)
- `q`: S&P 500 trailing dividend yield (~1.30%)
- `ΔT`: exact time between expiries from `definition` schema (years)

### Z-Score

```
z_t = (spread_t − FV_t) / σ_10min
```

Rolling 10-minute window, computed on 1-second RTH bars. `spread_t = back_mid − front_mid` using LVCF alignment (`pd.merge_asof`, `direction="backward"`).

### Entry

- Threshold: |z| > 2.5σ (edge-triggered: signal fires on first bar crossing the threshold)
- Fill: T+1 bar open (no look-ahead)
- Long spread when z < −2.5 (spread cheap vs FV); short spread when z > +2.5
- FOMC full calendar-day exclusion applied

### Exit — Standard Layers

| Layer | Fraction | TP | SL after TP1 |
|-------|----------|----|--------------|
| 1 | 90% | +0.50 pt | Move to breakeven |
| 2 | 10% | +0.75 pt | — |
| Full stop | 100% | — | −0.50 pt from entry |

### Exit — Low-Z Overlay

When |z\_fill| < 2.0 at T+1 (entry has faded toward equilibrium by fill time):

| Layer | Fraction | TP | SL |
|-------|----------|----|----|
| 1 | 100% | +0.25 pt | −0.50 pt |

### HC Add-On

When |z\_fill| > 3.0 at T+1 (signal has strengthened by fill time):

- Execute unconditional 10-lot add-on at T+2 bar open
- Total position doubles to 20 lots with blended entry price
- Add-on inherits same exit logic as primary signal
- No additional gate beyond FOMC exclusion

---

## Sessions Analyzed

| Session | UTC Window | Notes |
|---------|------------|-------|
| European | 07:00–12:29 | Primary alpha source |
| US RTH | 13:30–20:15 | Main US session |
| Post-close | 20:15–07:00 | Low volume |

---

## Regime Gates

Eight gates were evaluated on IS data (W1+W2). Two were accepted into V1:

1. **drift_4h** — blocks short entries during sustained bullish RTH drift (4-hour lookback); net IS improvement +$134
2. **low-z two-layer exit** — modified exit when |entry_z| < 2.0 at fill; net IS improvement +$58

Rejected gates: OFI gate, half-life gate, OI-ratio gate, early-roll day block, FOMC-only gate, time-of-day filter.

---

## Transaction Costs

| Component | $/lot (round-trip) |
|-----------|-------------------|
| Exchange fees | $4.60 |
| NFA fee | $0.04 |
| Broker commission | $3.40 |
| **Total** | **$8.04** |

Fill model: synthetic midprice (zero slippage assumption). Each trade = 1 round-trip.

---

## External Baseline — Monoyios & Sarno (2002) ESTAR

> Monoyios, M. & Sarno, L. (2002). "Mean Reversion in Stock Index Futures Markets: A Nonlinear Analysis." *Journal of Futures Markets*, 22(4), 285–314.

Signal: Φ(γ; z) = 1 − exp(−γ² · z²)

Implemented in `scripts/ms_baseline.py` with γ=0.5, entry at |z|>2.0, exit at |z|<0.25, SL=0.50pt.

| Pool | n | avg\_net/lot | p-value |
|------|---|-------------|---------|
| IS (W1+W2) | 3,000 | −$4.36 | <0.001*** |
| OOS (W3+W4) | 1,782 | −$4.00 | <0.001*** |

Failure mode: exit_z=0.25 fires at first tick-back on 1-second bars → hold_med=0.0min → $8.04 TC destroys all gross edge. Serves as the lower-performance reference demonstrating that naive ESTAR requires adaptation for intraday use.

---

## Internal Baseline — Z-Score Ungated

Same z-score signal with no regime gate (`regime_gate='none'`).

| Pool | n | avg\_net/lot | p-value |
|------|---|-------------|---------|
| IS aggregate | 554 | +$1.17 | 0.198 |
| OOS aggregate | 670 | +$2.01 | 0.027** |

---

## V1 Strategy Results

V1 = z-score signal + drift_4h gate + low-z overlay.

| Pool | n | avg\_net/lot | p-value | Notes |
|------|---|-------------|---------|-------|
| IS all (W1+W2) | 554 | +$1.17 | 0.198 | Not significant |
| OOS all (W3+W4) | 670 | +$2.01 | 0.027** | — |
| OOS European | 190 | +$5.15 | <0.001*** | Primary edge source |
| OOS V1 gate | 299 | +$2.47 | — | Sharpe CI [+0.06, +0.27] |

Annualized Sharpe (OOS): +3.88 (90% CI: [+1.38, +6.63])

**Verdict:** Marginal positive edge; not yet deployable. Only the European session achieves statistically significant edge. Evaluation is underpowered (4 roll windows); proper OOS requires ≥10 periods.

---

## Running the Pipeline

```bash
# Activate environment
source .venv/bin/activate

# Estimate data costs before pulling (review output before proceeding)
python scripts/databento_cost_estimate.py

# Pull tick data (idempotent — skips existing parquet files)
python scripts/databento_datapull.py

# Run M&S ESTAR external baseline
python scripts/ms_baseline.py

# Run full session backtest (Z-score Ungated + V1 gate variants)
python scripts/run_sessions.py

# Generate IS/OOS performance report with bootstrap CIs
python scripts/result_summary.py

# Regenerate all 36 supplementary notebooks
python notebooks/supplementary/generate_notebooks.py
```

---

## Statistical Methods

- **Significance test:** one-sample t-test on per-trade net P&L (H₀: avg_net = 0)
- **Bootstrap CI:** seed=42, 5,000 iterations, percentile method
- **Sharpe (per-trade):** mean/std of per-trade net P&L
- **Annualized Sharpe:** per-trade Sharpe × √(estimated annual trade count)
- **Recovery Factor:** |total\_net| / MDD
