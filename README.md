# ES Futures Calendar Spread — Mean-Reversion Research

![Python](https://img.shields.io/badge/python-3.14-blue)
![Data](https://img.shields.io/badge/data-Databento%20MBP--10-orange)
![Status](https://img.shields.io/badge/status-research-lightgrey)

Quantitative research project studying mean-reversion of the E-mini S&P 500 (ES) calendar spread against its cost-of-carry fair value during quarterly roll windows. Covers four roll periods from September 2024 through June 2025, with full signal development, regime gate engineering, and in-sample / out-of-sample backtest evaluation.

---

## Strategy Overview

**Alpha source:** The observed calendar spread (`ES_back − ES_front`) persistently deviates from its theoretical fair value:

```
FV = S × (r_f − q) × ΔT
```

where `r_f` = SOFR (via FRED), `q` = S&P 500 trailing dividend yield (~1.30%), and `ΔT` = time between expiries from Databento `definition` schema.

**Signal:** Enter when the z-score of `(spread − FV)` exceeds ±2σ on a 60-second rolling window; exit at |z| < 0.5σ or at a 2.5σ stop-loss. Position sizing is 10 lots per signal.

**Session decomposition:** Three session types analyzed — European (07:00–12:29 UTC), US RTH (13:30–20:15 UTC), Post-close — with European session identified as the primary alpha source.

**Transaction costs:** $8.04/lot round-trip (exchange fees + NFA + broker); fill model uses synthetic midprice (zero slippage assumption).

---

## Key Results

| Pool | n | Avg Net/Lot | p-value | 90% CI |
|------|---|-------------|---------|--------|
| IS all (W1+W2) | 554 | +$1.17 | 0.198 | [−$0.61, +$2.94] |
| OOS all (W3+W4) | 670 | +$2.01 | 0.027 ** | [+$0.24, +$3.79] |
| OOS European | 190 | +$5.15 | <0.001 *** | [+$3.10, +$7.21] |
| OOS V1 gate | 299 | +$2.47 | — | Sharpe CI [+0.06, +0.27] |

**V1 strategy** (drift_4h gate + low-z two-layer exit):

| Metric | Baseline | V1 |
|--------|----------|----|
| Net P&L (10-lot) | +$9,452 | +$10,488 |
| Max Drawdown | −$6,659 | −$3,115 |
| Recovery Factor | 1.42 | **3.37** |
| OOS per-trade Sharpe | +0.058 | **+0.159** |
| Annualized Sharpe (OOS V1) | — | **+3.88** (90% CI: [+1.38, +6.63]) |

**Verdict:** *Marginal positive edge; not yet deployable.* Only the European session achieves statistically significant edge. Evaluation is underpowered (4 roll windows); proper OOS requires ≥10 periods.

**Roll windows:**

| Window | Contracts | Roll Date | Role |
|--------|-----------|-----------|------|
| W1 | ESU4 → ESZ4 | Sep 16, 2024 | In-sample |
| W2 | ESZ4 → ESH5 | Dec 16, 2024 | In-sample |
| W3 | ESH5 → ESM5 | Mar 17, 2025 | Out-of-sample |
| W4 | ESM5 → ESU5 | Jun 16, 2025 | Out-of-sample |

---

## Repository Structure

```
notebooks/
  strategy.py              # Core backtest engine — gate system, simulate(), compute_stats()
  run_sessions.py          # CLI runner: per-session execution across all windows
  run_backtest.py          # Single-window backtest entrypoint
  result_summary.py        # IS/OOS performance report with bootstrap CIs
  results_dashboard.ipynb  # Portfolio equity curves and MDD visualization
  01_eda.ipynb             # W1 exploratory data analysis
  02_signal_analysis.ipynb # Five-signal evaluation (FV z-score, lead-lag, OBI, TFI, FOMC)
  03_zscore_simulation.py  # Z-score parameter sweep
  04_threshold_analysis.py # Entry/exit threshold optimization
  05_robustness_globex.py  # Globex session robustness check
  07_tearsheet.py          # W1 performance tearsheet
  08_fomc_trend.py         # FOMC event-day analysis
  09–13_...                # Additional window and signal analyses

Supplementary_notebooks/   # 36 publication-quality analysis notebooks (see below)
  config.py                # Shared paths and window definitions for supplementary notebooks
  generate_notebooks.py    # Script to regenerate all supplementary notebooks

databento_cost_estimate.py # Phase 2a: metadata cost estimation before data pull
databento_datapull.py      # Phase 2b: tick data pull (idempotent)
es_roll_analysis_workflow.md  # Full pipeline design specification
```

> `data/`, `results/`, and `reports/` are gitignored. Tick data lives on external storage (~25–35 GB per roll window for `mbp-10`).

---

## Supplementary Notebooks

Organized into eight analytical modules, each targeting a distinct research question:

| Module | Notebooks | Topic |
|--------|-----------|-------|
| **A** | A1–A5 | Performance: equity curves, bootstrap CI, P&L distributions |
| **B** | B1–B4 | FV deviation: daily mean deviation, hour×rollday heatmap, dual-axis overlay |
| **C** | C1–C3 | Microstructure: OU half-life, ACF lag-1, z-score paths with trade overlay |
| **D** | D1–D5 | Trade analytics: MAE/MFE scatter, exit type breakdown, entry hour, L/S decomposition, WR by z-bucket |
| **E** | E1–E9 | Bid-ask spread dynamics: per-window BA analysis, cross-window grid, width distributions, liquidity migration |
| **F** | F1–F4 | Spike events: Mar 14 (W3) and Jun 13 (W4) order book depth at spike |
| **G** | G1–G3 | Volume migration: crossover curves, OI proxy |
| **H** | H1–H3 | Pre-roll context: pre-roll deviation scatter, SOFR trajectory, session open HL gate |

---

## Environment Setup

```bash
# Python 3.14 venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your Databento API key in the environment (never hardcode it):

```bash
export DATABENTO_API_KEY="your_key_here"
```

Data files are expected at `/Volumes/SEAGATE/Databento_Futures/` by default. Update paths in `Supplementary_notebooks/config.py` if your storage location differs.

---

## Running the Analysis

With tick data already pulled, the analysis pipeline runs in order:

```bash
# Estimate data costs before pulling (review output before proceeding)
python databento_cost_estimate.py

# Pull tick data (idempotent — skips existing parquet files)
python databento_datapull.py

# Run full session backtest across all windows and gate variants
python notebooks/run_sessions.py

# Generate IS/OOS performance report with bootstrap CIs
python notebooks/result_summary.py

# Regenerate all 36 supplementary notebooks
python Supplementary_notebooks/generate_notebooks.py
```

---

## Signal Architecture Notes

Five alpha signals were evaluated across W1 and W2:

- **FV Deviation Z-Score** — selected; only signal with consistent positive edge
- **Lead-Lag (front → back)** — rejected; peak cross-correlation at lag=0s (contemporaneous, not predictive)
- **Order Book Imbalance (OBI)** — rejected; r ≈ +0.001 to −0.006 across windows (near-zero)
- **Trade Flow Imbalance (TFI)** — rejected; reverses sign across windows
- **FOMC event-driven jump** — implemented as exclusion filter only (entire FOMC calendar day excluded)

Eight regime gates were evaluated; two were accepted into V1:
1. **drift_4h gate** — blocks short entries during sustained bullish RTH drift (4-hour lookback); saved +$134 IS
2. **low-z two-layer exit** — modified exit for |z| < 2.0 bucket, two-layer partial close; saved +$58 IS

---

## Disclaimer

This repository contains research code and results for educational and analytical purposes. Nothing here constitutes financial advice or a recommendation to trade. Past backtest performance does not guarantee future results. ES futures carry substantial risk of loss.
