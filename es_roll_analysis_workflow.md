# ES Futures Roll Analysis — Project Workflow

## 1. Project Overview

### Objective
Exploit predictable price dynamics in the E-mini S&P 500 (ES) futures calendar spread during quarterly roll periods. Two primary alpha sources are targeted:

1. **Fair value deviation** — the calendar spread (`F_back - F_front`) deviates from its theoretical cost-of-carry value; mean reversion to fair value is tradeable
2. **Lead-lag signal** — the near-month contract leads the deferred contract by one tick during the roll window; the spread between them predicts short-term returns in the leading leg

### Scope
- Instrument: CME E-mini S&P 500 futures (ES), two adjacent quarterly contract months per roll
- Data source: Databento (`GLBX.MDP3` — CME Globex)
- Roll periods: dynamically detected via volume crossover (see Phase 1)
- Backtest history: last 4 quarterly rolls (~1 year)

### Repository Structure

```
es-roll-analysis/
│
├── data/                          # all raw and processed data
│   ├── ohlcv1d_{front}_{back}_{YYYY-MM}.parquet   # Phase 1 output (roll detection)
│   ├── mbp10_{front}_{back}_{roll_start}.parquet  # Phase 2 tick book data
│   ├── trades_{front}_{back}_{roll_start}.parquet
│   ├── stats_{front}_{back}_{roll_start}.parquet
│   └── definitions_{front}_{back}.parquet
│
├── scripts/
│   ├── 01_detect_rolls.py         # Phase 1: ohlcv-1d pull + crossover detection
│   ├── 02_estimate_costs.py       # Phase 2a: metadata cost estimation
│   ├── 03_pull_data.py            # Phase 2b: actual tick data pull
│   ├── 04_build_signals.py        # fair value + lead-lag signal construction
│   ├── 05_backtest.py             # signal backtest and PnL attribution
│   └── 06_analysis.py             # output plots and summary statistics
│
├── src/
│   ├── roll_detection.py          # crossover logic, business day utilities
│   ├── fair_value.py              # theoretical spread model
│   ├── signals.py                 # lead-lag and deviation signals
│   ├── execution.py               # simulated fill logic, cost model
│   └── utils.py                   # shared helpers
│
├── notebooks/
│   ├── 01_eda.ipynb               # exploratory data analysis
│   └── 02_signal_analysis.ipynb   # signal diagnostics
│
├── results/                       # backtest outputs, plots
├── databento_cost_estimate.py     # standalone cost estimator (already built)
├── config.py                      # all configuration in one place
└── requirements.txt
```

---

## 2. Configuration (`config.py`)

All parameters live here. No dates, thresholds, or paths hardcoded elsewhere.

```python
# config.py

DATABENTO_API_KEY = "YOUR_KEY"   # or read from env: os.getenv("DATABENTO_API_KEY")
DATASET           = "GLBX.MDP3"
STYPE             = "raw_symbol"
DATA_DIR          = "data"
RESULTS_DIR       = "results"

# Contract pairs and broad search windows for roll detection
# Format: (front, back, search_start, search_end)
# search_start/end are intentionally wide (~4 weeks) — actual roll window
# is determined dynamically by volume crossover detection in Phase 1
SEARCH_WINDOWS = [
    ("ESU4", "ESZ4", "2024-08-22", "2024-09-20"),
    ("ESZ4", "ESH5", "2024-11-21", "2024-12-20"),
    ("ESH5", "ESM5", "2025-02-20", "2025-03-21"),
    ("ESM5", "ESU5", "2025-05-22", "2025-06-20"),
]

# Roll window buffer around detected volume crossover
DAYS_BEFORE_CROSSOVER = 2
DAYS_AFTER_CROSSOVER  = 3

# Fair value model inputs
SOFR_SERIES      = "SOFR"          # FRED series ID for risk-free rate
SPX_DIV_TICKER   = "^SP500TR"      # for implied dividend yield

# Signal parameters
SPREAD_ZSCORE_WINDOW    = 60       # seconds — rolling window for spread z-score
SPREAD_ENTRY_THRESHOLD  = 2.0      # z-score units
SPREAD_EXIT_THRESHOLD   = 0.5
LEAD_LAG_TICK_DELAY     = 1        # ticks — expected lead of front over back

# Execution model
TICK_SIZE        = 0.25            # ES tick size (index points)
TICK_VALUE       = 12.50           # USD per tick
BID_ASK_SPREAD   = 0.25            # assumed 1 tick spread (conservative)
COMMISSION_RT    = 2.10            # USD round-trip per contract (IB rates)
```

---

## 3. Phase 1 — Roll Detection (`01_detect_rolls.py`)

### Purpose
Determine the **exact roll window** for each contract pair by detecting when volume migrates from the front-month to the back-month contract. This replaces hardcoded date assumptions with data-driven windows.

### Method

1. Pull `ohlcv-1d` for both contract legs over the broad search window defined in `config.SEARCH_WINDOWS`
2. Aggregate daily volume per symbol
3. Find the **first date where back-month volume ≥ front-month volume** — this is the crossover day
4. Define confirmed roll window as `[crossover - DAYS_BEFORE_CROSSOVER, crossover + DAYS_AFTER_CROSSOVER]` in business days
5. Persist `ohlcv-1d` data to `data/` for reuse in OI migration analysis (Phase 4)
6. Write confirmed windows to `data/confirmed_windows.json`

### Output: `data/confirmed_windows.json`

```json
[
  {
    "front": "ESU4",
    "back": "ESZ4",
    "search_start": "2024-08-22",
    "search_end": "2024-09-20",
    "crossover": "2024-09-12",
    "roll_start": "2024-09-10",
    "roll_end": "2024-09-17"
  },
  ...
]
```

All downstream scripts read from this file. No dates are hardcoded anywhere else.

### Fallback
If no crossover is detected (data gap, atypical roll), the script falls back to the midpoint of the search window and flags the entry as `"fallback": true` in the JSON. These entries are flagged in all downstream outputs.

---

## 4. Phase 2a — Cost Estimation (`02_estimate_costs.py`)

### Purpose
Estimate Databento credit spend **before** committing to the actual data pull. Zero data is downloaded.

### Method
Reads `data/confirmed_windows.json`, calls `client.metadata.get_cost()` and `client.metadata.get_billable_size()` for each schema × window combination.

### Schemas Estimated

| Schema | Billable | Purpose |
|---|---|---|
| `mbp-10` | Yes | Top-10 order book — primary signal |
| `trades` | Yes | Confirmed prints — fair value anchor |
| `statistics` | Yes | Settlement prices + open interest |
| `definition` | No (free) | Instrument metadata |

### Decision Gate
Print total estimated cost. **Do not proceed to Phase 2b if cost exceeds available credits.** Options to reduce cost:
- Reduce `DAYS_BEFORE_CROSSOVER` / `DAYS_AFTER_CROSSOVER` in `config.py`
- Drop to 3 roll windows
- Replace `mbp-10` with `mbp-1` (top-of-book only) — ~10x cheaper, sufficient for fair value signal, loses lead-lag depth

---

## 5. Phase 2b — Data Pull (`03_pull_data.py`)

### Purpose
Pull all tick data for the confirmed roll windows and persist to disk.

### Execution
Reads `data/confirmed_windows.json`. For each window × schema:
1. Check if file already exists in `data/` — skip if present (idempotent)
2. Pull via `client.timeseries.get_range()`
3. Save to parquet immediately

### File Naming Convention

```
data/{schema}_{front}_{back}_{roll_start}.parquet

# Examples:
data/mbp10_ESU4_ESZ4_2024-09-10.parquet
data/trades_ESU4_ESZ4_2024-09-10.parquet
data/stats_ESU4_ESZ4_2024-09-10.parquet
data/definitions_ESU4_ESZ4.parquet       # no date — not time-series
```

### Schema-Specific Notes

**`mbp-10`**
- Two instruments pulled simultaneously via `symbols=[front, back]`
- Nanosecond-resolution timestamps — do not resample before signal construction
- Expected: ~25–35 GB per roll window, two contracts combined

**`trades`**
- Pull separately from `mbp-10` — used only for fair value anchoring, not signal
- Much smaller: ~1–3 GB per window

**`statistics`**
- Contains: settlement price, open interest, trading volume (daily records)
- Used for OI migration validation and fair value dividend input
- Tiny: < 100 MB per window

**`definition`**
- Pull once per contract pair, not per window
- Contains: expiry datetime (nanosecond precision), tick size, contract multiplier, underlying
- Free — pull at start of pipeline, cache permanently

---

## 6. Phase 3 — Fair Value Model (`src/fair_value.py`)

### Theoretical Calendar Spread

$$F_{\text{back}} - F_{\text{front}} = S \cdot (r_f - q) \cdot (T_{\text{back}} - T_{\text{front}})$$

Where:
- $S$ = SPX spot price (use SPY mid as proxy, or `trades` midprice)
- $r_f$ = SOFR (pulled daily from FRED API — free, no key required)
- $q$ = S&P 500 dividend yield (trailing 12-month, updated daily)
- $T_{\text{back}} - T_{\text{front}}$ = time between expiries in years (from `definition` schema)

### Implementation Notes
- $r_f$ and $q$ update daily — recompute fair value at the open of each session
- Use exact expiry timestamps from `definition` schema, not approximate calendar dates
- Fair value is computed as a scalar per session, not tick-by-tick — the cost-of-carry model does not change intraday (absent dividend announcements)

### Spread Deviation Signal

$$\text{deviation}_t = (F_{\text{back},t} - F_{\text{front},t}) - \text{FV}_t$$

Positive deviation → spread is rich → sell back, buy front  
Negative deviation → spread is cheap → buy back, sell front

---

## 7. Phase 4 — Signal Construction (`04_build_signals.py`)

### Input
- `mbp-10` parquet files for front and back contracts
- Fair value scalar per session (from Phase 3)

### Step 1 — Time Synchronization
`mbp-10` records for the two contracts arrive with independent timestamps. Align to a common nanosecond timeline via **last-value-carry-forward** (LVCF): at each unique timestamp across both instruments, carry forward the most recent book state of the other instrument.

```
merged_df = pd.merge_asof(front_df, back_df, on="ts_event",
                           direction="backward", suffixes=("_front", "_back"))
```

### Step 2 — Midprice and Spread Construction

```
mid_front = (best_bid_front + best_ask_front) / 2
mid_back  = (best_bid_back  + best_ask_back)  / 2
spread    = mid_back - mid_front
deviation = spread - fair_value
```

### Step 3 — Lead-Lag Signal

Based on Guan et al. (2025, arXiv:2501.03171): near-month leads deferred by one tick.

Measure the **lead-lag spread** as:

```
ll_spread_t = mid_front_t - mid_back_t  (normalized by fair value)
```

When `ll_spread` deviates significantly from its rolling mean, it predicts a negative return in the front contract (mean reversion). Z-score the deviation over a rolling `SPREAD_ZSCORE_WINDOW` second window.

### Step 4 — OI Migration Signal (from `ohlcv-1d` and `statistics`)

Track daily open interest ratio:

```
oi_ratio = OI_back / (OI_front + OI_back)
```

Use this as a **regime filter**: only trade during the active roll period (when `oi_ratio` is between 0.2 and 0.8). Outside this range, one contract dominates and the spread microstructure is less informative.

### Signal Summary

| Signal | Input | Type | Use |
|---|---|---|---|
| `deviation` | mbp-10, fair value | Continuous | Primary entry/exit |
| `ll_zscore` | mbp-10 | Continuous | Entry confirmation |
| `oi_ratio` | ohlcv-1d, statistics | Daily | Regime filter |

---

## 8. Phase 5 — Backtest (`05_backtest.py`)

### Architecture
Event-driven backtest replaying the merged tick stream. No vectorized operations on the full DataFrame — process tick-by-tick to avoid look-ahead bias.

### Entry Logic
Enter when **both** conditions hold:
1. `abs(deviation) > SPREAD_ENTRY_THRESHOLD × rolling_std(deviation)`
2. `abs(ll_zscore) > SPREAD_ENTRY_THRESHOLD`

Direction: sign of `deviation` determines long/short spread.

### Exit Logic
Exit when either:
- `abs(deviation) < SPREAD_EXIT_THRESHOLD × rolling_std(deviation)` (signal decay)
- End of roll window
- Stop-loss: `deviation` moves 2× entry level against position

### Fill Model
- Assume fills at **best bid/ask** of the lagging leg (conservative)
- Calendar spread traded as a single unit where possible (CME native spread order)
- Slippage: 1 tick per leg on entry, 0.5 ticks on exit (liquidity is better at exit)
- Commission: `COMMISSION_RT` per contract round-trip

### PnL Attribution
Decompose PnL into:
- **Spread convergence** — fair value reversion component
- **Lead-lag capture** — directional component from ll_zscore signal
- **Roll yield** — passive carry from holding the calendar spread
- **Transaction costs** — commissions + slippage

### Output Metrics per Roll Window
- Total PnL (USD)
- Sharpe ratio (annualized)
- Max drawdown
- Win rate, average win/loss
- Number of round-trips
- Cost drag (% of gross PnL)

---

## 9. Phase 6 — Analysis and Output (`06_analysis.py`)

### Plots
1. **Volume migration chart** — daily front vs back volume across the search window, crossover annotated
2. **OI migration chart** — `oi_ratio` over the roll window
3. **Spread vs fair value** — `deviation` time series with entry/exit markers
4. **Lead-lag spread dynamics** — `ll_zscore` time series
5. **PnL curve** — cumulative PnL per roll window, overlaid across all 4 windows
6. **Cost breakdown** — stacked bar: gross PnL vs transaction costs vs net PnL per window

### Summary Table
Aggregated metrics across all roll windows — the primary deliverable for the project portfolio.

---

## 10. External Data Requirements

| Data | Source | Method | Cost |
|---|---|---|---|
| SOFR (daily) | FRED API | `requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv?id=SOFR")` | Free |
| S&P 500 dividend yield | Multpl.com or FRED (`SP500DIV`) | Scrape or FRED API | Free |
| SPX spot (intraday) | Polygon.io | `/v2/aggs/` endpoint | Free tier sufficient |
| CME expiry calendar | Databento `definition` schema | Already in pipeline | Free |

---

## 11. Dependencies

```
databento>=0.35.0
pandas>=2.0.0
numpy>=1.26.0
pyarrow>=14.0.0       # parquet read/write
requests>=2.31.0      # FRED API, Polygon
scipy>=1.12.0         # stats utilities
matplotlib>=3.8.0     # plots
jupyter>=1.0.0        # notebooks
```

Install: `pip install -r requirements.txt`

---

## 12. Execution Order

```bash
# One-time setup
pip install -r requirements.txt
export DATABENTO_API_KEY="your_key"

# Step 1 — detect roll windows (pulls ohlcv-1d, writes confirmed_windows.json)
python scripts/01_detect_rolls.py

# Step 2 — estimate costs before committing
python scripts/02_estimate_costs.py
# Review output. Proceed only if within credit budget.

# Step 3 — pull tick data (idempotent — safe to re-run)
python scripts/03_pull_data.py

# Step 4 — build signals
python scripts/04_build_signals.py

# Step 5 — run backtest
python scripts/05_backtest.py

# Step 6 — generate analysis outputs
python scripts/06_analysis.py
```

---

## 13. Key Risks and Mitigations

| Risk | Mitigation |
|---|---|
| `mbp-10` data volume exceeds credit budget | Run `02_estimate_costs.py` first; fallback to `mbp-1` if over budget |
| Volume crossover not detected in search window | Fallback to window midpoint; flag in output; widen `SEARCH_WINDOWS` |
| Fair value model stale intraday (dividend surprise) | Alert if SPX deviation from prior-day fair value > 5 points |
| Lead-lag signal degraded outside roll window | `oi_ratio` regime filter gates signal; only trade when 0.2 < ratio < 0.8 |
| Look-ahead bias in backtest | Tick-by-tick event loop; rolling stats computed strictly on past data |
| IBKR paper account 15-min data delay | Backtest uses historical Databento data only; live paper trading is separate |
