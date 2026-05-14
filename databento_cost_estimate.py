"""
Databento Cost Estimator — ES Futures Roll Analysis
=====================================================
Two-phase approach:

  PHASE 1 — Roll Detection (cheap: ohlcv-1d over broad 4-week window)
    Pull daily volume for both contract legs over a wide search window.
    Detect the volume crossover day: when back-month volume > front-month volume.
    Confirmed roll window = [crossover - 2 days, crossover + 3 days].
    ohlcv-1d data is saved to data/ for reuse in analysis — not re-pulled later.

  PHASE 2 — Cost Estimation (on confirmed windows only)
    Estimate costs for mbp-10, trades, statistics, definition
    using the dynamically detected roll windows.

Schemas:
  - ohlcv-1d   : daily bars — used only for roll detection (Phase 1)
  - mbp-10     : top-10 order book — primary lead-lag signal
  - trades     : confirmed prints — fair value anchor
  - statistics : settlement prices + open interest — OI migration signal
  - definition : instrument metadata — FREE, not billable
"""

import os
import databento as db
import pandas as pd
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_KEY = os.environ["DATABENTO_API_KEY"]

DATASET  = "GLBX.MDP3"
STYPE    = "raw_symbol"

# Broad search windows per contract pair — intentionally wide (~4 weeks).
# Roll detection will narrow these down to the actual active roll period.
# Format: (front, back, search_start, search_end)
SEARCH_WINDOWS = [
    ("ESU4", "ESZ4", "2024-08-22", "2024-09-20"),   # Sep 2024 expiry ~Sep 20
    ("ESZ4", "ESH5", "2024-11-21", "2024-12-20"),   # Dec 2024 expiry ~Dec 20
    ("ESH5", "ESM5", "2025-02-20", "2025-03-21"),   # Mar 2025 expiry ~Mar 21
    ("ESM5", "ESU5", "2025-05-22", "2025-06-20"),   # Jun 2025 expiry ~Jun 20
]

# Roll window buffer around detected crossover day
DAYS_BEFORE_CROSSOVER = 2   # capture build-up
DAYS_AFTER_CROSSOVER  = 3   # capture tail

# Output directory for all pulled data
DATA_DIR = "/Volumes/SEAGATE/Databento_Futures"

# Schemas for Phase 2 cost estimation
# Tuple: (schema_name, billable, description)
ESTIMATE_SCHEMAS = [
    ("mbp-10",     True,  "Top-10 order book — primary lead-lag signal"),
    ("trades",     True,  "Trade prints — fair value anchor"),
    ("statistics", True,  "Settlement + open interest — OI migration"),
    ("definition", False, "Instrument metadata — FREE"),
]


# ── HELPERS ───────────────────────────────────────────────────────────────────

def next_business_day(d: date, n: int) -> date:
    """Advance d by n business days (positive or negative)."""
    step = 1 if n >= 0 else -1
    count = abs(n)
    while count > 0:
        d += timedelta(days=step)
        if d.weekday() < 5:   # Mon–Fri
            count -= 1
    return d


def detect_crossover(client, front: str, back: str,
                     search_start: str, search_end: str) -> date | None:
    """
    Pulls ohlcv-1d for both legs over the broad search window.
    Returns the first date where back-month volume >= front-month volume.
    Returns None if no crossover found (data issue or too-narrow window).
    """
    try:
        data = client.timeseries.get_range(
            dataset=DATASET,
            symbols=[front, back],
            schema="ohlcv-1d",
            start=search_start,
            end=search_end,
            stype_in=STYPE,
        )
        df = data.to_df()

        # Persist immediately — reuse in analysis, avoid re-pulling
        out_path = os.path.join(DATA_DIR, f"ohlcv1d_{front}_{back}_{search_start[:7]}.parquet")
        df.to_parquet(out_path, index=True)
        print(f"    ✓ Saved: {out_path}")
    except Exception as e:
        print(f"    ⚠  ohlcv-1d pull failed for {front}/{back}: {e}")
        return None

    if df.empty:
        print(f"    ⚠  No data returned for {front}/{back}")
        return None

    df = df.reset_index()
    df["date"] = pd.to_datetime(df["ts_event"]).dt.date

    # Daily volume per symbol
    vol = (
        df.groupby(["date", "symbol"])["volume"]
        .sum()
        .unstack("symbol")
        .fillna(0)
    )

    if front not in vol.columns or back not in vol.columns:
        print(f"    ⚠  Missing columns — found: {list(vol.columns)}")
        return None

    crossover_mask = vol[back] >= vol[front]
    if not crossover_mask.any():
        print(f"    ⚠  No volume crossover found — widen search range")
        return None

    return vol.index[crossover_mask][0]


def confirmed_window(crossover: date) -> tuple[str, str]:
    """Convert crossover date to (start, end) strings for API calls."""
    start = next_business_day(crossover, -DAYS_BEFORE_CROSSOVER)
    end   = next_business_day(crossover,  DAYS_AFTER_CROSSOVER)
    return start.isoformat(), end.isoformat()


def estimate_cost(client, symbols, schema, start, end) -> float | None:
    try:
        return client.metadata.get_cost(
            dataset=DATASET,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=STYPE,
        )
    except Exception as e:
        print(f"    ⚠  Cost estimate failed ({schema}): {e}")
        return None


def estimate_size(client, symbols, schema, start, end) -> int | None:
    try:
        return client.metadata.get_billable_size(
            dataset=DATASET,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=STYPE,
        )
    except Exception as e:
        print(f"    ⚠  Size estimate failed ({schema}): {e}")
        return None


def fmt_bytes(n):
    if n is None:
        return "N/A"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def fmt_usd(x):
    if x is None:
        return "N/A"
    return f"${x:.4f}"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    client = db.Historical(key=API_KEY)
    os.makedirs(DATA_DIR, exist_ok=True)

    # ── PHASE 1: ROLL DETECTION ────────────────────────────────────────────────
    print("=" * 72)
    print("  PHASE 1 — Roll Detection via Volume Crossover (ohlcv-1d)")
    print("=" * 72)
    print(f"  {'Pair':<12} {'Search Window':<26} {'Crossover':>12}  Confirmed Window")
    print(f"  {'─'*10} {'─'*24} {'─'*12}  {'─'*22}")

    confirmed = []   # (front, back, roll_start, roll_end, crossover)

    for front, back, search_start, search_end in SEARCH_WINDOWS:
        crossover = detect_crossover(client, front, back, search_start, search_end)

        if crossover is None:
            # Fallback: centre of search window
            s = date.fromisoformat(search_start)
            e = date.fromisoformat(search_end)
            crossover = s + (e - s) // 2
            tag = "FALLBACK"
        else:
            tag = str(crossover)

        roll_start, roll_end = confirmed_window(crossover)
        confirmed.append((front, back, roll_start, roll_end, crossover))

        print(f"  {f'{front}/{back}':<12} {f'{search_start} → {search_end}':<26} "
              f"{tag:>12}  {roll_start} → {roll_end}")

    # ── PHASE 2: COST ESTIMATION ───────────────────────────────────────────────
    print(f"\n{'=' * 72}")
    print(f"  PHASE 2 — Cost Estimation on Confirmed Roll Windows")
    print("=" * 72)

    grand_total_cost = 0.0
    grand_total_size = 0
    results = []

    for schema, billable, description in ESTIMATE_SCHEMAS:
        schema_cost = 0.0
        schema_size = 0

        print(f"\n{'─'*72}")
        print(f"  SCHEMA: {schema.upper()}  —  {description}")

        if not billable:
            print(f"  (not billable — skipping cost estimation)")
            print(f"{'─'*72}")
            continue

        print(f"{'─'*72}")
        print(f"  {'Pair':<12} {'Confirmed Window':<26} {'Size':>10} {'Cost':>12}")
        print(f"  {'─'*10} {'─'*24} {'─'*10} {'─'*12}")

        for front, back, roll_start, roll_end, crossover in confirmed:
            symbols = [front, back]

            cost = estimate_cost(client, symbols, schema, roll_start, roll_end)
            size = estimate_size(client, symbols, schema, roll_start, roll_end)

            cost_val = cost if cost is not None else 0.0
            size_val = size if size is not None else 0

            schema_cost += cost_val
            schema_size += size_val

            print(f"  {f'{front}/{back}':<12} {f'{roll_start} → {roll_end}':<26} "
                  f"{fmt_bytes(size):>10} {fmt_usd(cost):>12}")

            results.append({
                "schema":     schema,
                "front":      front,
                "back":       back,
                "crossover":  str(crossover),
                "roll_start": roll_start,
                "roll_end":   roll_end,
                "size_bytes": size_val,
                "cost_usd":   cost_val,
            })

        print(f"  {'─'*10} {'─'*24} {'─'*10} {'─'*12}")
        print(f"  {'Schema subtotal':<38} {fmt_bytes(schema_size):>10} {fmt_usd(schema_cost):>12}")

        grand_total_cost += schema_cost
        grand_total_size += schema_size

    # ── SUMMARY ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  TOTAL ESTIMATED COST  (definition excluded — free)")
    print(f"{'─'*72}")
    print(f"  Total billable size : {fmt_bytes(grand_total_size)}")
    print(f"  Total cost          : {fmt_usd(grand_total_cost)}")
    print(f"  Check remaining credits: https://databento.com/portal/billing")
    print(f"{'═'*72}")

    if results:
        print(f"\n  Cost by schema:")
        schema_totals = {}
        for r in results:
            schema_totals.setdefault(r["schema"], 0.0)
            schema_totals[r["schema"]] += r["cost_usd"]
        for s, total in schema_totals.items():
            pct = (total / grand_total_cost * 100) if grand_total_cost > 0 else 0
            bar = "█" * int(pct / 5)
            print(f"  {s:<12} {fmt_usd(total):>10}  {bar} {pct:.1f}%")

    print(f"\n  ✓ Estimate complete. No tick data downloaded.")
    print(f"  Next: run databento_pull.py with these confirmed windows.")
    print("=" * 72)


if __name__ == "__main__":
    main()