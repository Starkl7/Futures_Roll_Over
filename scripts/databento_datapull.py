"""
Databento Data Pull — ES Futures Roll Analysis
==============================================
Phase 2b: pulls tick data for all confirmed roll windows and saves to parquet.

Pull order (cheapest/smallest first — abort early if budget is tight):
  definition  — free, instrument metadata, pulled once per contract pair
  statistics  — settlement prices + open interest (~<100 MB per window)
  trades      — confirmed prints (~1-3 GB per window)
  mbp-10      — top-10 order book (~25-35 GB per window raw;
                significantly smaller as snappy-compressed parquet)

Idempotent: if a parquet file already exists for a given schema/pair/window
it is skipped entirely. Safe to re-run after a partial pull.

Run databento_cost_estimate.py first to verify remaining credit budget.
"""

import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import databento as db
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────

API_KEY  = os.environ["DATABENTO_API_KEY"]
DATASET  = "GLBX.MDP3"
STYPE    = "raw_symbol"
DATA_DIR = Path("/Volumes/SEAGATE/Databento_Futures")

# Confirmed roll windows from EDA (notebooks/01_eda.ipynb)
# Crossover = first day back-month volume >= front-month volume (always Monday
# of expiry week for ES quarterly rolls).
# Roll window = [crossover − 2 biz days, crossover + 3 biz days] inclusive.
# Format: (front, back, crossover, roll_start, roll_end)
CONFIRMED_WINDOWS = [
    ("ESU4", "ESZ4", "2024-09-16", "2024-09-12", "2024-09-19"),
    ("ESZ4", "ESH5", "2024-12-16", "2024-12-12", "2024-12-19"),
    ("ESH5", "ESM5", "2025-03-17", "2025-03-13", "2025-03-20"),
    ("ESM5", "ESU5", "2025-06-16", "2025-06-12", "2025-06-19"),
]

# Pull order: free/tiny first, largest last.
# Set billable=False for schemas that don't consume credits.
SCHEMAS = [
    ("definition", False),  # free — once per pair, not per window
    ("statistics", True),   # settlement + OI, daily granularity
    ("trades",     True),   # individual trade prints
    ("mbp-10",     True),   # top-10 book (largest — pull last)
]

_FILE_PREFIX = {
    "definition": "definitions",
    "statistics": "stats",
    "trades":     "trades",
    "mbp-10":     "mbp10",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────

def out_path(schema: str, front: str, back: str, roll_start: str) -> Path:
    prefix = _FILE_PREFIX[schema]
    if schema == "definition":
        # No date suffix — definitions don't change within a contract's life
        return DATA_DIR / f"{prefix}_{front}_{back}.parquet"
    return DATA_DIR / f"{prefix}_{front}_{back}_{roll_start}.parquet"


def api_end(d: str) -> str:
    """Databento end is exclusive — add 1 calendar day so d itself is included."""
    return (date.fromisoformat(d) + timedelta(days=1)).isoformat()


def pull_to_parquet(client: db.Historical, schema: str, symbols: list[str],
                    start: str, end: str, dest: Path) -> bool:
    """
    Stream schema data to a temporary .dbn.zst file, then convert to parquet.
    Cleans up the intermediate file on both success and failure.
    Returns True on success.
    """
    tmp = dest.with_suffix(".dbn.zst")

    try:
        print(f"    Downloading  {schema} → {tmp.name} ...", flush=True)
        store = client.timeseries.get_range(
            dataset=DATASET,
            symbols=symbols,
            schema=schema,
            start=start,
            end=end,
            stype_in=STYPE,
            path=tmp,           # stream to disk; returned DBNStore reads from file
        )

        print(f"    Converting   → {dest.name} ...", flush=True)
        store.to_parquet(
            dest,
            map_symbols=True,   # replace instrument_id with raw symbol strings
        )
        tmp.unlink(missing_ok=True)
        print(f"    ✓ {dest.name}  ({_fmt_size(dest)})")
        return True

    except Exception as exc:
        print(f"    ✗ Failed: {exc}")
        tmp.unlink(missing_ok=True)
        return False


def _fmt_size(path: Path) -> str:
    n = path.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def recover_dbn_files() -> None:
    """Convert any leftover .dbn.zst temp files to parquet (from interrupted runs)."""
    dbn_files = sorted(
        f for f in DATA_DIR.iterdir()
        if f.name.endswith(".dbn.zst") and not f.name.startswith(".")
    )
    if not dbn_files:
        return
    print(f"{'═' * 70}")
    print(f"  RECOVERY — {len(dbn_files)} interrupted download(s) found")
    print(f"{'─' * 70}")
    for tmp in dbn_files:
        # e.g. mbp10_ESU4_ESZ4_2024-09-12.dbn.zst → mbp10_ESU4_ESZ4_2024-09-12.parquet
        dest = tmp.with_suffix("").with_suffix(".parquet")
        print(f"  {tmp.name}  →  {dest.name} ...", flush=True)
        try:
            store = db.read_dbn(tmp)
            store.to_parquet(dest, map_symbols=True)
            tmp.unlink()
            print(f"  ✓ {dest.name}  ({_fmt_size(dest)})")
        except Exception as exc:
            print(f"  ✗ Failed: {exc}")


def pull_mbp10_chunked(client: db.Historical, front: str, back: str,
                       roll_start: str, roll_end: str) -> None:
    """
    Pull mbp-10 one calendar day at a time.
    Keeps each streaming request well under the 5 GB advisory limit.
    Saves per-day parquet files: mbp10_{front}_{back}_{roll_start}_{date}.parquet
    Saturday is skipped (CME closed).
    """
    day      = date.fromisoformat(roll_start)
    end_date = date.fromisoformat(roll_end)

    while day <= end_date:
        if day.weekday() == 5:          # Saturday — CME closed
            day += timedelta(days=1)
            continue

        day_str = day.isoformat()
        dest = DATA_DIR / f"mbp10_{front}_{back}_{roll_start}_{day_str}.parquet"
        tmp  = dest.with_suffix(".dbn.zst")

        print(f"\n    {day_str} ({day.strftime('%a')})", flush=True)

        if dest.exists():
            print(f"      ↷ Exists ({_fmt_size(dest)}) — skipping")
            day += timedelta(days=1)
            continue

        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*5 GB.*")
                store = client.timeseries.get_range(
                    dataset=DATASET,
                    symbols=[front, back],
                    schema="mbp-10",
                    start=day_str,
                    end=(day + timedelta(days=1)).isoformat(),
                    stype_in=STYPE,
                    path=tmp,
                )
            store.to_parquet(dest, map_symbols=True)
            tmp.unlink(missing_ok=True)
            print(f"      ✓ {_fmt_size(dest)}")
        except Exception as exc:
            print(f"      ✗ {exc}")
            tmp.unlink(missing_ok=True)

        day += timedelta(days=1)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = db.Historical(key=API_KEY)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    recover_dbn_files()

    pulled   = 0
    skipped  = 0
    failures: list[tuple[str, str]] = []

    for schema, billable in SCHEMAS:
        print(f"\n{'═' * 70}")
        label = f"SCHEMA: {schema.upper()}"
        print(f"  {label}{'  (free)' if not billable else ''}")
        print(f"{'─' * 70}")

        seen_pairs: set[tuple[str, str]] = set()

        for front, back, crossover, roll_start, roll_end in CONFIRMED_WINDOWS:
            pair = f"{front}/{back}"

            if schema == "definition":
                if (front, back) in seen_pairs:
                    continue
                seen_pairs.add((front, back))
                # Pull a single session — definitions are stable for the life of the contract
                start, end = roll_start, api_end(roll_start)
                window_label = f"(as of {roll_start})"
            else:
                start, end = roll_start, api_end(roll_end)
                window_label = f"{roll_start} → {roll_end}"

            if schema == "mbp-10":
                print(f"\n  {pair}  {window_label}")
                pull_mbp10_chunked(client, front, back, roll_start, roll_end)
                continue

            dest = out_path(schema, front, back, roll_start)
            print(f"\n  {pair}  {window_label}")

            if dest.exists():
                print(f"    ↷ Exists ({_fmt_size(dest)}) — skipping")
                skipped += 1
                continue

            ok = pull_to_parquet(client, schema, [front, back], start, end, dest)
            if ok:
                pulled += 1
            else:
                failures.append((schema, pair))

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  COMPLETE  —  {pulled} pulled,  {skipped} skipped,  {len(failures)} failed")
    if failures:
        print(f"\n  Failed pulls:")
        for s, p in failures:
            print(f"    ✗  {s:<12}  {p}")
    print(f"{'═' * 70}\n")


if __name__ == "__main__":
    main()
