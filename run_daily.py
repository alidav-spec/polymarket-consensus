"""
run_daily.py - Daily cron job for the Polymarket Consensus Tracker.

What it does each run:
  1. Pulls fresh leaderboard and position data
  2. Detects current consensus positions
  3. Saves a dated snapshot to data/snapshots/consensus_YYYY-MM-DD.parquet
  4. Checks ALL previously tracked unresolved markets for resolution
     (even ones that dropped out of today's consensus)
  5. Records outcome + accuracy in data/resolution_tracker.parquet
  6. Writes a brief summary log line

Exit codes:
  0  success
  1  error (exception or empty data)

Usage:
  python run_daily.py
  python run_daily.py --top-n 300
  python run_daily.py --no-positions-refresh   # skip slow position pull, use cached
"""

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    CONSENSUS_FILE, DATA_DIR, POSITIONS_FILE,
    REPORT_FILE, RESOLUTION_FILE, TOP_N_TRADERS, TRADERS_FILE,
)
from consensus import detect_consensus
from leaderboard import fetch_leaderboard
from positions import pull_all_positions
from tracker import build_accuracy_table, update_resolution_tracker

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("daily")


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------

def _save_snapshot(consensus_df: pd.DataFrame) -> Path:
    snap_dir = DATA_DIR / "snapshots"
    snap_dir.mkdir(exist_ok=True)
    path = snap_dir / f"consensus_{date.today().isoformat()}.parquet"
    consensus_df.to_parquet(path, index=False)
    return path


def _prev_resolved_count() -> int:
    try:
        df = pd.read_parquet(RESOLUTION_FILE)
        return int(df["resolved"].sum())
    except FileNotFoundError:
        return 0


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(top_n: int = TOP_N_TRADERS, refresh_positions: bool = True) -> int:
    """
    Execute the full daily pipeline.
    Returns 0 on success, 1 on error.
    """
    start = datetime.now(timezone.utc)
    log.info("=== Polymarket Consensus Tracker -- daily run ===")
    log.info(f"top_n={top_n}  refresh_positions={refresh_positions}")

    try:
        # -- 1. Leaderboard ---------------------------------------------------
        traders_df = fetch_leaderboard(top_n)
        if traders_df.empty:
            log.error("Leaderboard returned no data.")
            return 1

        # -- 2. Positions -----------------------------------------------------
        if refresh_positions:
            positions_df = pull_all_positions(traders_df)
        else:
            log.info("Skipping position refresh (--no-positions-refresh); using cached data.")
            try:
                positions_df = pd.read_parquet(POSITIONS_FILE)
                log.info(f"  Loaded {len(positions_df):,} positions from cache.")
            except FileNotFoundError:
                log.error("No cached positions file found; run without --no-positions-refresh first.")
                return 1

        if positions_df.empty:
            log.warning("No positions found. Continuing with empty consensus.")

        # -- 3. Consensus -----------------------------------------------------
        consensus_df = detect_consensus(positions_df)

        # -- 4. Snapshot ------------------------------------------------------
        snap_path = _save_snapshot(consensus_df)
        log.info(f"Snapshot saved -> {snap_path}")

        # -- 5. Resolution tracking -------------------------------------------
        prev_resolved = _prev_resolved_count()
        tracker_df    = update_resolution_tracker(consensus_df)

        newly_resolved = 0
        total_resolved = 0
        overall_acc    = ""
        if not tracker_df.empty:
            total_resolved = int(tracker_df["resolved"].sum())
            newly_resolved = total_resolved - prev_resolved
            resolved_rows  = tracker_df[
                tracker_df["resolved"] & tracker_df["consensus_correct"].notna()
            ]
            if not resolved_rows.empty:
                n_correct = int(resolved_rows["consensus_correct"].astype(float).sum())
                n_total   = len(resolved_rows)
                overall_acc = f" | accuracy {n_correct}/{n_total} ({100*n_correct/n_total:.1f}%)"

        # -- 6. Accuracy table (log it) ---------------------------------------
        acc = build_accuracy_table(tracker_df)
        if not acc.empty:
            log.info("Accuracy by category:")
            has_edge = "edge_vs_market" in acc.columns
            for _, row in acc.iterrows():
                edge_str = ""
                if has_edge:
                    sign = "+" if row["edge_vs_market"] >= 0 else ""
                    edge_str = f"  mkt={100*row['avg_market_prob']:.0f}%  edge={sign}{row['edge_vs_market']:.1f}pp"
                log.info(
                    f"  {str(row['category']):<20} "
                    f"{int(row['correct'])}/{int(row['total_markets'])} "
                    f"({row['accuracy_pct']:.1f}%)"
                    + edge_str
                )

        # -- 7. Final summary log line ----------------------------------------
        n_consensus = int(consensus_df["is_consensus"].sum()) if not consensus_df.empty else 0
        elapsed     = (datetime.now(timezone.utc) - start).seconds
        log.info(
            f"DONE [{elapsed}s] -- "
            f"{len(traders_df)} traders | "
            f"{len(positions_df):,} positions | "
            f"{n_consensus} consensus | "
            f"{newly_resolved} newly resolved | "
            f"{total_resolved} total resolved"
            + overall_acc
        )

        return 0

    except Exception as exc:
        log.error(f"Daily run failed: {exc}", exc_info=True)
        return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Daily Polymarket Consensus Tracker run."
    )
    parser.add_argument(
        "--top-n", type=int, default=TOP_N_TRADERS,
        help=f"Traders to track (default: {TOP_N_TRADERS}; API returns 50/page)",
    )
    parser.add_argument(
        "--no-positions-refresh", action="store_true",
        help="Skip pulling fresh positions; use the last cached data/positions.parquet.",
    )
    args = parser.parse_args()

    sys.exit(run(top_n=args.top_n, refresh_positions=not args.no_positions_refresh))
