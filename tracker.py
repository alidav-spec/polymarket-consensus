"""
Component 4 - Resolution tracker.
For each consensus position, checks whether the market has resolved and
records whether the top-trader consensus side was correct.

Persists results in data/resolution_tracker.parquet (append-safe).
Already-resolved markets are never re-fetched.
first_detected_date and avg_prob_at_detection are frozen at first detection.

Output schema:
    condition_id              str
    question                  str
    category                  str
    consensus_side            str   - "Yes" / "No"
    consensus_trader_count    int   - updated each run (may grow over time)
    avg_prob_at_detection     float - market probability when first detected
    first_detected_date       str   - ISO date of first consensus detection
    resolved                  bool
    winning_outcome           str | None
    consensus_correct         bool | None  - None if unresolved
    resolved_date             str | None   - ISO date of resolution
"""

import json
import sys
import time
from datetime import date

import pandas as pd
import requests

from config import (
    CONSENSUS_FILE, HEADERS, MARKETS_URL,
    REQUEST_DELAY, RESOLUTION_FILE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(raw) -> list | dict:
    if isinstance(raw, (list, dict)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return []


def _fetch_market(condition_id: str, session: requests.Session) -> dict | None:
    """Return market dict for a condition_id, or None on failure."""
    try:
        resp = session.get(
            MARKETS_URL,
            params={"condition_ids": condition_id, "limit": 1},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            return data[0]
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"  Warning: market fetch failed for {condition_id[:12]}...: {exc}", file=sys.stderr)
    return None


def _determine_winner(market: dict) -> str | None:
    """
    Return the winning outcome name if the market is clearly resolved,
    otherwise None.  Resolved when closed=True and one price >= 0.95.
    """
    is_closed = market.get("closed") or not market.get("active", True)
    if not is_closed:
        return None

    outcomes = _safe_json(market.get("outcomes", "[]"))
    prices   = _safe_json(market.get("outcomePrices", "[]"))

    try:
        prices_f = [float(p) for p in prices]
    except (ValueError, TypeError):
        return None

    if not prices_f:
        return None

    max_price = max(prices_f)
    if max_price < 0.95:
        return None  # not yet resolved, or voided

    winner_idx = prices_f.index(max_price)
    return str(outcomes[winner_idx]) if winner_idx < len(outcomes) else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def update_resolution_tracker(
    consensus_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    if consensus_df is None:
        consensus_df = pd.read_parquet(CONSENSUS_FILE)

    if consensus_df.empty:
        print("  No consensus data to track.")
        return pd.DataFrame()

    today = date.today().isoformat()

    # Load existing tracker
    try:
        existing = pd.read_parquet(RESOLUTION_FILE)
        # Build a fast lookup: condition_id -> existing record dict
        existing_by_cid: dict[str, dict] = {
            row["condition_id"]: row.to_dict()
            for _, row in existing.iterrows()
        }
        already_resolved: set[str] = {
            cid for cid, rec in existing_by_cid.items() if rec.get("resolved")
        }
    except FileNotFoundError:
        existing = pd.DataFrame()
        existing_by_cid = {}
        already_resolved = set()

    # Current consensus: dominant side per market
    strong = consensus_df[consensus_df["is_consensus"]].copy()
    per_market = (
        strong
        .sort_values("trader_count", ascending=False)
        .drop_duplicates(subset="condition_id", keep="first")
    )

    # Check: current consensus markets not yet resolved
    # PLUS previously tracked markets that are still unresolved (even if they
    # dropped out of today's consensus -- we still want to know the outcome)
    prev_unresolved = {
        cid for cid, rec in existing_by_cid.items()
        if not rec.get("resolved")
    }
    current_consensus_cids = set(per_market["condition_id"])
    all_to_check_cids = (current_consensus_cids | prev_unresolved) - already_resolved

    # Build a lookup of current consensus rows for quick access
    current_by_cid = {row["condition_id"]: row for _, row in per_market.iterrows()}

    print(
        f"Checking resolution for {len(all_to_check_cids)} markets "
        f"({len(already_resolved)} already resolved, skipped)..."
    )

    session = requests.Session()
    session.headers.update(HEADERS)

    new_records: list[dict] = []
    for cid in all_to_check_cids:
        market   = _fetch_market(cid, session)
        winner   = _determine_winner(market) if market else None
        resolved = winner is not None

        prev = existing_by_cid.get(cid, {})

        # Use current consensus data if available, else fall back to stored data
        if cid in current_by_cid:
            row             = current_by_cid[cid]
            question        = row["question"]
            category        = row["category"]
            consensus_side  = row["side"]
            trader_count    = int(row["trader_count"])
            avg_prob        = float(row.get("avg_prob", 0))
        else:
            question        = prev.get("question", "")
            category        = prev.get("category", "Unknown")
            consensus_side  = prev.get("consensus_side", "")
            trader_count    = int(prev.get("consensus_trader_count", 0))
            avg_prob        = float(prev.get("avg_prob_at_detection", 0))

        new_records.append(
            {
                "condition_id":           cid,
                "question":               question,
                "category":               category,
                "consensus_side":         consensus_side,
                "consensus_trader_count": trader_count,
                # Freeze these at first detection -- never overwrite
                "avg_prob_at_detection":  prev.get("avg_prob_at_detection") or avg_prob,
                "first_detected_date":    prev.get("first_detected_date") or today,
                "resolved":               resolved,
                "winning_outcome":        winner,
                "consensus_correct":      (
                    winner.lower() == str(consensus_side).lower()
                    if resolved else None
                ),
                "resolved_date":          today if resolved else prev.get("resolved_date"),
            }
        )
        time.sleep(REQUEST_DELAY)

    if not new_records:
        print("  Nothing new to update.")
        return existing if not existing.empty else pd.DataFrame()

    new_df = pd.DataFrame(new_records)

    # Combine: keep resolved records that weren't re-checked this run,
    # plus all freshly checked records
    if not existing.empty:
        untouched = existing[~existing["condition_id"].isin(new_df["condition_id"])]
        combined  = pd.concat([untouched, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_parquet(RESOLUTION_FILE, index=False)

    resolved_df = combined[combined["resolved"]]
    n_correct   = int((resolved_df["consensus_correct"] == True).sum() or 0)
    n_resolved  = len(resolved_df)
    print(f"  Resolved markets: {n_resolved} / {len(combined)}")
    if n_resolved:
        pct = 100 * n_correct / n_resolved
        print(f"  Overall accuracy: {n_correct}/{n_resolved} ({pct:.1f}%)")
    print(f"  Saved -> {RESOLUTION_FILE}")

    return combined


def build_accuracy_table(tracker_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Return per-category accuracy summary for resolved consensus markets.
    Includes avg_market_prob_at_detection and edge_vs_market so callers can
    compare consensus accuracy against what the market already implied.
    """
    if tracker_df is None:
        try:
            tracker_df = pd.read_parquet(RESOLUTION_FILE)
        except FileNotFoundError:
            return pd.DataFrame()

    resolved = tracker_df[
        tracker_df["resolved"] & tracker_df["consensus_correct"].notna()
    ].copy()

    if resolved.empty:
        return pd.DataFrame()

    resolved["consensus_correct"] = resolved["consensus_correct"].astype(float)
    has_prob = "avg_prob_at_detection" in resolved.columns

    agg_spec: dict = {
        "total_markets": ("condition_id", "count"),
        "correct":       ("consensus_correct", "sum"),
    }
    if has_prob:
        agg_spec["avg_market_prob"] = ("avg_prob_at_detection", "mean")

    accuracy = (
        resolved
        .groupby("category")
        .agg(**agg_spec)
        .assign(accuracy_pct=lambda d: 100 * d["correct"] / d["total_markets"])
        .reset_index()
    )

    if has_prob:
        accuracy["edge_vs_market"] = (
            accuracy["accuracy_pct"] - 100 * accuracy["avg_market_prob"]
        )

    accuracy = accuracy.sort_values("accuracy_pct", ascending=False)
    return accuracy


if __name__ == "__main__":
    update_resolution_tracker()
