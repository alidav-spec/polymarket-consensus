"""
Component 3 - Consensus detector.
Groups open positions by (market, side) and flags markets where
CONSENSUS_THRESHOLD or more top traders are on the same side.

Output schema (data/consensus.parquet):
    condition_id              str
    question                  str
    category                  str
    side                      str   - "Yes" or "No"
    trader_count              int   - distinct wallets on this side
    total_value_usd           float - combined position value
    avg_prob                  float - mean current probability
    traders                   str   - comma-separated usernames
    is_consensus              bool  - trader_count >= CONSENSUS_THRESHOLD
    total_traders_in_market   int   - traders on either side
    consensus_ratio           float - trader_count / total_traders_in_market
    traders_split             bool  - both sides have 3+ top traders
    consensus_strength        int   - trader_count minus traders on other sides
                                      (net lead; higher = more one-sided)
"""

import pandas as pd

from config import CONSENSUS_FILE, CONSENSUS_THRESHOLD, POSITIONS_FILE


def detect_consensus(positions_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if positions_df is None:
        positions_df = pd.read_parquet(POSITIONS_FILE)

    if positions_df.empty:
        print("  No positions data to analyze.")
        pd.DataFrame().to_parquet(CONSENSUS_FILE, index=False)
        return pd.DataFrame()

    print(
        f"Analyzing consensus across {len(positions_df):,} positions "
        f"({positions_df['wallet'].nunique()} traders, "
        f"{positions_df['condition_id'].nunique()} markets)..."
    )

    # Per (market, side): count traders and aggregate stats
    consensus = (
        positions_df
        .groupby(["condition_id", "question", "category", "side"], dropna=False)
        .agg(
            trader_count    = ("wallet",            "nunique"),
            total_value_usd = ("current_value_usd", "sum"),
            avg_prob        = ("current_prob",       "mean"),
            traders         = ("username",           lambda s: ", ".join(sorted(set(s.dropna())))),
        )
        .reset_index()
    )

    consensus["is_consensus"] = consensus["trader_count"] >= CONSENSUS_THRESHOLD

    # Total distinct traders per market (any side)
    market_totals = (
        positions_df
        .groupby("condition_id")["wallet"]
        .nunique()
        .rename("total_traders_in_market")
        .reset_index()
    )
    consensus = consensus.merge(market_totals, on="condition_id", how="left")
    consensus["consensus_ratio"] = (
        consensus["trader_count"] / consensus["total_traders_in_market"]
    )

    # Flag split markets (both sides have 3+ top traders)
    split_cids = (
        consensus[consensus["trader_count"] >= 3]
        .groupby("condition_id")
        .size()
        .loc[lambda s: s > 1]
        .index
    )
    consensus["traders_split"] = consensus["condition_id"].isin(split_cids)

    # consensus_strength = trader_count - (all other sides combined)
    # For a binary market: strength = lead_side - trailing_side
    # For uncontested: strength = trader_count (no one on other side)
    all_sides = (
        consensus
        .groupby("condition_id")["trader_count"]
        .sum()
        .rename("total_all_sides")
        .reset_index()
    )
    consensus = consensus.merge(all_sides, on="condition_id", how="left")
    consensus["consensus_strength"] = (
        2 * consensus["trader_count"] - consensus["total_all_sides"]
    ).astype(int)

    # Sort: consensus first, then by strength, then by value
    consensus = consensus.sort_values(
        ["is_consensus", "consensus_strength", "total_value_usd"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    consensus.to_parquet(CONSENSUS_FILE, index=False)

    n_consensus = int(consensus["is_consensus"].sum())
    n_markets   = consensus.loc[consensus["is_consensus"], "condition_id"].nunique()
    print(
        f"  Consensus positions ({CONSENSUS_THRESHOLD}+ traders): "
        f"{n_consensus} across {n_markets} markets"
    )
    print(f"  Saved -> {CONSENSUS_FILE}")
    return consensus


if __name__ == "__main__":
    detect_consensus()
