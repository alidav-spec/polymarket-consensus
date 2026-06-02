"""
Component 2 - Position puller.
For each trader wallet, fetches open positions from data-api.polymarket.com.
All fields (question, price, side, value) are available directly in the
positions response -- no secondary market-lookup needed.

Position record schema:
    wallet            str   - trader's proxy wallet address
    rank              int   - leaderboard rank
    username          str
    condition_id      str   - Polymarket market condition ID
    question          str   - market question text
    category          str   - derived from event slug keywords
    side              str   - "Yes" or "No" (which outcome the trader holds)
    current_prob      float - current market probability for that outcome (0-1)
    size_shares       float - position size in shares
    current_value_usd float - estimated USD value at current probability
"""

import sys
import time
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import (
    HEADERS, POSITIONS_FILE, POSITIONS_LIMIT,
    POSITIONS_URL, REQUEST_DELAY, TRADERS_FILE,
)


# ---------------------------------------------------------------------------
# Category classifier (keyword-based, from eventSlug + title)
# ---------------------------------------------------------------------------

_CATEGORY_RULES: list[tuple[list[str], str]] = [
    (
        ["bitcoin", "btc", "crypto", "ethereum", "eth", "solana", "sol",
         "coinbase", "doge", "xrp", "defi", "blockchain", "token", "nft",
         "binance", "microstrategy"],
        "Crypto",
    ),
    (
        ["election", "vote", "democrat", "republican", "president", "senate",
         "congress", "parliament", "iran", "ukraine", "russia", "nato",
         "invasion", "military", "missile", "nuclear", "war", "deal",
         "ceasefire", "trump", "biden", "harris", "modi", "putin",
         "geopolit", "sanction", "regime", "peace", "strait", "hormuz"],
        "Politics",
    ),
    (
        ["tennis", "roland", "garros", "wimbledon", "nfl", "nba", "nhl",
         "mlb", "soccer", "football", "baseball", "golf", "ufc", "boxing",
         "superbowl", "world-cup", "championship", "league", "match",
         "tournament", "olympic", "swimming", "race", "knicks", "lakers",
         "celtics", "76ers", "bulls", "heat", "bucks", "warriors"],
        "Sports",
    ),
    (
        ["openai", "chatgpt", "llm", "claude", "gemini", "gpt",
         "artificial-intelligence", "ai-", "-ai-", "machine-learning",
         "nvidia", "apple", "google", "microsoft", "meta", "amazon",
         "tech", "software", "hardware"],
        "Tech",
    ),
    (
        ["nasdaq", "s-p-500", "dow", "stock", "earnings", "ipo", "fed-rate",
         "interest-rate", "inflation", "gdp", "recession", "market",
         "economy", "finance", "bank", "gold", "oil"],
        "Finance",
    ),
    (
        ["hurricane", "earthquake", "flood", "climate", "temperature",
         "weather", "storm", "volcano"],
        "Science",
    ),
    (
        ["oscar", "grammy", "emmy", "celebrity", "movie", "music", "album",
         "film", "tv-show", "streaming", "award", "box-office"],
        "Entertainment",
    ),
]


def _classify_category(event_slug: str, title: str) -> str:
    text = (event_slug + " " + title).lower()
    for keywords, category in _CATEGORY_RULES:
        if any(k in text for k in keywords):
            return category
    return "Other"


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _fetch_wallet_positions(wallet: str, session: requests.Session) -> list[dict]:
    params = {"user": wallet, "limit": POSITIONS_LIMIT, "sizeThreshold": "0.01"}
    try:
        resp = session.get(POSITIONS_URL, params=params, timeout=30)
        resp.raise_for_status()
        return resp.json() or []
    except Exception as exc:
        print(f"  Warning: fetch failed for {wallet[:12]}...: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def pull_all_positions(
    traders_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if traders_df is None:
        traders_df = pd.read_parquet(TRADERS_FILE)

    print(f"Pulling positions for {len(traders_df)} traders...")

    session = requests.Session()
    session.headers.update(HEADERS)

    raw_positions: list[dict] = []

    for _, trader in tqdm(
        traders_df.iterrows(), total=len(traders_df), desc="Fetching positions"
    ):
        wallet = trader["wallet"]
        positions = _fetch_wallet_positions(wallet, session)
        for pos in positions:
            pos["_wallet"]   = wallet
            pos["_rank"]     = trader["rank"]
            pos["_username"] = trader.get("username", "")
        raw_positions.extend(positions)
        time.sleep(REQUEST_DELAY)

    if not raw_positions:
        print("  No open positions found for any trader in the top 100.")
        empty = pd.DataFrame(
            columns=["wallet", "rank", "username", "condition_id", "question",
                     "category", "side", "current_prob", "size_shares",
                     "current_value_usd"]
        )
        empty.to_parquet(POSITIONS_FILE, index=False)
        return empty

    print(f"  Raw position records: {len(raw_positions)}")

    # The positions endpoint includes conditionId, outcome, curPrice, title,
    # and eventSlug directly -- no secondary market lookup required.
    records: list[dict] = []
    skipped = 0

    for pos in raw_positions:
        cid = pos.get("conditionId") or pos.get("condition_id")
        if not cid:
            skipped += 1
            continue

        outcome = pos.get("outcome") or pos.get("side") or "Unknown"
        cur_price = float(pos.get("curPrice") or pos.get("price") or 0.0)
        size      = float(pos.get("size") or pos.get("position") or 0.0)
        value     = float(pos.get("currentValue") or pos.get("value") or 0.0)
        if value == 0.0 and size and cur_price:
            value = size * cur_price

        title      = pos.get("title") or pos.get("question") or cid[:20]
        event_slug = pos.get("eventSlug") or pos.get("slug") or ""

        records.append(
            {
                "wallet":            pos["_wallet"],
                "rank":              pos["_rank"],
                "username":          pos["_username"],
                "condition_id":      cid,
                "question":          title,
                "category":          _classify_category(event_slug, title),
                "side":              str(outcome),
                "current_prob":      cur_price,
                "size_shares":       size,
                "current_value_usd": value,
            }
        )

    if skipped:
        print(f"  Skipped {skipped} records with no condition_id.")

    df = pd.DataFrame(records)

    if df.empty:
        print("  Warning: could not parse any positions.", file=sys.stderr)
    else:
        print(
            f"  Structured {len(df):,} positions | "
            f"{df['wallet'].nunique()} traders | "
            f"{df['condition_id'].nunique()} markets"
        )
        df.to_parquet(POSITIONS_FILE, index=False)
        print(f"  Saved -> {POSITIONS_FILE}")

    return df


if __name__ == "__main__":
    pull_all_positions()
