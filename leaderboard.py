"""
Component 1 - Leaderboard scraper.
Fetches the top N traders from Polymarket by all-time PnL and stores
their wallet addresses in data/traders.parquet.
"""

import sys
import time
import requests
import pandas as pd
from config import LEADERBOARD_URL, TOP_N_TRADERS, TRADERS_FILE, HEADERS, DATA_DIR, REQUEST_DELAY


_PAGE_SIZE = 50  # API hard-caps each page at 50 results


def fetch_leaderboard(top_n: int = TOP_N_TRADERS) -> pd.DataFrame:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"Fetching top {top_n} traders from Polymarket leaderboard (all-time PnL)...")

    data: list[dict] = []
    offset = 0
    while len(data) < top_n:
        want = min(_PAGE_SIZE, top_n - len(data))
        params = {"window": "all", "limit": want, "offset": offset}
        resp = requests.get(LEADERBOARD_URL, params=params, headers=HEADERS, timeout=30)
        resp.raise_for_status()
        page = resp.json() or []
        if not page:
            break
        data.extend(page)
        print(f"  Page offset={offset}: got {len(page)} traders (total so far: {len(data)})")
        if len(page) < want:
            break  # last page
        offset += len(page)
        time.sleep(REQUEST_DELAY)

    if not data:
        print("ERROR: Leaderboard returned empty data.", file=sys.stderr)
        return pd.DataFrame()

    df = pd.DataFrame(data)

    df = df.rename(columns={
        "proxyWallet":   "wallet",
        "userName":      "username",
        "vol":           "volume_usd",
        "pnl":           "pnl_usd",
        "xUsername":     "twitter_handle",
        "verifiedBadge": "verified",
        "profileImage":  "avatar_url",
    })

    wanted = ["rank", "wallet", "username", "volume_usd", "pnl_usd",
              "twitter_handle", "verified"]
    df = df[[c for c in wanted if c in df.columns]].copy()

    df["rank"]       = pd.to_numeric(df["rank"],       errors="coerce").astype("Int64")
    df["volume_usd"] = pd.to_numeric(df["volume_usd"], errors="coerce")
    df["pnl_usd"]    = pd.to_numeric(df["pnl_usd"],    errors="coerce")

    df.to_parquet(TRADERS_FILE, index=False)
    print(f"  Saved {len(df)} traders -> {TRADERS_FILE}")
    return df


if __name__ == "__main__":
    fetch_leaderboard()
