from pathlib import Path

# --- API endpoints ---
LEADERBOARD_URL = "https://data-api.polymarket.com/v1/leaderboard"
POSITIONS_URL   = "https://data-api.polymarket.com/positions"
MARKETS_URL     = "https://gamma-api.polymarket.com/markets"

# --- Tuning ---
TOP_N_TRADERS       = 200   # how many leaderboard traders to track (API returns 50/page)
CONSENSUS_THRESHOLD = 5     # min traders on same side to flag as consensus
REQUEST_DELAY       = 0.3   # seconds between API calls (rate-limit courtesy)
POSITIONS_LIMIT     = 500   # max positions fetched per trader

# --- Storage ---
DATA_DIR         = Path("data")
TRADERS_FILE     = DATA_DIR / "traders.parquet"
POSITIONS_FILE   = DATA_DIR / "positions.parquet"
CONSENSUS_FILE   = DATA_DIR / "consensus.parquet"
RESOLUTION_FILE  = DATA_DIR / "resolution_tracker.parquet"
REPORT_FILE      = Path("consensus_report.txt")

HEADERS = {
    "User-Agent": "PolyConsensus/1.0 (Research Tool)",
    "Accept": "application/json",
}
