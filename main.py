"""
Polymarket Consensus Tracker - Orchestrator.

Runs all four components in sequence and writes a human-readable
consensus_report.txt to the project root.

Usage:
    python main.py
"""

from datetime import datetime, timezone

import pandas as pd

from config import (
    CONSENSUS_THRESHOLD, REPORT_FILE, TOP_N_TRADERS,
)
from consensus import detect_consensus
from leaderboard import fetch_leaderboard
from positions import pull_all_positions
from tracker import build_accuracy_table, update_resolution_tracker


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_usd(v: float) -> str:
    v = float(v or 0)
    if abs(v) >= 1_000_000:
        return f"${v / 1_000_000:.1f}M"
    if abs(v) >= 1_000:
        return f"${v / 1_000:.0f}K"
    return f"${v:.2f}"


def _pct(v) -> str:
    try:
        return f"{float(v):.1%}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------

def generate_report(
    traders_df:   pd.DataFrame,
    positions_df: pd.DataFrame,
    consensus_df: pd.DataFrame,
    tracker_df:   pd.DataFrame,
) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    W = 72
    lines: list[str] = []

    def h1(text: str) -> None:
        lines.append("=" * W)
        lines.append(f"  {text}")
        lines.append("=" * W)

    def h2(text: str) -> None:
        lines.append(text)
        lines.append("-" * min(len(text) + 2, W))

    def blank() -> None:
        lines.append("")

    # -- Title ---------------------------------------------------------------
    h1("POLYMARKET TOP TRADER CONSENSUS REPORT")
    lines.append(f"  Generated : {now}")
    lines.append(f"  Traders   : top {TOP_N_TRADERS} by all-time PnL")
    lines.append(f"  Consensus : {CONSENSUS_THRESHOLD}+ top traders on the same side")
    blank()

    # -- Section 1: Leaderboard ----------------------------------------------
    h2("SECTION 1 -- TOP TRADERS")
    blank()
    if traders_df.empty:
        lines.append("  No trader data available.")
    else:
        lines.append(
            f"  {'Rank':<6}{'Username':<28}{'All-Time PnL':>14}{'Volume':>14}"
        )
        lines.append("  " + "-" * 62)
        for _, t in traders_df.head(25).iterrows():
            name = str(t.get("username") or t.get("wallet", "")[:14])[:27]
            pnl  = _fmt_usd(t.get("pnl_usd", 0))
            vol  = _fmt_usd(t.get("volume_usd", 0))
            lines.append(f"  {str(t['rank']):<6}{name:<28}{pnl:>14}{vol:>14}")
        if len(traders_df) > 25:
            lines.append(f"  ... {len(traders_df) - 25} more traders in data/traders.parquet")
    blank()

    # -- Section 2: Position coverage ----------------------------------------
    h2("SECTION 2 -- OPEN POSITION COVERAGE")
    blank()
    if positions_df.empty:
        lines.append("  No open positions found.")
    else:
        n_traders  = positions_df["wallet"].nunique()
        n_pos      = len(positions_df)
        n_markets  = positions_df["condition_id"].nunique()
        total_val  = positions_df["current_value_usd"].sum()
        lines += [
            f"  Traders with open positions : {n_traders} / {len(traders_df)}",
            f"  Total position records      : {n_pos:,}",
            f"  Unique markets covered      : {n_markets:,}",
            f"  Combined position value     : {_fmt_usd(total_val)}",
        ]
        blank()
        if "category" in positions_df.columns:
            cat = (
                positions_df
                .groupby("category")
                .agg(count=("condition_id", "count"), val=("current_value_usd", "sum"))
                .sort_values("val", ascending=False)
            )
            lines.append(f"  {'Category':<24}{'Positions':>10}{'Value':>14}")
            lines.append("  " + "-" * 50)
            for cat_name, row in cat.iterrows():
                lines.append(
                    f"  {str(cat_name):<24}{int(row['count']):>10}{_fmt_usd(row['val']):>14}"
                )
    blank()

    # -- Section 3: Consensus positions --------------------------------------
    h2(f"SECTION 3 -- CONSENSUS POSITIONS  ({CONSENSUS_THRESHOLD}+ TRADERS AGREEING)")
    blank()

    strong = (
        consensus_df[consensus_df["is_consensus"]].copy()
        if not consensus_df.empty else pd.DataFrame()
    )

    if strong.empty:
        lines.append(
            f"  No markets found with {CONSENSUS_THRESHOLD}+ top traders on the same side."
        )
        if not consensus_df.empty:
            near = consensus_df[consensus_df["trader_count"] >= 3].head(10)
            if not near.empty:
                blank()
                lines.append("  Near-consensus (3-4 traders agreeing):")
                lines.append(f"  {'Market':<52}{'Side':<6}{'N':>4}")
                lines.append("  " + "-" * 64)
                for _, r in near.iterrows():
                    q = str(r["question"])[:51]
                    lines.append(f"  {q:<52}{str(r['side']):<6}{int(r['trader_count']):>4}")
    else:
        # Apply filters: skip near-certainties, longshots, and tiny positions
        filtered = strong[
            (strong["avg_prob"] > 0.05) &
            (strong["avg_prob"] < 0.95) &
            (strong["total_value_usd"] >= 1_000)
        ].copy()

        # Sort by consensus_strength (net lead over other sides) descending
        if "consensus_strength" in filtered.columns:
            filtered = filtered.sort_values("consensus_strength", ascending=False)

        n_total    = len(strong)
        n_filtered = len(filtered)
        n_mkt      = filtered["condition_id"].nunique()
        lines.append(
            f"  {n_total} raw consensus positions -> {n_filtered} after filters "
            f"(prob 5-95%, value >= $1K) across {n_mkt} markets."
        )
        blank()

        if filtered.empty:
            lines.append(
                "  All consensus positions were filtered out "
                "(near-certainties, longshots, or low-value)."
            )
            blank()
            lines.append("  Top 10 raw positions (for reference):")
            hdr = f"  {'Market':<52}{'Side':<14}{'Traders':>7}{'Prob':>7}{'Value':>10}"
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))
            for _, r in strong.head(10).iterrows():
                lines.append(
                    f"  {str(r['question'])[:51]:<52}{str(r['side'])[:13]:<14}"
                    f"{int(r['trader_count']):>7}{_pct(r['avg_prob']):>7}"
                    f"{_fmt_usd(r['total_value_usd']):>10}"
                )
        else:
            hdr = (
                f"  {'Market':<50}{'Side':<14}{'Strength':>8}"
                f"{'Traders':>8}{'Prob':>7}{'Value':>10}"
            )
            lines.append(hdr)
            lines.append("  " + "-" * (len(hdr) - 2))

            for _, r in filtered.iterrows():
                lines.append(
                    f"  {str(r['question'])[:49]:<50}{str(r['side'])[:13]:<14}"
                    f"{int(r.get('consensus_strength', r['trader_count'])):>8}"
                    f"{int(r['trader_count']):>8}{_pct(r['avg_prob']):>7}"
                    f"{_fmt_usd(r['total_value_usd']):>10}"
                )

            # Show top split markets (cap at 10 to avoid wall of text)
            splits = filtered[filtered["traders_split"]]
            if not splits.empty:
                blank()
                lines.append("  Trader disagreements (both sides 3+ traders, top 10 by strength):")
                shown = 0
                for cid in splits.sort_values("consensus_strength", ascending=False)["condition_id"].unique():
                    if shown >= 10:
                        break
                    rows  = consensus_df[consensus_df["condition_id"] == cid]
                    q     = str(rows.iloc[0]["question"])[:60]
                    sides = ", ".join(
                        f"{r['side']} ({int(r['trader_count'])})"
                        for _, r in rows.sort_values("trader_count", ascending=False).iterrows()
                    )
                    lines.append(f"    {q}  [{sides}]")
                    shown += 1

        # Brief filter summary (no full dump)
        removed = strong[
            ~(
                (strong["avg_prob"] > 0.05) &
                (strong["avg_prob"] < 0.95) &
                (strong["total_value_usd"] >= 1_000)
            )
        ]
        if not removed.empty:
            n_nc  = int((removed["avg_prob"] >= 0.95).sum())
            n_ls  = int((removed["avg_prob"] <= 0.05).sum())
            n_lv  = int(
                ((removed["avg_prob"] > 0.05) & (removed["avg_prob"] < 0.95) &
                 (removed["total_value_usd"] < 1_000)).sum()
            )
            blank()
            lines.append(
                f"  Removed: {n_nc} near-certainties (>95%), "
                f"{n_ls} longshots (<5%), "
                f"{n_lv} low-value (<$1K)."
            )
    blank()

    # -- Section 4: Resolution accuracy -------------------------------------
    h2("SECTION 4 -- RESOLUTION ACCURACY")
    blank()

    if tracker_df.empty:
        lines.append("  No resolution data available yet.")
    else:
        resolved   = tracker_df[tracker_df["resolved"]]
        unresolved = tracker_df[~tracker_df["resolved"]]
        lines += [
            f"  Consensus markets tracked   : {len(tracker_df)}",
            f"  Resolved                    : {len(resolved)}",
            f"  Awaiting resolution         : {len(unresolved)}",
        ]
        blank()

        if not resolved.empty:
            n_correct = int((resolved["consensus_correct"] == True).sum())
            n_total   = len(resolved)
            pct_str   = f"{100 * n_correct / n_total:.1f}%"
            lines.append(f"  OVERALL ACCURACY : {n_correct} / {n_total}  ({pct_str})")
            blank()

            acc = build_accuracy_table(tracker_df)
            if not acc.empty:
                has_edge = "edge_vs_market" in acc.columns
                lines.append("  Accuracy by category:")
                hdr_acc = (
                    f"  {'Category':<22}{'Markets':>8}{'Correct':>8}{'Accuracy':>10}"
                    + ("{'Mkt Prob':>10}{'Edge':>8}" if has_edge else "")
                )
                lines.append(
                    f"  {'Category':<22}{'Markets':>8}{'Correct':>8}{'Accuracy':>10}"
                    + ("  {'Mkt Prob':>8}{'Edge':>7}" if has_edge else "")
                )
                lines.append("  " + "-" * (52 + (17 if has_edge else 0)))
                for _, row in acc.iterrows():
                    base = (
                        f"  {str(row['category']):<22}"
                        f"{int(row['total_markets']):>8}"
                        f"{int(row['correct']):>8}"
                        f"{row['accuracy_pct']:>9.1f}%"
                    )
                    if has_edge:
                        sign = "+" if row["edge_vs_market"] >= 0 else ""
                        base += (
                            f"  {100*row['avg_market_prob']:>7.1f}%"
                            f"  {sign}{row['edge_vs_market']:>5.1f}pp"
                        )
                    lines.append(base)
                blank()

            # Recent resolutions
            recent = (
                resolved
                .sort_values("consensus_trader_count", ascending=False)
                .head(15)
            )
            lines.append("  Top consensus markets (resolved):")
            lines.append(
                f"  {'Result':<9}{'Market':<48}{'Consensus':>10}{'Outcome':>10}"
            )
            lines.append("  " + "-" * 79)
            for _, row in recent.iterrows():
                result = "CORRECT  " if row["consensus_correct"] else "WRONG    "
                q      = str(row["question"])[:47]
                cons   = f"{row['consensus_side']} ({int(row['consensus_trader_count'])})"
                out    = str(row["winning_outcome"] or "--")
                lines.append(f"  {result:<9}{q:<48}{cons:>10}{out:>10}")
        else:
            lines.append("  No consensus markets have resolved yet.")

    blank()
    h1("END OF REPORT")
    data_files = " | ".join([
        "data/traders.parquet",
        "data/positions.parquet",
        "data/consensus.parquet",
        "data/resolution_tracker.parquet",
    ])
    lines.append(f"  Data files: {data_files}")
    lines.append("=" * W)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  POLYMARKET CONSENSUS TRACKER")
    print("=" * 60)
    print()

    print("[1/4] Leaderboard --")
    traders_df = fetch_leaderboard()

    print("\n[2/4] Positions --")
    positions_df = pull_all_positions(traders_df)

    print("\n[3/4] Consensus detection --")
    consensus_df = detect_consensus(positions_df)

    print("\n[4/4] Resolution tracking --")
    tracker_df = update_resolution_tracker(consensus_df)

    print("\nGenerating report...")
    report = generate_report(traders_df, positions_df, consensus_df, tracker_df)

    REPORT_FILE.write_text(report, encoding="utf-8")
    print(f"Report written -> {REPORT_FILE}\n")
    print(report)


if __name__ == "__main__":
    main()
