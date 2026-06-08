"""
Pull full season game summaries (no play-by-play) for ML feature engineering.
Output:
  data/historical/game_summaries_{season}.json  — one record per game

Usage:
  python -m src.ingestion.historical_ingestion
  python -m src.ingestion.historical_ingestion --season 2024
"""

import argparse
import json
import os
import time

from src.ingestion.mlb_feed_ingestion import (
    get_game_feed,
    get_schedule,
    extract_game_summary,
)

HISTORICAL_DIR = "data/historical"


def run_historical_ingestion(season: int = 2025):
    os.makedirs(HISTORICAL_DIR, exist_ok=True)

    summaries_file = f"{HISTORICAL_DIR}/game_summaries_{season}.json"

    start_date = f"{season}-03-01"
    end_date = f"{season}-11-01"

    print(f"Fetching {season} regular season schedule ({start_date} → {end_date})...")
    game_pks = get_schedule(start_date, end_date)
    print(f"Games found: {len(game_pks)}")

    if os.path.exists(summaries_file):
        with open(summaries_file, "r") as f:
            all_summaries = json.load(f)
        print(f"Resuming: {len(all_summaries)} summaries already stored")
    else:
        all_summaries = []

    processed = set(s["game_pk"] for s in all_summaries)
    remaining = [pk for pk in game_pks if pk not in processed]
    print(f"Games remaining: {len(remaining)}")

    start_time = time.time()
    errors = 0
    SAVE_EVERY = 50

    for i, game_pk in enumerate(remaining):
        try:
            feed = get_game_feed(game_pk)
            if not feed:
                errors += 1
                continue

            summary = extract_game_summary(feed, game_pk)
            if summary:
                all_summaries.append(summary)

            if (i + 1) % SAVE_EVERY == 0:
                elapsed = time.time() - start_time
                pct = round((i + 1) / len(remaining) * 100, 1)
                print(f"[{pct}%] {i+1}/{len(remaining)} games | {len(all_summaries)} summaries | {round(elapsed)}s | errors: {errors}")
                with open(summaries_file, "w") as f:
                    json.dump(all_summaries, f)

            time.sleep(0.2)

        except Exception as e:
            errors += 1
            print(f"Error game {game_pk}: {e}")

    with open(summaries_file, "w") as f:
        json.dump(all_summaries, f)

    elapsed = time.time() - start_time
    print(f"\nDone. {len(all_summaries)} game summaries | {errors} errors | {round(elapsed)}s")
    return all_summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch season game summaries for ML training")
    parser.add_argument("--season", type=int, default=2025, help="MLB season year (default: 2025)")
    args = parser.parse_args()

    run_historical_ingestion(season=args.season)
