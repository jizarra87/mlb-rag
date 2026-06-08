"""
Pull full season plays and game summaries for a given MLB season.
Outputs:
  data/historical/plays_{season}.json           — play-by-play events
  data/historical/game_summaries_{season}.json  — one record per game (outcome, score, starters)

Both files support resume: re-running skips already-processed games.

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
    extract_plays,
    extract_game_summary,
)

HISTORICAL_DIR = "data/historical"


def run_historical_ingestion(season: int = 2025):
    os.makedirs(HISTORICAL_DIR, exist_ok=True)

    plays_file = f"{HISTORICAL_DIR}/plays_{season}.json"
    summaries_file = f"{HISTORICAL_DIR}/game_summaries_{season}.json"

    start_date = f"{season}-03-01"
    end_date = f"{season}-11-01"

    print(f"Fetching {season} regular season schedule ({start_date} → {end_date})...")
    game_pks = get_schedule(start_date, end_date)
    print(f"Games found: {len(game_pks)}")

    if os.path.exists(plays_file):
        with open(plays_file, "r") as f:
            all_plays = json.load(f)
        print(f"Resuming: {len(all_plays)} plays already stored")
    else:
        all_plays = []

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

            plays = extract_plays(feed, game_pk)
            all_plays.extend(plays)

            summary = extract_game_summary(feed, game_pk)
            if summary:
                all_summaries.append(summary)

            if (i + 1) % SAVE_EVERY == 0:
                elapsed = time.time() - start_time
                pct = round((i + 1) / len(remaining) * 100, 1)
                print(f"[{pct}%] {i+1}/{len(remaining)} games | {len(all_summaries)} summaries | {len(all_plays)} plays | {round(elapsed)}s | errors: {errors}")
                _save(plays_file, all_plays, summaries_file, all_summaries)

            time.sleep(0.2)

        except Exception as e:
            errors += 1
            print(f"Error game {game_pk}: {e}")

    _save(plays_file, all_plays, summaries_file, all_summaries)
    elapsed = time.time() - start_time
    print(f"\nDone. {len(all_summaries)} summaries | {len(all_plays)} plays | {errors} errors | {round(elapsed)}s")
    return all_plays, all_summaries


def _save(plays_file, plays, summaries_file, summaries):
    with open(plays_file, "w") as f:
        json.dump(plays, f)
    with open(summaries_file, "w") as f:
        json.dump(summaries, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch season game summaries for ML training")
    parser.add_argument("--season", type=int, default=2025, help="MLB season year (default: 2025)")
    args = parser.parse_args()

    run_historical_ingestion(season=args.season)
