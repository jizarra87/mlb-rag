"""
Backfill game_summaries.json for games already in full_ingestion.json
that were ingested before summary extraction was added.

Usage:
  python -m src.ingestion.backfill_summaries
"""

import json
import os
import time

from src.ingestion.mlb_feed_ingestion import get_game_feed, extract_game_summary

PLAYS_FILE = "data/full_ingestion.json"
SUMMARIES_FILE = "data/game_summaries.json"


def backfill():
    with open(PLAYS_FILE) as f:
        plays = json.load(f)

    all_game_pks = sorted(set(p["game_pk"] for p in plays))
    print(f"Unique games in plays file: {len(all_game_pks)}")

    if os.path.exists(SUMMARIES_FILE):
        with open(SUMMARIES_FILE) as f:
            summaries = json.load(f)
    else:
        summaries = []

    already_done = set(s["game_pk"] for s in summaries)
    missing = [pk for pk in all_game_pks if pk not in already_done]
    print(f"Summaries already stored: {len(already_done)}")
    print(f"Games to backfill: {len(missing)}")

    errors = 0
    start_time = time.time()

    for i, game_pk in enumerate(missing):
        try:
            feed = get_game_feed(game_pk)
            if not feed:
                errors += 1
                continue

            summary = extract_game_summary(feed, game_pk)
            if summary:
                summaries.append(summary)

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                pct = round((i + 1) / len(missing) * 100, 1)
                print(f"[{pct}%] {i+1}/{len(missing)} | {len(summaries)} summaries | {round(elapsed)}s | errors: {errors}")
                with open(SUMMARIES_FILE, "w") as f:
                    json.dump(summaries, f)

            time.sleep(0.2)

        except Exception as e:
            errors += 1
            print(f"Error game {game_pk}: {e}")

    with open(SUMMARIES_FILE, "w") as f:
        json.dump(summaries, f)

    elapsed = time.time() - start_time
    print(f"\nDone. {len(summaries)} total summaries | {errors} errors | {round(elapsed)}s")


if __name__ == "__main__":
    backfill()
