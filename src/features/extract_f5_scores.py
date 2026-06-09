"""
Extract F5 (first 5 innings) run totals from play-by-play data.

Sums runs_scored for plays where inning <= 5, split by home/away team.
Outputs data/features/f5_scores.json: {game_pk: {home_f5, away_f5, total_f5, home_team, away_team}}

Usage:
  python -m src.features.extract_f5_scores
"""

import json
import os
from collections import defaultdict

PLAY_FILES = [
    "data/historical/plays_2023.json",
    "data/historical/plays_2024.json",
    "data/historical/plays_2025.json",
    "data/full_ingestion.json",
]
OUTPUT = "data/features/f5_scores.json"


def extract(play_files):
    # {game_pk: {home_team, away_team, home_f5, away_f5}}
    games = {}
    runs = defaultdict(lambda: {"home_f5": 0, "away_f5": 0})

    for path in play_files:
        if not os.path.exists(path):
            print(f"  Skipping {path} (not found)")
            continue
        with open(path, encoding="utf-8", errors="ignore") as f:
            plays = json.load(f)
        print(f"  Loaded {len(plays):,} plays from {path}")

        for play in plays:
            pk = str(play.get("game_pk", ""))
            if not pk:
                continue

            # Record team metadata once per game
            if pk not in games:
                games[pk] = {
                    "home_team": play.get("home_team", ""),
                    "away_team": play.get("away_team", ""),
                }

            inning = play.get("inning", 99)
            if inning > 5:
                continue

            scored = play.get("runs_scored", 0) or 0
            if scored <= 0:
                continue

            batting_team = play.get("team", "")
            home_team = play.get("home_team", "")
            if batting_team == home_team:
                runs[pk]["home_f5"] += scored
            else:
                runs[pk]["away_f5"] += scored

    result = {}
    for pk, meta in games.items():
        r = runs[pk]
        result[pk] = {
            "home_team": meta["home_team"],
            "away_team": meta["away_team"],
            "home_f5":   r["home_f5"],
            "away_f5":   r["away_f5"],
            "total_f5":  r["home_f5"] + r["away_f5"],
        }

    return result


def run():
    os.makedirs("data/features", exist_ok=True)
    print("Extracting F5 scores...")
    result = extract(PLAY_FILES)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f)
    totals = [v["total_f5"] for v in result.values()]
    print(f"Saved {len(result):,} games -> {OUTPUT}")
    if totals:
        avg = sum(totals) / len(totals)
        print(f"Average F5 total: {avg:.2f} runs  (min={min(totals)}, max={max(totals)})")


if __name__ == "__main__":
    run()
