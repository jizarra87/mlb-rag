"""
Build a game-level feature table for ML training.

Reads game_summaries from one or more seasons and produces a flat CSV
with rolling team stats and starting pitcher ERA computed strictly from
past games (no data leakage).

Outputs:
  data/features/features_{season}.csv   — per-season feature table
  data/features/features_all.csv        — combined across all seasons

Usage:
  python -m src.features.build_features
  python -m src.features.build_features --seasons 2025 2026
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import csv

FEATURES_DIR = "data/features"

WINDOW_TEAM = 10    # rolling window for team run avg and win %
WINDOW_PITCHER = 5  # rolling window for pitcher ERA


def load_summaries(paths):
    summaries = []
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        completed = [g for g in data if g.get("winner") is not None]
        summaries.extend(completed)
        print(f"Loaded {len(completed)} completed games from {path}")
    summaries.sort(key=lambda g: g["date"])
    return summaries


def build_features(summaries):
    # Rolling state tracked per team and pitcher — only past games
    team_wins = defaultdict(list)       # team -> [1/0, ...] per game in order
    team_runs = defaultdict(list)       # team -> [runs_scored, ...]
    pitcher_era_data = defaultdict(list) # pitcher -> [(ip_approx, earned_runs), ...]

    rows = []

    for game in summaries:
        home = game["home_team"]
        away = game["away_team"]
        home_score = game["home_score"]
        away_score = game["away_score"]
        home_starter = game.get("home_starter") or "Unknown"
        away_starter = game.get("away_starter") or "Unknown"

        # --- compute features from PAST games only ---
        home_win_pct = _win_pct(team_wins[home])
        away_win_pct = _win_pct(team_wins[away])

        home_avg_runs = _rolling_avg(team_runs[home], WINDOW_TEAM)
        away_avg_runs = _rolling_avg(team_runs[away], WINDOW_TEAM)

        home_starter_era = _pitcher_era(pitcher_era_data[home_starter])
        away_starter_era = _pitcher_era(pitcher_era_data[away_starter])

        home_games_played = len(team_wins[home])
        away_games_played = len(team_wins[away])

        target = 1 if game["winner"] == home else 0

        rows.append({
            "game_pk": game["game_pk"],
            "date": game["date"],
            "venue": game["venue"],
            "home_team": home,
            "away_team": away,
            "home_starter": home_starter,
            "away_starter": away_starter,
            "home_win_pct": round(home_win_pct, 4),
            "away_win_pct": round(away_win_pct, 4),
            "win_pct_diff": round(home_win_pct - away_win_pct, 4),
            "home_avg_runs": round(home_avg_runs, 4),
            "away_avg_runs": round(away_avg_runs, 4),
            "avg_runs_diff": round(home_avg_runs - away_avg_runs, 4),
            "home_starter_era": round(home_starter_era, 4),
            "away_starter_era": round(away_starter_era, 4),
            "era_diff": round(home_starter_era - away_starter_era, 4),
            "home_games_played": home_games_played,
            "away_games_played": away_games_played,
            "target": target,
        })

        # --- update rolling state AFTER computing features ---
        home_won = 1 if game["winner"] == home else 0
        team_wins[home].append(home_won)
        team_wins[away].append(1 - home_won)
        team_runs[home].append(home_score)
        team_runs[away].append(away_score)

        # Approximate innings pitched: assume ~27 outs per 9 innings,
        # each at-bat ~= 1 out faced. We use game runs allowed as ERA proxy.
        pitcher_era_data[home_starter].append(away_score)  # runs allowed by home starter
        pitcher_era_data[away_starter].append(home_score)

    return rows


def _win_pct(results):
    if not results:
        return 0.5  # neutral prior for first game
    window = results[-WINDOW_TEAM:]
    return sum(window) / len(window)


def _rolling_avg(values, window):
    if not values:
        return 4.5  # MLB average runs per game
    return sum(values[-window:]) / len(values[-window:])


def _pitcher_era(runs_allowed_list):
    if not runs_allowed_list:
        return 4.50  # league average ERA prior
    window = runs_allowed_list[-WINDOW_PITCHER:]
    # Approximate: ERA = (runs_allowed / games) * 9
    return (sum(window) / len(window)) * 9


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        print(f"No rows to save to {path}")
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


def run(seasons):
    os.makedirs(FEATURES_DIR, exist_ok=True)

    season_paths = {
        2025: "data/historical/game_summaries_2025.json",
        2026: "data/game_summaries.json",
    }

    all_rows = []

    for season in seasons:
        path = season_paths.get(season)
        if not path or not os.path.exists(path):
            print(f"No summaries file found for season {season}, skipping")
            continue

        summaries = load_summaries([path])
        rows = build_features(summaries)
        save_csv(rows, f"{FEATURES_DIR}/features_{season}.csv")
        all_rows.extend(rows)

    if len(seasons) > 1:
        # Combined: re-build in strict date order across seasons
        all_summaries = load_summaries([season_paths[s] for s in seasons if os.path.exists(season_paths.get(s, ""))])
        combined_rows = build_features(all_summaries)
        save_csv(combined_rows, f"{FEATURES_DIR}/features_all.csv")

    print("\nSample row:")
    if all_rows:
        for k, v in list(all_rows[10].items()):
            print(f"  {k}: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args()
    run(args.seasons)
