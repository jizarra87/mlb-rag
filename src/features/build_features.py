"""
Build a game-level feature table for ML training.

Reads game_summaries from one or more seasons and produces a flat CSV
with rolling team stats and starting pitcher ERA computed strictly from
past games (no data leakage). Rolling state resets at each season boundary.

Outputs:
  data/features/features_{season}.csv   — per-season feature table
  data/features/features_all.csv        — combined across all seasons

Usage:
  python -m src.features.build_features
  python -m src.features.build_features --seasons 2025 2026
"""

import argparse
import csv
import json
import os
from collections import defaultdict

FEATURES_DIR = "data/features"

WINDOW_TEAM = 20    # rolling window for team run avg and win %
WINDOW_PITCHER = 5  # rolling window for pitcher ERA (starts)


def load_summaries(path):
    with open(path) as f:
        data = json.load(f)
    completed = [g for g in data if g.get("winner") is not None]
    completed.sort(key=lambda g: g["date"])
    print(f"Loaded {len(completed)} completed games from {path}")
    return completed


def build_features(summaries):
    """
    Compute rolling features with no data leakage: each row's features
    are derived only from games that happened before it.
    Rolling state is isolated to this call — no cross-season bleed.
    """
    team_wins = defaultdict(list)
    team_runs = defaultdict(list)
    # pitcher: list of runs allowed per start (not multiplied by 9)
    pitcher_runs = defaultdict(list)

    rows = []

    for game in summaries:
        home = game["home_team"]
        away = game["away_team"]
        home_score = game["home_score"]
        away_score = game["away_score"]
        home_starter = game.get("home_starter") or "Unknown"
        away_starter = game.get("away_starter") or "Unknown"

        # --- features from past games only ---
        home_win_pct   = _win_pct(team_wins[home])
        away_win_pct   = _win_pct(team_wins[away])
        home_avg_runs  = _rolling_avg(team_runs[home],  WINDOW_TEAM, default=4.5)
        away_avg_runs  = _rolling_avg(team_runs[away],  WINDOW_TEAM, default=4.5)
        home_era       = _rolling_era(pitcher_runs[home_starter])
        away_era       = _rolling_era(pitcher_runs[away_starter])

        target = 1 if game["winner"] == home else 0

        rows.append({
            "game_pk":          game["game_pk"],
            "date":             game["date"],
            "venue":            game["venue"],
            "home_team":        home,
            "away_team":        away,
            "home_starter":     home_starter,
            "away_starter":     away_starter,
            "home_win_pct":     round(home_win_pct, 4),
            "away_win_pct":     round(away_win_pct, 4),
            "win_pct_diff":     round(home_win_pct - away_win_pct, 4),
            "home_avg_runs":    round(home_avg_runs, 4),
            "away_avg_runs":    round(away_avg_runs, 4),
            "avg_runs_diff":    round(home_avg_runs - away_avg_runs, 4),
            "home_era":         round(home_era, 4),
            "away_era":         round(away_era, 4),
            "era_diff":         round(home_era - away_era, 4),
            # Confidence proxies: how many games in the rolling window
            "target":           target,
        })

        # --- update state AFTER writing this row ---
        home_won = 1 if game["winner"] == home else 0
        team_wins[home].append(home_won)
        team_wins[away].append(1 - home_won)
        team_runs[home].append(home_score)
        team_runs[away].append(away_score)
        # runs allowed = opponent's score
        pitcher_runs[home_starter].append(away_score)
        pitcher_runs[away_starter].append(home_score)

    return rows


def _win_pct(results):
    if not results:
        return 0.5
    window = results[-WINDOW_TEAM:]
    return sum(window) / len(window)


def _rolling_avg(values, window, default=4.5):
    if not values:
        return default
    w = values[-window:]
    return sum(w) / len(w)


def _rolling_era(runs_per_start):
    """
    ERA proxy: average runs allowed per start × 9 ÷ estimated innings.
    Assumes a starter averages ~5.5 innings per outing (MLB average).
    Returns a value on a realistic ERA scale (2–6 typical range).
    """
    if not runs_per_start:
        return 4.50  # league average prior
    window = runs_per_start[-WINDOW_PITCHER:]
    avg_runs = sum(window) / len(window)
    assumed_ip = 5.5
    return (avg_runs / assumed_ip) * 9


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

    for season in seasons:
        path = season_paths.get(season)
        if not path or not os.path.exists(path):
            print(f"No summaries file for season {season}, skipping")
            continue
        summaries = load_summaries(path)
        rows = build_features(summaries)
        save_csv(rows, f"{FEATURES_DIR}/features_{season}.csv")

    if len(seasons) > 1:
        # Combined across seasons — state resets at each season boundary
        # to avoid 2025 momentum bleeding into 2026 early games
        all_rows = []
        for season in seasons:
            path = season_paths.get(season)
            if path and os.path.exists(path):
                summaries = load_summaries(path)
                all_rows.extend(build_features(summaries))
        all_rows.sort(key=lambda r: r["date"])
        save_csv(all_rows, f"{FEATURES_DIR}/features_all.csv")

    print("\nSample row (game 10):")
    sample_path = f"{FEATURES_DIR}/features_{seasons[0]}.csv"
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            rows = list(csv.DictReader(f))
        if len(rows) > 10:
            for k, v in rows[10].items():
                print(f"  {k:25s} {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args()
    run(args.seasons)
