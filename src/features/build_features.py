"""
Build a game-level feature table for ML training.

Joins game summaries with pre-computed pitcher and team stats from plays.
All stats are computed strictly from games before the current one (no leakage).
Rolling win% and run averages reset at season boundaries.

Outputs:
  data/features/features_{season}.csv   — per-season feature table
  data/features/features_all.csv        — combined across all seasons

Run compute_player_stats.py first to generate pitcher_stats.json and team_stats.json.

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
WINDOW_TEAM  = 20  # rolling window for win% and runs


def load_summaries(path):
    with open(path) as f:
        data = json.load(f)
    completed = [g for g in data if g.get("winner") is not None]
    completed.sort(key=lambda g: g["date"])
    print(f"Loaded {len(completed)} completed games from {path}")
    return completed


def load_lookup(path):
    if not os.path.exists(path):
        print(f"  Warning: {path} not found — those features will use defaults")
        return {}
    with open(path) as f:
        return json.load(f)


def build_features(summaries, pitcher_lookup, team_lookup):
    team_wins       = defaultdict(list)
    team_runs_scored = defaultdict(list)
    team_runs_allowed = defaultdict(list)
    rows = []

    for game in summaries:
        home         = game["home_team"]
        away         = game["away_team"]
        home_score   = game["home_score"]
        away_score   = game["away_score"]
        home_starter = game.get("home_starter") or "Unknown"
        away_starter = game.get("away_starter") or "Unknown"
        game_pk      = str(game["game_pk"])

        # --- Pythagorean win expectation (more predictive than raw win%) ---
        home_pyth = _pythagorean(team_runs_scored[home], team_runs_allowed[home])
        away_pyth = _pythagorean(team_runs_scored[away], team_runs_allowed[away])

        # --- pitcher stats from plays (diff only) ---
        p_stats = pitcher_lookup.get(game_pk, {})
        home_p  = p_stats.get(home_starter, _default_pitcher())
        away_p  = p_stats.get(away_starter, _default_pitcher())

        # --- team offensive stats from plays (diff only) ---
        t_stats = team_lookup.get(game_pk, {})
        home_t  = t_stats.get(home, _default_team())
        away_t  = t_stats.get(away, _default_team())

        target = 1 if game["winner"] == home else 0

        rows.append({
            "game_pk":      game["game_pk"],
            "date":         game["date"],
            "venue":        game["venue"],
            "home_team":    home,
            "away_team":    away,
            "home_starter": home_starter,
            "away_starter": away_starter,
            # Pythagorean win expectation
            "home_pyth":    round(home_pyth, 4),
            "away_pyth":    round(away_pyth, 4),
            "pyth_diff":    round(home_pyth - away_pyth, 4),
            # pitcher matchup diffs
            "era_diff":     round(home_p["era"]  - away_p["era"],  4),
            "k9_diff":      round(home_p["k9"]   - away_p["k9"],   4),
            "bb9_diff":     round(home_p["bb9"]  - away_p["bb9"],  4),
            "whip_diff":    round(home_p["whip"] - away_p["whip"], 4),
            # team offense — individual columns for predict.py lookup
            "home_ops":     home_t["ops"],
            "away_ops":     away_t["ops"],
            "home_obp":     home_t["obp"],
            "away_obp":     away_t["obp"],
            "home_slg":     home_t["slg"],
            "away_slg":     away_t["slg"],
            "home_runs_pg": home_t["runs_per_game"],
            "away_runs_pg": away_t["runs_per_game"],
            # diffs used by the model
            "ops_diff":     round(home_t["ops"]           - away_t["ops"],           4),
            "obp_diff":     round(home_t["obp"]           - away_t["obp"],           4),
            "slg_diff":     round(home_t["slg"]           - away_t["slg"],           4),
            "runs_pg_diff": round(home_t["runs_per_game"] - away_t["runs_per_game"], 4),
            "target":       target,
        })

        # update rolling state AFTER writing the row
        home_won = 1 if game["winner"] == home else 0
        team_wins[home].append(home_won)
        team_wins[away].append(1 - home_won)
        team_runs_scored[home].append(home_score)
        team_runs_scored[away].append(away_score)
        team_runs_allowed[home].append(away_score)
        team_runs_allowed[away].append(home_score)

    return rows


def _pythagorean(runs_scored, runs_allowed, exp=1.83):
    """
    Pythagorean win expectation: RS^exp / (RS^exp + RA^exp).
    Uses last WINDOW_TEAM games. Returns 0.5 (neutral) with no prior data.
    """
    rs = runs_scored[-WINDOW_TEAM:]
    ra = runs_allowed[-WINDOW_TEAM:]
    if not rs:
        return 0.5
    total_rs = sum(rs)
    total_ra = sum(ra)
    if total_rs + total_ra == 0:
        return 0.5
    rs_exp = total_rs ** exp
    ra_exp = total_ra ** exp
    return rs_exp / (rs_exp + ra_exp)


def _default_pitcher():
    return {"era": 4.50, "k9": 7.5, "bb9": 3.0, "whip": 1.30}


def _default_team():
    return {"avg": 0.250, "obp": 0.320, "slg": 0.400, "ops": 0.720, "runs_per_game": 4.5}


def save_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        print(f"No rows to save to {path}")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} rows → {path}")


def run(seasons):
    os.makedirs(FEATURES_DIR, exist_ok=True)

    pitcher_lookup = load_lookup(f"{FEATURES_DIR}/pitcher_stats.json")
    team_lookup    = load_lookup(f"{FEATURES_DIR}/team_stats.json")

    season_paths = {
        2023: "data/historical/game_summaries_2023.json",
        2024: "data/historical/game_summaries_2024.json",
        2025: "data/historical/game_summaries_2025.json",
        2026: "data/game_summaries.json",
    }

    for season in seasons:
        path = season_paths.get(season)
        if not path or not os.path.exists(path):
            print(f"No summaries file for season {season}, skipping")
            continue
        summaries = load_summaries(path)
        rows = build_features(summaries, pitcher_lookup, team_lookup)
        save_csv(rows, f"{FEATURES_DIR}/features_{season}.csv")

    if len(seasons) > 1:
        all_summaries = []
        for season in seasons:
            path = season_paths.get(season)
            if path and os.path.exists(path):
                all_summaries.extend(load_summaries(path))
        all_summaries.sort(key=lambda g: g["date"])
        combined = build_features(all_summaries, pitcher_lookup, team_lookup)
        save_csv(combined, f"{FEATURES_DIR}/features_all.csv")

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
