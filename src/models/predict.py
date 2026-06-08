"""
Predict win probability for an upcoming game.

Looks up each team's and starter's current rolling stats from the
pre-computed feature lookups, then runs them through the trained model.

Usage:
  python -m src.models.predict \
      --home "Boston Red Sox" --away "New York Yankees" \
      --home-starter "Brayan Bello" --away-starter "Gerrit Cole"

  # Starters are optional — defaults to league-average pitcher stats
  python -m src.models.predict \
      --home "Los Angeles Dodgers" --away "Houston Astros"
"""

import argparse
import json
import os
import pickle

MODELS_DIR   = "data/models"
FEATURES_DIR = "data/features"


def load_model():
    path = f"{MODELS_DIR}/model.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at {path}. Run train.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_latest_team_stats(team):
    """Return the most recent rolling stats for a team from the feature CSV."""
    import csv
    path = f"{FEATURES_DIR}/features_all.csv"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))

    # find last game involving this team
    for row in reversed(rows):
        if row["home_team"] == team:
            return {
                "pyth":        float(row["home_pyth"]),
                "runs_pg":     float(row.get("runs_pg_diff", 0)) + 4.5,  # approx
            }
        if row["away_team"] == team:
            return {
                "pyth":        float(row["away_pyth"]),
                "runs_pg":     4.5 - float(row.get("runs_pg_diff", 0)),
            }
    return None


def load_latest_pitcher_stats(pitcher):
    """Return the most recent rolling stats for a pitcher from pitcher_stats.json."""
    path = f"{FEATURES_DIR}/pitcher_stats.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        lookup = json.load(f)

    # pitcher_stats is keyed by game_pk — scan all games for this pitcher
    last_stats = None
    for game_stats in lookup.values():
        if pitcher in game_stats:
            last_stats = game_stats[pitcher]
    return last_stats


def build_feature_vector(home_team, away_team, home_starter, away_starter, feature_cols):
    DEFAULT_PITCHER = {"era": 4.50, "k9": 7.5, "bb9": 3.0, "whip": 1.30}
    DEFAULT_TEAM    = {"pyth": 0.500, "ops": 0.720, "obp": 0.320, "slg": 0.400, "runs_per_game": 4.5}

    home_t = load_latest_team_stats(home_team)   or DEFAULT_TEAM
    away_t = load_latest_team_stats(away_team)   or DEFAULT_TEAM
    home_p = (load_latest_pitcher_stats(home_starter) or DEFAULT_PITCHER) if home_starter else DEFAULT_PITCHER
    away_p = (load_latest_pitcher_stats(away_starter) or DEFAULT_PITCHER) if away_starter else DEFAULT_PITCHER

    # Build the same feature dict shape as build_features.py
    features = {
        "home_pyth":    home_t["pyth"],
        "away_pyth":    away_t["pyth"],
        "pyth_diff":    round(home_t["pyth"] - away_t["pyth"], 4),
        "era_diff":     round(home_p["era"]  - away_p["era"],  4),
        "k9_diff":      round(home_p["k9"]   - away_p["k9"],   4),
        "bb9_diff":     round(home_p["bb9"]  - away_p["bb9"],  4),
        "whip_diff":    round(home_p["whip"] - away_p["whip"], 4),
        "ops_diff":     0.0,
        "obp_diff":     0.0,
        "slg_diff":     0.0,
        "runs_pg_diff": 0.0,
    }

    return [[features[col] for col in feature_cols]]


def predict(home_team, away_team, home_starter=None, away_starter=None):
    bundle = load_model()
    model       = bundle["model"]
    scaler      = bundle["scaler"]
    feature_cols = bundle["features"]

    X = build_feature_vector(home_team, away_team, home_starter, away_starter, feature_cols)
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[0]

    home_prob = probs[1]
    away_prob = probs[0]

    print(f"\n{'='*50}")
    print(f"  {home_team} (home)  vs  {away_team} (away)")
    if home_starter:
        print(f"  Starters: {home_starter} vs {away_starter or 'Unknown'}")
    print(f"{'='*50}")
    print(f"  {home_team:<30} {home_prob*100:.1f}%")
    print(f"  {away_team:<30} {away_prob*100:.1f}%")
    print(f"{'='*50}")
    winner = home_team if home_prob > away_prob else away_team
    conf = max(home_prob, away_prob)
    print(f"  Predicted winner: {winner} ({conf*100:.1f}% confidence)")
    print()

    return {"home_team": home_team, "away_team": away_team,
            "home_prob": round(home_prob, 4), "away_prob": round(away_prob, 4)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",         required=True,  help="Home team name")
    parser.add_argument("--away",         required=True,  help="Away team name")
    parser.add_argument("--home-starter", default=None,   help="Home starting pitcher")
    parser.add_argument("--away-starter", default=None,   help="Away starting pitcher")
    args = parser.parse_args()

    predict(
        home_team=args.home,
        away_team=args.away,
        home_starter=args.home_starter,
        away_starter=args.away_starter,
    )
