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

# Map short/common names to full team names as stored in the CSV
TEAM_NAME_MAP = {
    "yankees": "New York Yankees",
    "new york yankees": "New York Yankees",
    "red sox": "Boston Red Sox",
    "boston red sox": "Boston Red Sox",
    "dodgers": "Los Angeles Dodgers",
    "los angeles dodgers": "Los Angeles Dodgers",
    "astros": "Houston Astros",
    "houston astros": "Houston Astros",
    "braves": "Atlanta Braves",
    "atlanta braves": "Atlanta Braves",
    "mets": "New York Mets",
    "new york mets": "New York Mets",
    "cubs": "Chicago Cubs",
    "chicago cubs": "Chicago Cubs",
    "white sox": "Chicago White Sox",
    "chicago white sox": "Chicago White Sox",
    "cardinals": "St. Louis Cardinals",
    "st. louis cardinals": "St. Louis Cardinals",
    "st louis cardinals": "St. Louis Cardinals",
    "giants": "San Francisco Giants",
    "san francisco giants": "San Francisco Giants",
    "padres": "San Diego Padres",
    "san diego padres": "San Diego Padres",
    "phillies": "Philadelphia Phillies",
    "philadelphia phillies": "Philadelphia Phillies",
    "rangers": "Texas Rangers",
    "texas rangers": "Texas Rangers",
    "angels": "Los Angeles Angels",
    "los angeles angels": "Los Angeles Angels",
    "mariners": "Seattle Mariners",
    "seattle mariners": "Seattle Mariners",
    "twins": "Minnesota Twins",
    "minnesota twins": "Minnesota Twins",
    "rays": "Tampa Bay Rays",
    "tampa bay rays": "Tampa Bay Rays",
    "orioles": "Baltimore Orioles",
    "baltimore orioles": "Baltimore Orioles",
    "guardians": "Cleveland Guardians",
    "cleveland guardians": "Cleveland Guardians",
    "tigers": "Detroit Tigers",
    "detroit tigers": "Detroit Tigers",
    "royals": "Kansas City Royals",
    "kansas city royals": "Kansas City Royals",
    "brewers": "Milwaukee Brewers",
    "milwaukee brewers": "Milwaukee Brewers",
    "pirates": "Pittsburgh Pirates",
    "pittsburgh pirates": "Pittsburgh Pirates",
    "reds": "Cincinnati Reds",
    "cincinnati reds": "Cincinnati Reds",
    "rockies": "Colorado Rockies",
    "colorado rockies": "Colorado Rockies",
    "marlins": "Miami Marlins",
    "miami marlins": "Miami Marlins",
    "nationals": "Washington Nationals",
    "washington nationals": "Washington Nationals",
    "athletics": "Athletics",
    "oakland athletics": "Athletics",
    "a's": "Athletics",
    "blue jays": "Toronto Blue Jays",
    "toronto blue jays": "Toronto Blue Jays",
    "diamondbacks": "Arizona Diamondbacks",
    "arizona diamondbacks": "Arizona Diamondbacks",
    "d-backs": "Arizona Diamondbacks",
}


def normalize_team(name: str) -> str:
    return TEAM_NAME_MAP.get(name.lower().strip(), name)


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

    for row in reversed(rows):
        if row["home_team"] == team:
            return {
                "pyth":         float(row["home_pyth"]),
                "ops":          float(row["home_ops"])  if "home_ops"  in row else 0.720,
                "obp":          float(row["home_obp"])  if "home_obp"  in row else 0.320,
                "slg":          float(row["home_slg"])  if "home_slg"  in row else 0.400,
                "runs_per_game":float(row["home_runs_pg"]) if "home_runs_pg" in row else 4.5,
            }
        if row["away_team"] == team:
            return {
                "pyth":         float(row["away_pyth"]),
                "ops":          float(row["away_ops"])  if "away_ops"  in row else 0.720,
                "obp":          float(row["away_obp"])  if "away_obp"  in row else 0.320,
                "slg":          float(row["away_slg"])  if "away_slg"  in row else 0.400,
                "runs_per_game":float(row["away_runs_pg"]) if "away_runs_pg" in row else 4.5,
            }
    return None


def load_latest_pitcher_stats(pitcher):
    """Return the most recent rolling stats for a pitcher from pitcher_stats.json."""
    path = f"{FEATURES_DIR}/pitcher_stats.json"
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8", errors="ignore") as f:
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

    home_t = load_latest_team_stats(normalize_team(home_team)) or DEFAULT_TEAM
    away_t = load_latest_team_stats(normalize_team(away_team)) or DEFAULT_TEAM
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
        "ops_diff":     round(home_t.get("ops", 0.720)           - away_t.get("ops", 0.720),           4),
        "obp_diff":     round(home_t.get("obp", 0.320)           - away_t.get("obp", 0.320),           4),
        "slg_diff":     round(home_t.get("slg", 0.400)           - away_t.get("slg", 0.400),           4),
        "runs_pg_diff": round(home_t.get("runs_per_game", 4.5)   - away_t.get("runs_per_game", 4.5),   4),
    }

    return [[features[col] for col in feature_cols]]


def predict(home_team, away_team, home_starter=None, away_starter=None):
    home_team = normalize_team(home_team)
    away_team = normalize_team(away_team)

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
