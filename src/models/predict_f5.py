"""
Predict F5 (first 5 innings) total runs for an upcoming game.

Usage:
  python -m src.models.predict_f5 \
      --home "Yankees" --away "Red Sox" \
      --home-starter "Gerrit Cole" --away-starter "Brayan Bello" \
      --line 4.5
"""

import argparse
import os
import pickle

from scipy.stats import norm

from src.models.predict import build_feature_vector, normalize_team

MODELS_DIR = "data/models"


def load_f5_model():
    path = f"{MODELS_DIR}/model_f5.pkl"
    if not os.path.exists(path):
        raise FileNotFoundError(f"No F5 model found at {path}. Run train_f5.py first.")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_f5(home_team, away_team, home_starter=None, away_starter=None, line=None):
    home_team = normalize_team(home_team)
    away_team = normalize_team(away_team)

    bundle = load_f5_model()
    model        = bundle["model"]
    scaler       = bundle["scaler"]
    feature_cols = bundle["features"]
    std_residual = bundle["std_residual"]

    X = build_feature_vector(home_team, away_team, home_starter, away_starter, feature_cols)
    X_scaled = scaler.transform(X)
    expected_total = float(model.predict(X_scaled)[0])
    expected_total = max(0.0, expected_total)

    # Over/under probabilities via normal approximation
    result = {
        "home_team":      home_team,
        "away_team":      away_team,
        "expected_total": round(expected_total, 2),
    }

    print(f"\n{'='*52}")
    print(f"  {home_team} vs {away_team}  —  F5 Runs")
    if home_starter or away_starter:
        hs = home_starter or "Unknown"
        as_ = away_starter or "Unknown"
        print(f"  Starters: {hs} vs {as_}")
    print(f"{'='*52}")
    print(f"  Expected F5 total:  {expected_total:.1f} runs")

    if line is not None:
        over_prob  = float(1 - norm.cdf(line, loc=expected_total, scale=std_residual))
        under_prob = float(norm.cdf(line, loc=expected_total, scale=std_residual))
        result["line"]       = line
        result["over_prob"]  = round(over_prob, 4)
        result["under_prob"] = round(under_prob, 4)
        print(f"  Line:               {line}")
        print(f"  Over  {line}:        {over_prob*100:.1f}%")
        print(f"  Under {line}:        {under_prob*100:.1f}%")
        lean = "OVER" if over_prob > under_prob else "UNDER"
        conf = max(over_prob, under_prob)
        print(f"\n  Lean: {lean} ({conf*100:.1f}% confidence)")

    print(f"{'='*52}\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",         required=True)
    parser.add_argument("--away",         required=True)
    parser.add_argument("--home-starter", default=None)
    parser.add_argument("--away-starter", default=None)
    parser.add_argument("--line",         type=float, default=None, help="Over/under line (e.g. 4.5)")
    args = parser.parse_args()

    predict_f5(
        home_team=args.home,
        away_team=args.away,
        home_starter=args.home_starter,
        away_starter=args.away_starter,
        line=args.line,
    )
