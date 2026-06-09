"""
Train a model to predict F5 (first 5 innings) total runs.

Uses the same game-level features as the win model but targets f5_total.
Trains a regression model for expected total and derives over/under probabilities
by fitting a Poisson distribution to the residuals.

Outputs:
  data/models/model_f5.pkl  — {model, scaler, features, mean_total, std_total}

Usage:
  python -m src.models.train_f5
  python -m src.models.train_f5 --features data/features/features_f5_all.csv
"""

import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "data/models"

FEATURE_COLS = [
    "home_pyth", "away_pyth", "pyth_diff",
    "era_diff", "k9_diff", "bb9_diff", "whip_diff",
    "ops_diff", "obp_diff", "slg_diff", "runs_pg_diff",
]

SPLIT_DATE = "2026-01-01"


def load_csv(path):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return list(csv.DictReader(f))


def to_arrays(rows, cols):
    X, y = [], []
    for r in rows:
        try:
            X.append([float(r[c]) for c in cols])
            y.append(float(r["f5_total"]))
        except (ValueError, KeyError):
            continue
    return np.array(X), np.array(y)


def evaluate(name, model, X, y, scaler=None):
    X_in = scaler.transform(X) if scaler else X
    preds = model.predict(X_in)
    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5
    print(f"  {name:30s}  MAE={mae:.3f}  RMSE={rmse:.3f}")
    return mae, rmse, preds


def run(features_path):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading features from {features_path}...")
    rows = load_csv(features_path)
    print(f"Total rows: {len(rows)}")

    train_rows = [r for r in rows if r["date"] < SPLIT_DATE]
    test_rows  = [r for r in rows if r["date"] >= SPLIT_DATE]
    print(f"Train: {len(train_rows)} games")
    print(f"Test:  {len(test_rows)} games")

    X_train, y_train = to_arrays(train_rows, FEATURE_COLS)
    X_test,  y_test  = to_arrays(test_rows,  FEATURE_COLS)

    if len(X_train) == 0 or len(X_test) == 0:
        print("Not enough data. Aborting.")
        return

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\nTraining models...")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_sc, y_train)

    gb = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    print("\n--- Train performance ---")
    evaluate("Ridge Regression", ridge, X_train, y_train, scaler)
    evaluate("Gradient Boosting", gb, X_train, y_train)

    print("\n--- Test performance (2026) ---")
    _, _, ridge_preds = evaluate("Ridge Regression", ridge, X_test, y_test, scaler)
    _, _, gb_preds    = evaluate("Gradient Boosting", gb, X_test, y_test)

    # Naive baseline
    naive_pred = y_train.mean()
    naive_mae = mean_absolute_error(y_test, [naive_pred] * len(y_test))
    print(f"\n  Naive baseline (always predict {naive_pred:.2f}): MAE={naive_mae:.3f}")

    print("\n--- Feature importances (Gradient Boosting) ---")
    for col, imp in sorted(zip(FEATURE_COLS, gb.feature_importances_), key=lambda x: -x[1]):
        bar = "|" * int(imp * 100)
        print(f"  {col:30s} {imp:.4f}  {bar}")

    # Distribution stats for over/under probability estimation
    residuals = y_train - ridge.predict(X_train_sc)
    std_residual = float(np.std(residuals))
    mean_total = float(y_train.mean())

    print(f"\nResidual std (used for over/under): {std_residual:.3f}")
    print(f"Mean F5 total (training): {mean_total:.3f}")

    model_path = f"{MODELS_DIR}/model_f5.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({
            "model":        ridge,
            "scaler":       scaler,
            "features":     FEATURE_COLS,
            "std_residual": std_residual,
            "mean_total":   mean_total,
        }, f)
    print(f"\nModel saved -> {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/features/features_f5_all.csv")
    args = parser.parse_args()
    run(args.features)
