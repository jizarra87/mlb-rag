"""
Train a game outcome prediction model.

Strategy:
  - Train on 2025 season, evaluate on 2026 season (time-based split,
    no data leakage)
  - Baseline: logistic regression
  - Main model: gradient boosting (sklearn)
  - Saves trained model to data/models/model.pkl

Usage:
  python -m src.models.train
  python -m src.models.train --features data/features/features_all.csv
"""

import argparse
import csv
import os
import pickle

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler

MODELS_DIR = "data/models"

FEATURE_COLS = [
    # team quality
    "home_win_pct", "away_win_pct", "win_pct_diff",
    # run scoring
    "home_avg_runs", "away_avg_runs", "avg_runs_diff",
    # pitcher stats (from plays)
    "home_era", "away_era", "era_diff",
    "home_k9", "away_k9",
    "home_bb9", "away_bb9",
    "home_whip", "away_whip", "whip_diff",
    # team offense (from plays)
    "home_ops", "away_ops", "ops_diff",
    "home_obp", "away_obp",
    "home_slg", "away_slg",
    "home_runs_pg", "away_runs_pg", "runs_pg_diff",
]

SPLIT_DATE = "2026-01-01"  # train = 2023+2024+2025, test = 2026


def load_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def to_floats(rows, cols):
    X, y = [], []
    for r in rows:
        try:
            X.append([float(r[c]) for c in cols])
            y.append(int(r["target"]))
        except (ValueError, KeyError):
            continue
    return X, y


def evaluate(name, model, X, y):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    acc = accuracy_score(y, preds)
    auc = roc_auc_score(y, probs)
    ll = log_loss(y, probs)
    print(f"  {name:30s}  acc={acc:.3f}  auc={auc:.3f}  log_loss={ll:.3f}")
    return acc, auc


def run(features_path):
    os.makedirs(MODELS_DIR, exist_ok=True)

    print(f"Loading features from {features_path}...")
    rows = load_csv(features_path)
    print(f"Total rows: {len(rows)}")

    train_rows = [r for r in rows if r["date"] < SPLIT_DATE]
    test_rows  = [r for r in rows if r["date"] >= SPLIT_DATE]
    print(f"Train: {len(train_rows)} games (before {SPLIT_DATE})")
    print(f"Test:  {len(test_rows)} games (from {SPLIT_DATE})")

    X_train, y_train = to_floats(train_rows, FEATURE_COLS)
    X_test,  y_test  = to_floats(test_rows,  FEATURE_COLS)

    if not X_train or not X_test:
        print("Not enough data to train. Aborting.")
        return

    # Scale features (needed for logistic regression)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("\nTraining models...")

    # Baseline
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train_sc, y_train)

    # Main model
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
    )
    gb.fit(X_train, y_train)

    print("\n--- Train performance ---")
    evaluate("Logistic Regression", lr, X_train_sc, y_train)
    evaluate("Gradient Boosting",   gb, X_train,    y_train)

    print("\n--- Test performance (2026) ---")
    evaluate("Logistic Regression", lr, X_test_sc, y_test)
    evaluate("Gradient Boosting",   gb, X_test,    y_test)

    # Feature importance
    print("\n--- Feature importances (Gradient Boosting) ---")
    for col, imp in sorted(zip(FEATURE_COLS, gb.feature_importances_), key=lambda x: -x[1]):
        bar = "█" * int(imp * 100)
        print(f"  {col:30s} {imp:.4f}  {bar}")

    # Home win rate baseline
    home_win_rate = sum(y_test) / len(y_test)
    print(f"\nNaive baseline (always predict home win): {home_win_rate:.3f}")

    # Save best model + scaler
    model_path = f"{MODELS_DIR}/model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump({"model": gb, "scaler": scaler, "features": FEATURE_COLS}, f)
    print(f"\nModel saved → {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features",
        default="data/features/features_all.csv",
        help="Path to features CSV (default: data/features/features_all.csv)"
    )
    args = parser.parse_args()
    run(args.features)
