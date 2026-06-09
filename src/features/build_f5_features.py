"""
Build F5 (first 5 innings) feature table by joining F5 scores onto existing game features.

Reads:
  data/features/features_all.csv   — game-level features from build_features.py
  data/features/f5_scores.json     — F5 run totals from extract_f5_scores.py

Outputs:
  data/features/features_f5_all.csv — same rows with added f5_home, f5_away, f5_total columns

Usage:
  python -m src.features.build_f5_features
"""

import csv
import json
import os

FEATURES_DIR = "data/features"


def run():
    features_path = f"{FEATURES_DIR}/features_all.csv"
    f5_path = f"{FEATURES_DIR}/f5_scores.json"

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"{features_path} not found — run build_features.py first")
    if not os.path.exists(f5_path):
        raise FileNotFoundError(f"{f5_path} not found — run extract_f5_scores.py first")

    with open(features_path, encoding="utf-8", errors="ignore") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows):,} rows from {features_path}")

    with open(f5_path, encoding="utf-8") as f:
        f5_lookup = json.load(f)
    print(f"Loaded {len(f5_lookup):,} F5 score entries")

    out_rows = []
    skipped = 0
    for row in rows:
        pk = str(row["game_pk"])
        if pk not in f5_lookup:
            skipped += 1
            continue
        f5 = f5_lookup[pk]
        row["f5_home"]  = f5["home_f5"]
        row["f5_away"]  = f5["away_f5"]
        row["f5_total"] = f5["total_f5"]
        out_rows.append(row)

    print(f"Matched {len(out_rows):,} rows  (skipped {skipped} without F5 data)")

    out_path = f"{FEATURES_DIR}/features_f5_all.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Saved -> {out_path}")

    totals = [float(r["f5_total"]) for r in out_rows]
    avg = sum(totals) / len(totals)
    buckets = {}
    for t in totals:
        b = int(t)
        buckets[b] = buckets.get(b, 0) + 1
    print(f"\nF5 total distribution (avg={avg:.2f}):")
    for k in sorted(buckets):
        bar = "#" * (buckets[k] // 30)
        print(f"  {k:2d} runs: {buckets[k]:4d}  {bar}")


if __name__ == "__main__":
    run()
