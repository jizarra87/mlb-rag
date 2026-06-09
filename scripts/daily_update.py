"""
Standalone daily update script — use this if you don't want to run Airflow.

Pulls yesterday's games, appends to full_ingestion.json and game_summaries.json,
upserts new vectors into Qdrant, then rebuilds features + retrains the model.

Usage:
  python scripts/daily_update.py              # yesterday
  python scripts/daily_update.py --date 2026-06-08   # specific date
  python scripts/daily_update.py --skip-train        # skip model retrain
"""

import argparse
import json
import os
import sys
import uuid
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(".env.dev")

from src.ingestion.mlb_feed_ingestion import run_feed_ingestion


def append_json(path, new_items, key="game_pk"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []
    existing_keys = {item.get(key) for item in existing}
    added = [item for item in new_items if item.get(key) not in existing_keys]
    if added:
        existing.extend(added)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f)
    return len(added)


def upsert_to_qdrant(plays, date_str):
    from llama_index.embeddings.openai import OpenAIEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    if not plays:
        print("  No plays to embed")
        return

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    qdrant = QdrantClient(
        os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    batch_size = 50
    total = 0
    for i in range(0, len(plays), batch_size):
        batch = plays[i:i + batch_size]
        vectors = embed_model.get_text_embedding_batch([p["text"] for p in batch])
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={k: v for k, v in play.items()},
            )
            for play, vec in zip(batch, vectors)
        ]
        qdrant.upsert(collection_name="mlb_articles", points=points)
        total += len(points)
        print(f"  Upserted {total}/{len(plays)} vectors...")

    print(f"  Done — {total} vectors added to Qdrant")


def rebuild_model():
    import subprocess
    steps = [
        ["python", "-m", "src.features.compute_player_stats"],
        ["python", "-m", "src.features.build_features", "--seasons", "2025", "2026"],
        ["python", "-m", "src.features.extract_f5_scores"],
        ["python", "-m", "src.features.build_f5_features"],
        ["python", "-m", "src.models.train"],
        ["python", "-m", "src.models.train_f5"],
    ]
    for cmd in steps:
        print(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ERROR: {result.stderr[-500:]}")
        else:
            print(f"  OK")


def run(target_date: str, skip_train: bool = False):
    print(f"\n{'='*50}")
    print(f"  MLB Daily Update — {target_date}")
    print(f"{'='*50}\n")

    # Step 1 — Ingest
    print(f"[1/3] Ingesting games for {target_date}...")
    new_plays, new_summaries = run_feed_ingestion(
        start_date=target_date,
        end_date=target_date,
    )
    print(f"  Fetched {len(new_plays)} plays, {len(new_summaries)} summaries")

    added_plays = append_json("data/full_ingestion.json", new_plays)
    added_summaries = append_json("data/game_summaries.json", new_summaries)
    print(f"  New plays appended: {added_plays}")
    print(f"  New summaries appended: {added_summaries}")

    # Step 2 — Embed + upsert
    print(f"\n[2/3] Upserting {added_plays} new plays into Qdrant...")
    day_plays = [p for p in new_plays if p.get("date") == target_date]
    upsert_to_qdrant(day_plays, target_date)

    # Step 3 — Rebuild model
    if skip_train:
        print("\n[3/3] Skipping model retrain (--skip-train)")
    else:
        print("\n[3/3] Rebuilding features and retraining model...")
        rebuild_model()

    print(f"\nDone! {added_plays} new plays, {added_summaries} new summaries ingested.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",       default=None, help="Date to ingest (YYYY-MM-DD), default: yesterday")
    parser.add_argument("--skip-train", action="store_true", help="Skip feature rebuild and model retrain")
    args = parser.parse_args()

    target = args.date or (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    run(target, skip_train=args.skip_train)
