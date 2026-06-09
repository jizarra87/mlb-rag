"""
Daily MLB ingestion DAG.

Runs every day at 6 AM ET to pull the previous day's games:
  1. Ingest plays + game summaries for yesterday
  2. Append new plays to full_ingestion.json
  3. Upsert new vectors into Qdrant (incremental — no full re-index)
  4. Rebuild features CSV + retrain model if enough new games

The DAG uses a simple date macro so it always processes the day before
execution, regardless of when it actually runs.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

DEFAULT_ARGS = {
    "owner": "mlb-rag",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def ingest_yesterday(**context):
    import json
    import os
    from src.ingestion.mlb_feed_ingestion import run_feed_ingestion

    yesterday = (context["execution_date"] - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"Ingesting games for {yesterday}...")

    new_plays, new_summaries = run_feed_ingestion(
        start_date=yesterday,
        end_date=yesterday,
    )
    print(f"Fetched {len(new_plays)} plays and {len(new_summaries)} summaries")

    # Append plays to full_ingestion.json
    plays_path = "data/full_ingestion.json"
    if os.path.exists(plays_path):
        with open(plays_path, encoding="utf-8") as f:
            existing = json.load(f)
    else:
        existing = []

    existing_pks = {p.get("game_pk") for p in existing}
    added = [p for p in new_plays if p.get("game_pk") not in existing_pks]
    if added:
        existing.extend(added)
        with open(plays_path, "w", encoding="utf-8") as f:
            json.dump(existing, f)
        print(f"Appended {len(added)} new plays")
    else:
        print("No new plays to append (already up to date)")

    # Append summaries to game_summaries.json
    summaries_path = "data/game_summaries.json"
    if os.path.exists(summaries_path):
        with open(summaries_path, encoding="utf-8") as f:
            existing_summaries = json.load(f)
    else:
        existing_summaries = []

    existing_summary_pks = {s.get("game_pk") for s in existing_summaries}
    new_s = [s for s in new_summaries if s.get("game_pk") not in existing_summary_pks]
    if new_s:
        existing_summaries.extend(new_s)
        with open(summaries_path, "w", encoding="utf-8") as f:
            json.dump(existing_summaries, f)
        print(f"Appended {len(new_s)} new summaries")

    # Push counts for downstream tasks
    context["ti"].xcom_push(key="new_plays",     value=len(added))
    context["ti"].xcom_push(key="new_summaries", value=len(new_s))

    # Invalidate bot's in-memory plays cache
    try:
        from src.rag import query_engine
        query_engine._plays_cache = None
    except Exception:
        pass


def upsert_embeddings(**context):
    """Embed and upsert only the new plays into Qdrant — no full re-index."""
    import json
    import uuid
    import os
    from dotenv import load_dotenv
    from llama_index.embeddings.openai import OpenAIEmbedding
    from qdrant_client import QdrantClient
    from qdrant_client.models import PointStruct

    new_play_count = context["ti"].xcom_pull(key="new_plays", task_ids="ingest_yesterday")
    if not new_play_count:
        print("No new plays — skipping embedding")
        return

    load_dotenv(".env.dev")
    plays_path = "data/full_ingestion.json"
    with open(plays_path, encoding="utf-8") as f:
        all_plays = json.load(f)

    # Only embed plays from yesterday
    yesterday = (context["execution_date"] - timedelta(days=1)).strftime("%Y-%m-%d")
    new_plays = [p for p in all_plays if p.get("date") == yesterday]
    print(f"Embedding {len(new_plays)} plays for {yesterday}...")

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    qdrant = QdrantClient(
        os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    batch_size = 50
    total_upserted = 0
    for i in range(0, len(new_plays), batch_size):
        batch = new_plays[i:i + batch_size]
        texts = [p["text"] for p in batch]
        vectors = embed_model.get_text_embedding_batch(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={k: v for k, v in play.items()},
            )
            for play, vec in zip(batch, vectors)
        ]
        qdrant.upsert(collection_name="mlb_articles", points=points)
        total_upserted += len(points)

    print(f"Upserted {total_upserted} vectors into Qdrant")


def rebuild_features(**context):
    """Rebuild features CSV and retrain model when enough new games accumulated."""
    new_summaries = context["ti"].xcom_pull(key="new_summaries", task_ids="ingest_yesterday")
    if not new_summaries:
        print("No new summaries — skipping feature rebuild")
        return

    import subprocess
    print("Rebuilding features...")
    subprocess.run(["python", "-m", "src.features.compute_player_stats"], check=True)
    subprocess.run(["python", "-m", "src.features.build_features", "--seasons", "2025", "2026"], check=True)
    subprocess.run(["python", "-m", "src.features.extract_f5_scores"], check=True)
    subprocess.run(["python", "-m", "src.features.build_f5_features"], check=True)
    subprocess.run(["python", "-m", "src.models.train"], check=True)
    subprocess.run(["python", "-m", "src.models.train_f5"], check=True)
    print("Model retrained with latest data")


with DAG(
    dag_id="mlb_daily_ingestion",
    default_args=DEFAULT_ARGS,
    description="Ingest yesterday's MLB games, update Qdrant, rebuild model",
    schedule_interval="0 10 * * *",  # 10 AM UTC = 6 AM ET
    start_date=datetime(2026, 6, 1),
    catchup=False,
    tags=["mlb", "ingestion"],
) as dag:

    t1 = PythonOperator(
        task_id="ingest_yesterday",
        python_callable=ingest_yesterday,
        provide_context=True,
    )

    t2 = PythonOperator(
        task_id="upsert_embeddings",
        python_callable=upsert_embeddings,
        provide_context=True,
    )

    t3 = PythonOperator(
        task_id="rebuild_features",
        python_callable=rebuild_features,
        provide_context=True,
    )

    t1 >> t2 >> t3
