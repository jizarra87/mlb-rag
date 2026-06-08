import argparse
from src.ingestion.mlb_feed_ingestion import run_feed_ingestion
from src.embeddings.embed_and_store import embed_and_store


def ingest_mlb(start_date=None, end_date=None):
    return run_feed_ingestion(start_date=start_date, end_date=end_date)


def store_embeddings(documents):
    embed_and_store(documents)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MLB RAG ingestion pipeline")
    parser.add_argument("--start-date", default=None, help="Start date YYYY-MM-DD (default: 2026-03-20)")
    parser.add_argument("--end-date", default=None, help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    print("Starting MLB pipeline...")
    docs = ingest_mlb(start_date=args.start_date, end_date=args.end_date)
    store_embeddings(docs)
    print("Pipeline completed.")