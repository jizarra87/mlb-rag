from ingestion.mlb_historical import ingest_historical
from processing.create_documents_history import create_documents
from embeddings.embed_and_store import embed_and_store


def ingest_mlb():
    ingest_historical()


def process_documents():
    create_documents()


def store_embeddings():
    embed_and_store()



if __name__ == "__main__":
    print("Starting MLB pipeline...")

    ingest_mlb()
    process_documents()
    store_embeddings()

    print("Pipeline completed.")
