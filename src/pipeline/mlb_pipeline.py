from ingestion.mlb_api import mlb_api
from processing.create_documents import create_documents
from embeddings.embed_and_store import embed_and_store


def ingest_mlb():
    mlb_api()


def process_documents():
    create_documents()


def store_embeddings():
    embed_and_store()