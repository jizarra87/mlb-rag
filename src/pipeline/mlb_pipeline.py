from src.ingestion.scrape_espn import scrape_games
from src.processing.create_documents import create_documents
from src.embeddings.embed_and_store import embed_and_store


def ingest_mlb():
    scrape_games()


def process_documents():
    create_documents()


def store_embeddings():
    embed_and_store()