from ingestion.mlb_api import mlb_api
from ingestion.mlb_feed_ingestion import run_feed_ingestion
from processing.create_documents_history import create_documents
from embeddings.embed_and_store import embed_and_store



def ingest_mlb():
    return run_feed_ingestion()
    #mlb_api()


def process_documents():
    create_documents()


def store_embeddings(documents):
    embed_and_store(documents)



if __name__ == "__main__":
    print("Starting MLB pipeline...")

    docs = ingest_mlb()
    #process_documents()
    store_embeddings(docs)

    print("Pipeline completed.")