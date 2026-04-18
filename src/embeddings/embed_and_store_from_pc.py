import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os
from qdrant_client.models import VectorParams, Distance
import time

load_dotenv(".env.dev")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")

def embed_and_store():
    with open("data/full_ingestion.json", "r") as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents from file")
    start_time = time.time()
    errors = 0
    # conectar a Qdrant
    client = QdrantClient(QDRANT_HOST, port=6333)

    # modelo de embeddings
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # cargar documentos procesados
    #with open("data/clean/documents.json") as f:
    #    documents = json.load(f)


    BATCH_SIZE = 50
    points = []

    for i in range(0, len(documents), BATCH_SIZE):
        batch_docs = documents[i:i+BATCH_SIZE]

        texts = [doc["text"] for doc in batch_docs]

        try:
            vectors = embed_model.get_text_embedding_batch(texts)
            time.sleep(0.2)

            for doc, vector in zip(batch_docs, vectors):
                metadata = {k: v for k, v in doc.items() if k != "text"}

                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": doc["text"],
                            **metadata
                        }
                    )
                )

        except Exception as e:
            print(f"Error batch {i}: {e}")

        if i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0

            print(f"""
            Processed {i}/{len(documents)} docs
            Errors: {errors}
            Elapsed: {round(elapsed, 2)} sec
            Speed: {round(rate, 2)} docs/sec
            """)
    collection_name = "mlb_articles"

    # 🔥 limpiar colección antes de cargar
    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        print("Collection does not exist, creating new one")

    # Crear colección si no existe
    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    # insertar vectores en Qdrant
    print(f"Total points to insert: {len(points)}")
    BATCH_SIZE = 100

    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i:i+BATCH_SIZE]

        client.upsert(
            collection_name=collection_name,
            points=batch
        )

        print(f"Inserted batch {i} - {i+len(batch)}")
    print("Upsert completed successfully")


    print("Vectors stored:", len(points))



if __name__ == "__main__":
    print("Starting MLB pipeline...")

    #docs = ingest_mlb()
    #process_documents()
    embed_and_store()

    print("Pipeline completed.")