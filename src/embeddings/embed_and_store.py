import logging
import os
import time
import uuid

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(".env.dev")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))


def embed_and_store(documents):
    start_time = time.time()
    errors = 0
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    batch_size = 50
    points = []

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
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
                            **metadata,
                        },
                    )
                )
        except Exception:
            errors += 1
            logger.exception("Error embedding batch starting at doc %s", i)

        if i % 500 == 0:
            elapsed = time.time() - start_time
            rate = i / elapsed if elapsed > 0 else 0

            print(
                f"""
            Processed {i}/{len(documents)} docs
            Errors: {errors}
            Elapsed: {round(elapsed, 2)} sec
            Speed: {round(rate, 2)} docs/sec
            """
            )

    if errors:
        raise RuntimeError(
            f"Embedding aborted after {errors} failed batch(es); "
            "existing Qdrant collection was left unchanged."
        )

    collection_name = "mlb_articles"

    try:
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    except Exception:
        print("Collection does not exist, creating new one")

    if collection_name not in [c.name for c in client.get_collections().collections]:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    print(f"Total points to insert: {len(points)}")
    upsert_batch_size = 100

    for i in range(0, len(points), upsert_batch_size):
        batch = points[i:i + upsert_batch_size]

        client.upsert(
            collection_name=collection_name,
            points=batch,
        )

        print(f"Inserted batch {i} - {i + len(batch)}")

    print("Upsert completed successfully")
    print("Vectors stored:", len(points))
