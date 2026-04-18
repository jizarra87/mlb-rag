import json
import logging
import os
import uuid

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

load_dotenv(".env.dev")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DOCUMENTS_PATH = "data/processed/documents.json"
COLLECTION_NAME = "mlb_articles"


def embed_and_store():
    client = QdrantClient(
        os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", "6333")),
    )

    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    with open(DOCUMENTS_PATH) as f:
        documents = json.load(f)

    points = []

    for index, doc in enumerate(documents, start=1):
        text = doc["text"]
        metadata = doc["metadata"]
        logger.info("Embedding document %s", index)

        vector = embed_model.get_text_embedding(text)

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "text": text,
                    **metadata,
                },
            )
        )

    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("Vectors stored:", len(points))
