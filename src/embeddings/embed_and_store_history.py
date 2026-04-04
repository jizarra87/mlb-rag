import json
import uuid
import logging
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os

load_dotenv(".env.dev")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def embed_and_store():
    # conectar a Qdrant
    client = QdrantClient("qdrant", port=6333)

    # modelo de embeddings
    embed_model = OpenAIEmbedding(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )

    # cargar documentos procesados
    with open("data/processed/documents.json") as f:
        documents = json.load(f)


    points = []
    ct = 0
    for doc in documents:
        ct += 1
        logger.info(f"Embedding document {ct}")
        text = doc["text"]
        metadata = doc["metadata"]

        vector = embed_model.get_text_embedding(text)

        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "text": text,
                **metadata
            }
        )

        points.append(point)


    # insertar vectores en Qdrant
    client.upsert(
        collection_name="mlb_articles",
        points=points
    )


    print("Vectors stored:", len(points))


if __name__ == "__main__":
    logger.info("Starting embeddings process")
    embed_and_store()