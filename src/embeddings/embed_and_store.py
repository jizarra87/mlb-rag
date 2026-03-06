import json
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
from llama_index.embeddings.openai import OpenAIEmbedding
from dotenv import load_dotenv
import os

load_dotenv(".env.dev")

# conectar a Qdrant
client = QdrantClient("qdrant", port=6333)

# modelo de embeddings
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# cargar documentos procesados
with open("data/clean/documents.json") as f:
    documents = json.load(f)


points = []

for doc in documents:

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