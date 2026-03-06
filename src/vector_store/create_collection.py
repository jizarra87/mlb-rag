from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="mlb_articles",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    ),
)

print("Collection created")