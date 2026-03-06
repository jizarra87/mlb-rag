import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv(".env.dev")

embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

client = QdrantClient("localhost", port=6333)


def search(question):

    vector = embed_model.get_text_embedding(question)

    results = client.query_points(
        collection_name="mlb_articles",
        query=vector,
        limit=3
    )

    for r in results.points:
        print(r.payload["text"])

if __name__ == "__main__":

    q = input("Ask a question: ")

    search(q)