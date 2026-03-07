from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding

client = QdrantClient("localhost", port=6333)

embed_model = OpenAIEmbedding()

def search(query):

    query_embedding = embed_model.get_text_embedding(query)

    results = client.search(
        collection_name="mlb_articles",
        query_vector=query_embedding,
        limit=3
    )

    for r in results:
        print(r.payload["text"])
        print("score:", r.score)
        print()

query = input("Ask question: ")
search(query)