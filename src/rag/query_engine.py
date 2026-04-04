import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


load_dotenv(".env.dev")


# Embedding model
embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY")
)

# LLM model
llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

# Qdrant client
client = QdrantClient("qdrant", port=6333)


# -----------------------------
# Retrieval Layer
# -----------------------------
def retrieve(question):

    vector = embed_model.get_text_embedding(question)

    results = client.search(
        collection_name="mlb_articles",
        query_vector=vector,
        limit=3
    )
    return results


# -----------------------------
# Generation Layer
# -----------------------------
def generate_answer(question):

    docs = retrieve(question)

    context = ""
    print(docs[0].payload)
    for d in docs:
        context += d.payload.get("text", "") + "\n"

    prompt = f"""
You are an MLB analyst.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer clearly using the context.
"""

    response = llm.complete(prompt)

    return response.text


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    question = input("Ask a question: ")

    answer = generate_answer(question)

    print("\nAnswer:")
    print(answer)