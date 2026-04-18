import os

from dotenv import load_dotenv
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from qdrant_client import QdrantClient

load_dotenv(".env.dev")


embed_model = OpenAIEmbedding(
    model="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

llm = OpenAI(
    model="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY"),
)

client = QdrantClient(
    os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", "6333")),
)


def retrieve(question):
    vector = embed_model.get_text_embedding(question)

    return client.search(
        collection_name="mlb_articles",
        query_vector=vector,
        limit=3,
        score_threshold=float(os.getenv("RAG_SCORE_THRESHOLD", "0.35")),
    )


def generate_answer(question):
    docs = retrieve(question)
    if not docs:
        return (
            "I couldn't find relevant MLB context for that question. "
            "Try adding a team, player, season, or date."
        )

    context_blocks = []

    for index, doc in enumerate(docs, start=1):
        payload = doc.payload or {}
        game_date = payload.get("date", "unknown date")
        away_team = payload.get("away_team", "Unknown away team")
        home_team = payload.get("home_team", "Unknown home team")
        snippet = payload.get("text", "")

        context_blocks.append(
            f"[Source {index}] Date: {game_date} | Matchup: {away_team} at {home_team}\n{snippet}"
        )

    context = "\n\n".join(context_blocks)

    prompt = f"""
You are an MLB analyst.

Use only the context below to answer the question.
If the context is not enough, say you do not know based on the indexed data.
Do not invent stats, dates, injuries, or outcomes.
End with a short Sources line that references the source numbers you used.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.complete(prompt)

    return response.text


if __name__ == "__main__":
    question = input("Ask a question: ")

    answer = generate_answer(question)

    print("\nAnswer:")
    print(answer)
