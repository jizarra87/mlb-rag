from fastapi import FastAPI
from src.rag.query_engine import generate_answer
from pydantic import BaseModel

app = FastAPI()


class Question(BaseModel):
    question: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask")
def ask(q: Question):
    answer = generate_answer(q.question)
    return {
        "question": q.question,
        "answer": answer
    }