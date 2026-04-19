MLB RAG Analytics Engine

A Retrieval-Augmented Generation (RAG) project for turning MLB play-by-play data into queryable baseball insights with OpenAI models and Qdrant.

Overview

This repository currently ingests MLB Stats API data, extracts play-level events, embeds them, stores them in Qdrant, and serves answers through a FastAPI app.

Instead of treating AI as a standalone tool, the project combines ingestion, embedding, retrieval, and LLM generation into a single baseball-focused workflow.

Key Features
- MLB Stats API schedule and live-feed ingestion
- Play-level event extraction for batters, pitchers, teams, and games
- OpenAI embeddings stored in Qdrant
- FastAPI endpoint for question answering
- Docker-based local environment
- Modular structure for ingestion, embedding, and retrieval logic

Architecture
- Data Ingestion: MLB Stats API schedule plus live game feeds
- Document Layer: Play-by-play event extraction
- Embedding Layer: OpenAI embeddings stored in Qdrant
- Retrieval Layer: Vector search plus custom player and matchup lookups
- LLM Layer: Context-aware answer generation
- Infrastructure: Docker-based local services

Tech Stack
- Python
- Docker
- Qdrant
- FastAPI
- OpenAI

Current Repository Notes
- The current pipeline entrypoint is `src/pipeline/mlb_pipeline.py`.
- The repository includes an Airflow service in `docker-compose.yml`, but the current `main` branch does not include a project DAG under `dags/`.
- The indexed corpus is built from MLB play-level events rather than season-level summary documents.

Next Steps
- Make ingestion windows configurable instead of hardcoded
- Improve ranking and retrieval strategies
- Add evaluation metrics for RAG quality
- Add stronger operational safeguards around index refreshes
