MLB RAG Analytics Engine

A production-style Retrieval-Augmented Generation (RAG) system designed to transform raw MLB data into contextual, queryable insights using LLMs.

Overview

This project simulates a real-world analytics product where users can ask natural language questions about MLB games and receive grounded, data-backed answers.

Instead of treating AI as a standalone tool, this system integrates data engineering, modeling, and LLM capabilities into a scalable pipeline.

Key Features
End-to-end data pipeline (ingestion → processing → embeddings)
Vector search using a dedicated vector database
Natural language querying over structured + unstructured data
Grounded responses (no hallucinations)
Modular and scalable architecture
Orchestrated with Airflow and containerized with Docker
CI/CD-ready project structure
Architecture
Data Ingestion: MLB API → raw JSON data
Processing Layer: Document creation and chunking
Embedding Layer: OpenAI embeddings stored in vector DB
Retrieval Layer: Similarity search (top-K relevant documents)
LLM Layer: Context-aware response generation
Orchestration: Airflow DAGs
Infrastructure: Docker-based environment
Tech Stack
Python
SQL
Airflow
Docker
Vector DB (Qdrant / Pinecone)
LLMs (OpenAI)
GitHub (CI/CD ready)
📈 Why This Matters

This project reflects how modern data systems evolve beyond dashboards into intelligent, queryable platforms. It combines data engineering fundamentals with AI to enable faster, more intuitive decision-making.

What I Focused On
Designing a system, not just a prototype
Clean, modular pipeline structure
Scalability and production readiness
Bridging data engineering with AI-driven analytics
🔗 Next Steps
Add real-time ingestion
Improve ranking and retrieval strategies
Introduce evaluation metrics for RAG quality
Expand to multi-domain datasets
