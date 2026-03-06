from datetime import datetime, timedelta
from src.pipeline.mlb_pipeline import ingest_mlb, process_documents, store_embeddings
from airflow import DAG
from airflow.operators.python import PythonOperator
import sys
import os

sys.path.append("/opt/airflow")



default_args = {
    "owner": "juan",
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    dag_id="mlb_rag_pipeline",
    default_args=default_args,
    description="MLB RAG ingestion pipeline",
    start_date=datetime(2026, 3, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlb", "rag"],
) as dag:

    ingest_task = PythonOperator(
        task_id="ingest_mlb_data",
        python_callable=ingest_mlb
    )

    process_task = PythonOperator(
        task_id="create_documents",
        python_callable=process_documents
    )

    embed_task = PythonOperator(
        task_id="embed_and_store",
        python_callable=store_embeddings
    )

    ingest_task >> process_task >> embed_task