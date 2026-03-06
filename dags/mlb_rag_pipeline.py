from datetime import datetime, timedelta
import subprocess

from airflow import DAG
from airflow.operators.python import PythonOperator


def run_mlb_api():
    subprocess.run(["python", "/opt/airflow/src/ingestion/mlb_api.py"], check=True)


def run_create_documents():
    subprocess.run(["python", "/opt/airflow/src/processing/create_documents.py"], check=True)


def run_embed_and_store():
    subprocess.run(["python", "/opt/airflow/src/embeddings/embed_and_store.py"], check=True)

    
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

    ingest_mlb_data = PythonOperator(
        task_id="ingest_mlb_data",
        python_callable=run_mlb_api,
    )

    create_documents = PythonOperator(
        task_id="create_documents",
        python_callable=run_create_documents,
    )

    embed_and_store = PythonOperator(
        task_id="embed_and_store",
        python_callable=run_embed_and_store,
    )

    ingest_mlb_data >> create_documents >> embed_and_store