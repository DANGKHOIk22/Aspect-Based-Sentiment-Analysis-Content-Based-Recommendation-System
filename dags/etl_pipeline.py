from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import os,sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from plugins.scraping_data import fetch_data  

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=2),
}

def fetch_and_process_data(**kwargs):
    genres = ['art','science','history']  
    execution_date=kwargs['execution_date']
    fine_tuned= False
    all_data = [] 

    for genre in genres:
        if fine_tuned == False:
            data = fetch_data(genre=genre)
        else :
            data = fetch_data(execution_date=execution_date,genre=genre)
        for d in data:
            all_data.append(d)
    
    # Push processed data to XCom
    kwargs['ti'].xcom_push(key='scraped_data', value=all_data)

def insert_data(**kwargs):
    ti = kwargs['ti']
    data = ti.xcom_pull(key='scraped_data', task_ids='fetch_data_tasks')
    
    if not data:
        raise ValueError("No data received from previous task")
    
    # Use connection pooling with context manager
    pg_hook = PostgresHook(postgres_conn_id='FotMob_conn')
    with pg_hook.get_conn() as conn:
        with conn.cursor() as cursor:
            # Use parameterized query to prevent SQL injection
            query = "INSERT INTO sentiment_data (sentiment, comment) VALUES (%s, %s)"
            # Assuming data is a dict of {sentiment: comment}
            
            cursor.executemany(query, data)
        
        conn.commit()
    return "Data inserted successfully"

with DAG(
    dag_id='etl_data',
    default_args=default_args,
    start_date=datetime(2025, 2, 22),
    schedule_interval='@daily',
    catchup=False
) as dag:
    fetch_data_task = PythonOperator(
        task_id='fetch_data_tasks',
        python_callable=fetch_and_process_data,
        provide_context=True  
    )

    # Task to create tables in the database
    create_db_table_task = PostgresOperator(
        task_id='create_db_tables',
        postgres_conn_id='FotMob_conn',
        sql='sql/create_tables.sql'
    )
    # Switching to PythonOperator for the insert task
    insert_data_task = PythonOperator(
        task_id='insert_data_task',
        python_callable=insert_data,
        provide_context=True  
    )

    fetch_data_task >>create_db_table_task>> insert_data_task
    #fetch_data_task >> insert_data_task