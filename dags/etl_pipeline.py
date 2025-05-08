from datetime import datetime, timedelta
import pandas as pd
import re
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
import os,sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)
from plugins.scraping_data import fetch_data  
from plugins.preprocessing import preprocess
from plugins.postgresql_operator import PostgresOperators


default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=2)
}

# Extract task
def extract_data(**kwargs):
    staging_operator = PostgresOperators('postgres')
    genres = ['art', 'science', 'history','biography','business', 
              'christian', 'classics', 'comics', 'cookbooks', 'fantasy', 'fiction', 'graphic novels']
    all_data = []

    for idx, genre in enumerate(genres):
        data = fetch_data(idx,genre)
        all_data.extend(data)  
    
    df = pd.DataFrame(all_data, columns=['book_id','book_title','genre', 'pages','rating', 'ratings_count', 'reviews_count', 'comment_text'])
    
    staging_operator.save_data_to_postgres(
        df,
        "stg_data_book",
        schema='staging'
    )

def transform_data():
    staging_operator = PostgresOperators('postgres')
    
    # Read data from staging
    df = staging_operator.get_data_to_pd("SELECT * FROM staging.stg_data_book")

    # Process data
    df = preprocess(df)

    return df  # Return the transformed DataFrame
def load_data():
    df = transform_data()  # Get the transformed DataFrame
    warehouse_operator = PostgresOperators('postgres')
    ## Specify the path
    directory = "data"
    path = os.path.join(directory, "data.csv")

    # Create the directory if it doesn't exist
    os.chdir(PROJECT_ROOT)  # Use the project root defined earlier
    directory = "data"
    path = os.path.join(directory, "data.csv")
    os.makedirs(directory, exist_ok=True)
    df.to_csv(path, index=False, encoding='utf-8-sig', quoting=1)

    # Save to warehouse
    warehouse_operator.save_data_to_postgres(
        df,
        "book_data",
        schema='warehouse',
        if_exists = 'replace'
    )

# Define the DAG
with DAG(
    dag_id='etl_data',
    default_args=default_args,
    start_date=datetime(2025, 2, 22),
    schedule_interval='@daily',
    catchup=False
) as dag:
    # Task Extract
    with TaskGroup("extract") as extract_group:
        extract_task = PythonOperator(
            task_id='extract_data_task',
            python_callable=extract_data,
            provide_context=True,
        )

    # Task Group Transform
    with TaskGroup("transform") as transform_group:
        transform_task = PythonOperator(
            task_id='transform_data_task',
            python_callable=transform_data,
        )
        
        
    # Task Group Load
    with TaskGroup("load") as load_group:
        load_task = PythonOperator(
            task_id='load_data_task',
            python_callable=load_data,
        )

    # Set dependencies
    extract_group >> transform_group >> load_group

