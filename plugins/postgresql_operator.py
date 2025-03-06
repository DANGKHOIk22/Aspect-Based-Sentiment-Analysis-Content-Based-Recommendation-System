from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
from sqlalchemy import create_engine

class PostgresOperators:
    def __init__(self, conn_id):
        self.conn_id = conn_id
        self.hook = PostgresHook(postgres_conn_id=self.conn_id)

    def get_connection(self):
        return self.hook.get_conn()

    def get_data_to_pd(self, sql):
        return self.hook.get_pandas_df(sql)

    def save_data_to_postgres(self, df, table_name, schema='public', if_exists='replace'):
        conn = self.hook.get_uri()
        engine = create_engine(conn)
        # Ensure the schema exists before writing
        self.create_schema(schema)
        df.to_sql(table_name, engine, schema=schema, if_exists=if_exists, index=False)

    def execute_query(self, sql):
        self.hook.run(sql)

    def create_schema(self, schema_name):
        """Create a schema if it doesn't exist."""
        create_schema_sql = f"CREATE SCHEMA IF NOT EXISTS {schema_name};"
        self.execute_query(create_schema_sql)