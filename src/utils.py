import yaml
import pandas as pd
import joblib
from datetime import datetime
import os
from sqlalchemy import create_engine, text
import psycopg2
from jinjasql import JinjaSql

JINJA = JinjaSql(param_style='pyformat')

CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

# Function to load config files
def config_load():
    try:
        with open(CONFIG_DIR, 'r') as file:
            config = yaml.safe_load(file)
    except FileNotFoundError as error:
        raise RuntimeError('Parameter file not found in path')
    return config

# Function to load csv files
def load_csv(file_path):
    data = open(file_path)
    return pd.read_csv(data)
    
# Function that return the current time
def time_stamp():
    return datetime.now()

# Write data to PostgreSQL
def create_table_df(data, engine_path, table_name, dtype=None):
    engine = create_engine(engine_path)

    with engine.begin() as connection:  
            # Write table 
            if dtype is not None:
                data.to_sql(table_name, connection, if_exists='append', index=False, dtype=dtype)
            else:
                data.to_sql(table_name, connection, if_exists='append', index=False)
    return print(f'SUCCESFULLY WRITE TABLE {table_name}')

# Query PostgreSQL to DF 
def get_data_df(sql,engine_path,data={}):
    engine = create_engine(engine_path)
    query_result, bin_params = JINJA.prepare_query(sql,data)
    df = pd.read_sql(query_result,engine)
    return df 

# Transform DF
def transform(df, *args, **kwargs):
    missing_values = df.isnull().sum()
    if missing_values.any():
        print(f"Warning: Missing values found in columns: {print(missing_values[missing_values > 0])}")
    
    # Check for missing value
    required_columns = ['iduser']
    for cols in required_columns:
        if df[cols].isnull().any():
            raise ValueError(f"Missing values detected in critical column: {cols}")

    # Check for duplicates
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Removing duplicates...")
        df = df.drop_duplicates()
        
    df = df.drop(columns=['temp_column'], errors='ignore')

    return df
    