import os,sys
p = os.path.abspath(os.path.join(os.path.dirname( '__file__' ),'src'))
sys.path.append(p)
import utils
import pandas as pd
from sqlalchemy import Integer, String, Float, DateTime

dtype = {
    'iduser': Integer,
    'start_watching': DateTime,
    'province': String,
    'city': String,
    'content_name': String,
    'playing_time_millisecond': Integer,
    'device_type': String,
    'content_type': String
}

def data_raw(engine):
    q = f"""
    SELECT 
        *
    FROM vmnc_raw
    """
    df = utils.get_data_df(sql=q,engine_path=engine)
    return df

def main():
    CONDATA = utils.config_load()
    table_name = 'vmnc_clean'
    df_raw = data_raw(engine=CONDATA['postgre_path'])

    df_clean = utils.transform(df_raw)
    
    df_clean['start_watching'] = pd.to_datetime(df_clean['start_watching'], format='%m/%d/%Y %H:%M')

    utils.create_table_df(data=df_clean.astype(str), engine_path=CONDATA['postgre_path'], table_name=table_name, dtype=dtype)
    print(f"SUCCESSFULLY TRANSFORM TABLE {table_name}")

main()