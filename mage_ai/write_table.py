import os,sys
p = os.path.abspath(os.path.join(os.path.dirname( '__file__' ),'src'))
sys.path.append(p)
import utils

def main():
    CONDATA = utils.config_load()

    df = utils.load_csv(CONDATA['raw_data_path'])
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    utils.create_table_df(data=df.astype(str), engine_path=CONDATA['postgre_path'], table_name='vmnc_raw')

main()