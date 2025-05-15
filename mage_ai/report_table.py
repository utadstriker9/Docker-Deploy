import os,sys
p = os.path.abspath(os.path.join(os.path.dirname( '__file__' ),'src'))
sys.path.append(p)
import utils

def data_content(engine):
    q = f"""
    SELECT 
        content_type,
        COUNT(DISTINCT iduser) AS total
    FROM vmnc_clean
    GROUP BY 1 
    ORDER BY 1 
    """
    df = utils.get_data_df(sql=q,engine_path=engine)
    return df

def data_province(engine):
    q = f"""
    SELECT 
        province,
        COUNT(DISTINCT iduser) AS total
    FROM vmnc_clean
    GROUP BY 1 
    ORDER BY 1 
    """
    df = utils.get_data_df(sql=q,engine_path=engine)
    return df

def main():
    CONDATA = utils.config_load()

    df_province = data_province(engine=CONDATA['postgre_path'])
    utils.create_table_df(data=df_province.astype(str), engine_path=CONDATA['postgre_path'], table_name='vmnc_report_province')

    df_content = data_content(engine=CONDATA['postgre_path'])
    utils.create_table_df(data=df_content.astype(str), engine_path=CONDATA['postgre_path'], table_name='vmnc_report_content')
main()