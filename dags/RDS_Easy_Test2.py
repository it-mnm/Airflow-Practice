import csv
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.mysql_hook import MySqlHook
from sklearn.model_selection import train_test_split


default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}

def mysql_hook():
    # hook 내용 작성
    logging.info("데이터베이스에서 데이터 가져오기")
    try:
        hook = MySqlHook.get_hook(conn_id="mysql-01") # 미리 정의한 mysql connection 적용
        connection = hook.get_conn() # connection 하기
        cursor = connection.cursor() # cursor객체 만들기
        cursor.execute("use vod_rec") # sql문 수행
        cont_log = pd.read_sql('select * from contlog', connection)
        vod_log = pd.read_sql('select * from vods_sumut', connection)
        vod_info = pd.read_sql('select * from vodinfo', connection)

        vod_log = vod_log[vod_log['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch']]
        cont_log = cont_log[cont_log['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt']]
        vod_info = vod_info[vod_info['e_bool']==0][['program_id','program_name', 'ct_cl', 'program_genre', 'release_date', 'age_limit']]

        # use_tms / running time
        vod_log['use_tms_ratio'] = vod_log['use_tms'] / vod_log['disp_rtm_sec']
        use_tms_ratio = vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].max().reset_index()

        # vod, cont 합치기
        rating = pd.concat([vod_log[['subsr_id', 'program_id']], cont_log[['subsr_id', 'program_id']]]).drop_duplicates().reset_index(drop=True)
        rating = rating.merge(use_tms_ratio, how='left')

        # train/test 분리
        train, test = train_test_split(rating.dropna(), test_size=0.25, random_state=0)
        train = rating.copy()
        train.loc[test.index, 'use_tms_ratio'] = np.nan


    except Exception as err:
        print(f"오류: {err}")

    finally:
        connection.close()



        
with DAG(
    dag_id="ex_mysql_to_csv",
    default_args=default_args,
    start_date=datetime(2022, 4, 30),
    schedule_interval='@once'
) as dag:
    task1 = PythonOperator(
        task_id="MySQL_query",
        python_callable=mysql_hook
    )

    task1