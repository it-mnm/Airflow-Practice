
import csv
import json
import pandas as pd
import numpy as np
import logging
import pendulum
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile 
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.mysql_hook import MySqlHook
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from surprise import SVD, Reader, Dataset, accuracy
from lightfm import LightFM, cross_validation
from lightfm.data import Dataset
from sklearn.metrics.pairwise import cosine_similarity




# 기본 설정
default_args = {
    'owner': 'admin',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}

local_tz = pendulum.timezone("Asia/Seoul")


#주차 정보 불러오기
def week_info_s3(**context):
    s3_hook = S3Hook(aws_conn_id='aws_default')

    # S3에 파일이 존재하는지 확인
    file_exists = s3_hook.check_for_key('week_info.json', 'airflowexample')


    #파일이 존재하지 않을 경우 새 파일 생성, 존재할 경우에는 읽어오기
    if not file_exists:
        # If the file doesn't exist, create a new structure
        logging.info("*****파일 존재하지 않으므로 새로생성*****")

        new_metrics = {"columns": ["week_info"],"values": [31]}
        updated_json_data = json.dumps(new_metrics, indent=2)
        current_week=new_metrics["values"][0]
        s3_hook.load_string(updated_json_data, 'week_info.json', 'airflowexample', replace=True)

    else:
        logging.info("*****파일 존재. 기존 파일 읽어옴*****")

        existing_data = s3_hook.read_key('week_info.json', 'airflowexample')
        existing_metrics = json.loads(existing_data)
        #이전 주차 정보 읽기
        current_week=existing_metrics["values"][0]

         # 기존 주차 정보 업데이트
        existing_metrics["values"] = [current_week + 1]
        updated_json_data = json.dumps(existing_metrics, indent=2)
        s3_hook.load_string(updated_json_data, 'week_info.json', 'airflowexample', replace=True)



        
    
    logging.info("*****현재 week 정보 받아옴*****")

    logging.info(current_week)
    context["task_instance"].xcom_push(key="current_week", value=current_week)



# MySQL 데이터베이스로부터 데이터를 가져오는 함수
def mysql_hook(**context):
    current_week = context["task_instance"].xcom_pull(task_ids="week_info", key="current_week")

    # hook 내용 작성
    logging.info("데이터베이스에서 데이터 가져오기")
    
    hook = MySqlHook.get_hook(conn_id="mysql-01")  # 미리 정의한 MySQL connection 적용
    connection = hook.get_conn()  # connection 하기
    cursor = connection.cursor()  # cursor 객체 만들기
    cursor.execute("use vod_rec")  # SQL 문 수행

    vods = pd.read_sql(f'select * from vods_sumut where week(log_dt)={current_week}', connection)
    logging.info(vods)


    connection.close()




# # DAG 설정
with DAG(
    dag_id="test",
    default_args=default_args,
    start_date=datetime(2023, 7, 29, tzinfo=local_tz),
    schedule_interval='@once'
) as dag:
    week_info = PythonOperator(
        task_id="week_info",
        python_callable=week_info_s3
    )
    data_query = PythonOperator(
        task_id="data_query",
        python_callable=mysql_hook
    )



    week_info >> data_query