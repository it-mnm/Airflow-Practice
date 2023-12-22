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
from scipy.sparse import coo_matrix


# 기본 설정
default_args = {
    'owner': 'admin',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}

local_tz = pendulum.timezone("Asia/Seoul")



# 피클 파일로 S3에 모델 업데이트
def model_update():
    s3_hook = S3Hook(aws_conn_id='aws_default')

    # S3에 파일이 존재하는지 확인
    s3_hook.check_for_key('model_genre.pkl', 'hello00.net-model')

    logging.info("*****pickle 파일 읽어오기*****")


# DAG 설정
with DAG(
    dag_id="pickle",
    default_args=default_args,
    start_date=datetime(2023, 12, 11, tzinfo=local_tz),
    schedule_interval='@once'
) as dag:
    pickle =PythonOperator(
        task_id="model_update",
        python_callable=model_update
    )
    pickle

