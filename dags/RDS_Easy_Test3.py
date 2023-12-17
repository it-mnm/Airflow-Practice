#RDS 데이터를 가져와 전처리 수행 후 머신러닝적용 및 성능평가

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
from tqdm import tqdm
from surprise import SVD, Reader, Dataset, accuracy

default_args = {
    'owner': 'admin',
    'retries': 5,
    'retry_delay': timedelta(minutes=10)
}

def mysql_hook(**context):
    # hook 내용 작성
    logging.info("데이터베이스에서 데이터 가져오기")
    
    hook = MySqlHook.get_hook(conn_id="mysql-01") # 미리 정의한 mysql connection 적용
    connection = hook.get_conn() # connection 하기
    cursor = connection.cursor() # cursor객체 만들기
    cursor.execute("use vod_rec") # sql문 수행
    cont_log = pd.read_sql('select * from contlog', connection)
    vod_log = pd.read_sql('select * from vods_sumut', connection)
    vod_info = pd.read_sql('select * from vodinfo', connection)
    context["task_instance"].xcom_push(key="cont_log", value=cont_log)
    context["task_instance"].xcom_push(key="vod_log", value=vod_log)
    context["task_instance"].xcom_push(key="vod_info", value=vod_info)
    connection.close()



def data_preprocessing(**context):
    logging.info("데이터 전처리")

    #mysql_hook 함수에서 사용했던 변수 불러오기
    cont_log = context["task_instance"].xcom_pull(task_ids="data_query", key="cont_log")
    vod_log = context["task_instance"].xcom_pull(task_ids="data_query", key="vod_log")
    vod_info = context["task_instance"].xcom_pull(task_ids="data_query", key="vod_info")

    # e_bool == 0 인 데이터만 뽑기
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
    context["task_instance"].xcom_push(key="train", value=train)
    context["task_instance"].xcom_push(key="test", value=test)



        

def model_running(**context):
    # 모델 함수
    logging.info("모델 적용 및 성능평가")
    train = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="train")
    test = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="test")

    class SVDRecommendationModel:
        def __init__(self, train_data, test_data):
            self.train_data, self.test_data = self.dataset(train_data, test_data)
            self.score_matrix = self.score_matrix(train_data)
            self.model = self.fit()

        def score_matrix(self, train):
            score_matrix = train.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio')
            return score_matrix

        def dataset(self, train, test):
            reader = Reader(rating_scale=(0, 1))
            train_data = Dataset.load_from_df(train.dropna(), reader)
            test_data = Dataset.load_from_df(test, reader)

            train_data = train_data.build_full_trainset()
            test_data = test_data.build_full_trainset().build_testset()
            return train_data, test_data

        def fit(self):
            model = SVD(random_state=0)
            model.fit(self.train_data)
            return model

        def predict(self, subsr_id, program_id):
            return self.model.predict(subsr_id, program_id).est

        def recommend(self, subsr_id, N):
            user_rated = self.score_matrix.loc[subsr_id].dropna().index.tolist()
            user_unrated = self.score_matrix.loc[subsr_id].drop(user_rated).index.tolist()
            predictions = [self.predict(subsr_id, program_id) for program_id in user_unrated]
            result = pd.DataFrame({'program_id': user_unrated, 'pred_rating': predictions})
            top_N = result.sort_values(by='pred_rating', ascending=False)[:N]
            return top_N

        @staticmethod
        def precision_recall_at_k(target, prediction):
            num_hit = len(set(prediction).intersection(set(target)))
            precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
            recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
            return precision, recall

        def evaluate(self, test_data, N=10):
            precisions = []
            recalls = []

            for user in tqdm(test_data['subsr_id'].unique()):
                targets = test_data[test_data['subsr_id']==user]['program_id'].values
                predictions = self.recommend(user, N)['program_id'].values
                precision, recall = self.precision_recall_at_k(targets, predictions)
                precisions.append(precision)
                recalls.append(recall)

            return np.mean(precisions), np.mean(recalls)

        def calculate_rmse(self):
            # test 데이터에 대해 예측 진행
            predictions = self.model.test(self.test_data)
            rmse = accuracy.rmse(predictions)
            return rmse 
    
    svd_model = SVDRecommendationModel(train, test)

    # 성능 구하는 코드
    precision, recall = svd_model.evaluate(test)

    # subsr_id = 0인 사람에게 추천리스트 10개 뽑기
    logging.info(svd_model.recommend(0, 10))



        
with DAG(
    dag_id="ex_mysql_to_csv3",
    default_args=default_args,
    start_date=datetime(2022, 4, 30),
    schedule_interval='*/1 * * * *'  # 매 1분마다 실행
) as dag:
    data_query = PythonOperator(
        task_id="data_query",
        python_callable=mysql_hook
    )
    data_preprocess = PythonOperator(
    task_id="data_preprocessing",
    python_callable=data_preprocessing
    )
    model_run = PythonOperator(
        task_id="model_running_and_create",
        python_callable=model_running
    )


    data_query >> data_preprocess >> model_run