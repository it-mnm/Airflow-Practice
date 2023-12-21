# RDS 데이터를 가져와 전처리 수행 후 머신러닝 적용 및 성능평가
# JSON 파일로 S3에 저장
# 성능 데이터 적재

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


# 주차 정보 불러오기 및 새로운 주차정보 저장
def week_info_s3(**context):
    s3_hook = S3Hook(aws_conn_id='aws_default')

    # S3에 파일이 존재하는지 확인
    file_exists = s3_hook.check_for_key('week_info/week_info.json', 'hello00.net-airflow')


    #파일이 존재하지 않을 경우 새 파일 생성, 존재할 경우에는 읽어오기
    if not file_exists:
        logging.info("*****파일 존재하지 않으므로 새로생성*****")
        new_metrics = {"columns": ["week_info"],"values": [31]}
        updated_json_data = json.dumps(new_metrics, indent=2)
        current_week=new_metrics["values"][0]
        s3_hook.load_string(updated_json_data, 'week_info/week_info.json', 'hello00.net-airflow', replace=True)

    else:
        logging.info("*****파일 존재. 기존 파일 읽어옴*****")
        existing_data = s3_hook.read_key('week_info/week_info.json', 'hello00.net-airflow')
        existing_metrics = json.loads(existing_data)
        #이전 주차 정보 읽기
        current_week=existing_metrics["values"][0]

        #기존 주차 정보 업데이트
        existing_metrics["values"] = [current_week + 1]
        updated_json_data = json.dumps(existing_metrics, indent=2)
        s3_hook.load_string(updated_json_data, 'week_info/week_info.json', 'hello00.net-airflow', replace=True)
    
    logging.info("*****현재 week 정보 받아옴*****")

    logging.info(current_week)
    context["task_instance"].xcom_push(key="current_week", value=current_week)


# MySQL 데이터베이스로부터 데이터를 가져오는 함수
def mysql_hook(**context):
    
    current_week = context["task_instance"].xcom_pull(task_ids="week_info", key="current_week")
    logging.info("데이터베이스에서 데이터 가져오기")
    
    hook = MySqlHook.get_hook(conn_id="mysql-01")  # 미리 정의한 MySQL connection 적용
    connection = hook.get_conn()  # connection 하기
    cursor = connection.cursor()  # cursor 객체 만들기
    cursor.execute("use vod_rec")  # SQL 문 수행



    users = pd.read_sql('select * from userinfo', connection)
    vods = pd.read_sql(f'select * from vods_sumut where week(log_dt)={current_week}', connection)
    conts = pd.read_sql(f'select * from contlog where week(log_dt)={current_week}', connection)
    program_info_all = pd.read_sql('select * from vodinfo', connection)

    context["task_instance"].xcom_push(key="users", value=users)
    context["task_instance"].xcom_push(key="vods", value=vods)
    context["task_instance"].xcom_push(key="conts", value=conts)
    context["task_instance"].xcom_push(key="program_info_all", value=program_info_all)
    context["task_instance"].xcom_push(key="program_info_all", value=program_info_all)

    connection.close()


# 데이터 전처리 함수
def data_preprocessing(**context):
    logging.info("데이터 전처리")

    # mysql_hook 함수에서 사용했던 변수 불러오기
    users = context["task_instance"].xcom_pull(task_ids="data_query", key="users")
    vods = context["task_instance"].xcom_pull(task_ids="data_query", key="vods")
    conts = context["task_instance"].xcom_pull(task_ids="data_query", key="conts")
    program_info_all = context["task_instance"].xcom_pull(task_ids="data_query", key="program_info_all")


    # e_bool == 0 인 데이터만 뽑기
    vod_log = vods[vods['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month']]
    cont_log = conts[conts['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'month']]
    vod_info = program_info_all[program_info_all['e_bool']==0][['program_id','program_name', 'ct_cl', 'program_genre', 'release_date', 'age_limit']]
    user_info = users.copy()

    # use_tms / running time
    vod_log['use_tms_ratio'] = vod_log['use_tms'] / vod_log['disp_rtm_sec']
    use_tms_ratio = vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].max().reset_index()

    click_cnt = cont_log.groupby(['subsr_id', 'program_id'])[['log_dt']].count().reset_index().rename(columns={'log_dt':'click_cnt'})

    # vod, cont 합치기
    rating = pd.concat([vod_log[['subsr_id', 'program_id']], cont_log[['subsr_id', 'program_id']]]).drop_duplicates().reset_index(drop=True)
    rating = rating.merge(use_tms_ratio, how='left').merge(click_cnt, how='left')

    vod_info = vod_info.merge(rating.groupby('program_id').sum()[['click_cnt']].reset_index(), how='left')
    rating = rating[['subsr_id', 'program_id', 'use_tms_ratio']]

    vod_info['program_id'] = vod_info['program_id'].astype('int')
    rating['program_id'] = rating['program_id'].astype('int')

    vod_info.loc[1743, 'program_genre'] = '드라마, 스릴러, 범죄, 가족'
    vod_info.loc[4059, 'program_genre'] = '드라마, 가족'
    vod_info.loc[3478, 'program_genre'] = '드라마, 로맨스'
    vod_info.loc[3482, 'program_genre'] = '드라마, 로맨스, 코미디, 학교'
    vod_info.loc[3487, 'program_genre'] = '드라마, 로맨스, 코미디'
    vod_info.loc[4056, 'program_genre'] = '드라마, 액션'
    vod_info.loc[3471, 'program_genre'] = '드라마, 코미디, 로맨스'
    vod_info.loc[2706, 'program_genre'] = '드라마, 공포, 시대극'

    context["task_instance"].xcom_push(key="rating", value=rating)
    context["task_instance"].xcom_push(key="vod_info", value=vod_info)
    context["task_instance"].xcom_push(key="user_info", value=user_info)


# 모델 적용 및 성능평가 함수
def model_running(**context):
    rating = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="rating")
    vod_info = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="vod_info")
    user_info = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="user_info")
    # 모델 함수
    logging.info("모델 적용 및 성능평가")

    class LightFM_Model:
        def __init__(self, data, vod_info, user_info):
            self.vod_info = vod_info
            self.user_info = user_info
            self.train, self.test = self.split_evaluate(data)
            self.train_interactions, self.train_weights = self.dataset(self.train)
            self.score_matrix = self.create_score_matrix(data)
            self.score_matrix_evaluate = self.create_score_matrix(self.train)
            self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics = self.evaluate(self.train_interactions, self.train_weights, self.test)
            self.model = self.train_model(data)
            self.all_diversity = self.evaluate_all(self.model)

        def split_evaluate(self, data):
            train, test = train_test_split(data.dropna(), test_size=0.25, random_state=0)
            train = data.copy()
            train.loc[test.index, 'use_tms_ratio'] = np.nan
            return train, test

        def create_score_matrix(self, data):
            score_matrix = data.pivot(columns='program_id', index='subsr_id', values='use_tms_ratio')
            return score_matrix

        def dataset(self, train):
            dataset = Dataset()
            dataset.fit(users = train['subsr_id'].sort_values().unique(),
                        items = train['program_id'].sort_values().unique())
            num_users, num_vods = dataset.interactions_shape()
            (train_interactions, train_weights) = dataset.build_interactions(train.dropna().values)
            # (test_interactions, test_weights) = dataset.build_interactions(test.values)
            return train_interactions, train_weights

        def fit(self, fitting_interactions, fitting_weights, epochs=10):
            model = LightFM(random_state=0)
            model.fit(interactions=fitting_interactions, sample_weight=fitting_weights, verbose=1, epochs=epochs)
            return model

        def predict(self, subsr_id, program_id, model):
            pred = model.predict([subsr_id], [program_id])
            return pred

        def recommend(self, subsr_id, score_matrix, model, N):
            # 안 본 vod 추출
            user_rated = score_matrix.loc[subsr_id].dropna().index.tolist()
            user_unrated = score_matrix.columns.drop(user_rated).tolist()
            # 안본 vod에 대해서 예측하기
            predictions = model.predict(int(subsr_id), user_unrated)
            result = pd.DataFrame({'program_id':user_unrated, 'pred_rating':predictions})
            # pred값에 따른 정렬해서 결과 띄우기
            top_N = result.sort_values(by='pred_rating', ascending=False)[:N]
            return top_N

        @staticmethod
        def precision_recall_at_k(target, prediction):
            num_hit = len(set(prediction).intersection(set(target)))
            precision = float(num_hit) / len(prediction) if len(prediction) > 0 else 0.0
            recall = float(num_hit) / len(target) if len(target) > 0 else 0.0
            return precision, recall

        @staticmethod
        def map_at_k(target, prediction, k=10):
            num_hits = 0
            precision_at_k = 0.0
            for i, p in enumerate(prediction[:k]):
                if p in target:
                    num_hits += 1
                    precision_at_k += num_hits / (i + 1)
            if not target.any():
                return 0.0
            return precision_at_k / min(k, len(target))

        @staticmethod
        def mar_at_k(target, prediction, k=10):
            num_hits = 0
            recall_at_k = 0.0
            for i, p in enumerate(prediction[:k]):
                if p in target:
                    num_hits += 1
                    recall_at_k += num_hits / len(target)
            if not target.any():
                return 0.0
            return recall_at_k / min(k, len(target))

        def evaluate(self, train_interactions, train_weights, test, N=10):
            evaluate_model = self.fit(train_interactions, train_weights)
            result = pd.DataFrame()
            precisions = []
            recalls = []
            map_values = []
            mar_values = []
            user_metrics = []

            for idx, user in enumerate(tqdm(test['subsr_id'].unique())):
                targets = test[test['subsr_id']==user]['program_id'].values
                predictions = self.recommend(user, self.score_matrix_evaluate, evaluate_model, N)['program_id'].values
                precision, recall = self.precision_recall_at_k(targets, predictions)
                map_at_k_value = self.map_at_k(targets, predictions)
                mar_at_k_value = self.mar_at_k(targets, predictions)
                precisions.append(precision)
                recalls.append(recall)
                map_values.append(map_at_k_value)
                mar_values.append(mar_at_k_value)
                user_metrics.append({'subsr_id': user, 'precision': precision, 'recall': recall, 'map_at_k': map_at_k_value, 'mar_at_k': mar_at_k_value})

                result.loc[idx, 'subsr_id'] = user
                for rank in range(len(predictions)):
                    result.loc[idx, f'vod_{rank}'] = predictions[rank]

            list_sim = cosine_similarity(result.iloc[:, 1:])
            list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))

            return np.mean(precisions), np.mean(recalls), np.mean(map_values), np.mean(mar_values), 1-list_similarity, pd.DataFrame(user_metrics)

        def evaluate_all(self, model, N=10):
            result = pd.DataFrame()
            for idx, user in enumerate(tqdm(self.user_info['subsr_id'])):
                predictions = self.recommend(user, self.score_matrix, model, N)['program_id'].values
                result.loc[idx, 'subsr_id'] = user
                for rank in range(len(predictions)):
                    result.loc[idx, f'vod_{rank}'] = predictions[rank]

            list_sim = cosine_similarity(result.iloc[:, 1:])
            list_similarity = np.sum(list_sim - np.eye(len(result))) / (len(result) * (len(result) - 1))
            return 1 - list_similarity


        def train_model(self, data):
            # 최종 학습 데이터셋 만들기
            dataset = Dataset()
            dataset.fit(users = data['subsr_id'].sort_values().unique(),
                        items = data['program_id'].sort_values().unique())
            num_users, num_vods = dataset.interactions_shape()
            (train_interactions, train_weights) = dataset.build_interactions(data.dropna().values)
            # fitting
            model = self.fit(train_interactions, train_weights)
            return model
    # 모델 클래스 객체 생성하고, 성능 출력하는 코드
    lfm_model = LightFM_Model(rating, vod_info, user_info)

    Precision = round(lfm_model.precision, 5)
    Recall = round(lfm_model.recall, 5)
    MAP = round(lfm_model.map, 5)
    MAR = round(lfm_model.mar, 5)
    test_Diversity = round(lfm_model.test_diversity, 5)
    all_Diversity = round(lfm_model.all_diversity, 5)

    context["task_instance"].xcom_push(key="Precision", value=Precision)
    context["task_instance"].xcom_push(key="Recall", value=Recall)
    context["task_instance"].xcom_push(key="MAP", value=MAP)
    context["task_instance"].xcom_push(key="MAR", value=MAR)
    context["task_instance"].xcom_push(key="test_Diversity", value=test_Diversity)
    context["task_instance"].xcom_push(key="all_Diversity", value=all_Diversity)


# JSON 파일로 변환 및 S3에 업로드 함수
def convert_to_json(**context):
    s3_hook = S3Hook(aws_conn_id='aws_default')
    current_week = context["task_instance"].xcom_pull(task_ids="week_info", key="current_week")

    # current_year = datetime.now().year
    # # 주차의 첫째날과 마지막날을 계산
    # first_day_of_week = datetime.strptime(f'{current_year}-W{current_week}-1', "%Y-W%W-%w").date()
    # last_day_of_week = first_day_of_week + timedelta(days=6)


    # current_week = f"{first_day_of_week.strftime('%Y.%m.%d')}~{last_day_of_week.strftime('%Y.%m.%d')}"

    # logging.info(f"{current_week} 주의 메트릭 처리")

    # Metrics to append
    metrics = [
        {"name": "Precision", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="Precision")},
        {"name": "Recall", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="Recall")},
        {"name": "MAP", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="MAP")},
        {"name": "MAR", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="MAR")},
        {"name": "test_Diversity", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="test_Diversity")},
        {"name": "all_Diversity", "value": context["task_instance"].xcom_pull(task_ids="model_running_and_create", key="all_Diversity")},
    ]



    for metric in metrics:
        metric_name = current_week
        metric_value = metric["value"]

        # S3에 파일이 존재하는지 확인
        file_exists = s3_hook.check_for_key(f'{metric_name.lower()}.json', 'hello00.net-airflow')


        #파일이 존재하지 않을 경우 새 파일 생성, 존재할 경우에는 기존 파일에 새로운 데이터 추가
        if not file_exists:
            # If the file doesn't exist, create a new structure
            new_metrics = {"columns": [[metric_name]], "values": [[metric_value]]}
            updated_json_data = json.dumps(new_metrics, indent=2)
        else:
            # Retrieve existing data from S3
            existing_data = s3_hook.read_key(f'{metric_name.lower()}.json', 'hello00.net-airflow')
            existing_metrics = json.loads(existing_data)

            # Append new metric to existing data
            existing_metrics["columns"] = [metric_name]
            existing_metrics["values"].append([metric_value])

            # Convert the updated data to JSON
            updated_json_data = json.dumps(existing_metrics, indent=2)

        # Save back to S3
        s3_hook.load_string(updated_json_data, f'{metric_name.lower()}.json', 'hello00.net-airflow', replace=True)




# DAG 설정
with DAG(
    dag_id="vod_rec_v3",
    default_args=default_args,
    start_date=datetime(2023, 12, 11, tzinfo=local_tz),
    schedule_interval='@once'  # 매주 월요일 00:00 마다 실행
) as dag:
    week_info = PythonOperator(
    task_id="week_info",
    python_callable=week_info_s3
    )
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
    upload_json_to_s3 =PythonOperator(
        task_id="upload_to_s3",
        python_callable=convert_to_json
    )

    week_info >> data_query >> data_preprocess >> model_run >> upload_json_to_s3
