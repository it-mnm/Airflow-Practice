import csv
import json
import pandas as pd
import numpy as np
import logging
import pendulum
import pickle
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


# 주차 정보 불러오기 및 새로운 주차정보 저장
def week_info_s3(**context):
    s3_hook = S3Hook(aws_conn_id='aws_default')

    # S3에 파일이 존재하는지 확인
    file_exists = s3_hook.check_for_key('week_info/week_info.json', 'hello00.net-airflow')


    #파일이 존재하지 않을 경우 새 파일 생성, 존재할 경우에는 읽어오기
    if not file_exists:
        logging.info("*****파일 존재하지 않으므로 새로생성*****")
        new_json = {"columns": ["week_info"],"values": [40]}
        updated_json_data = json.dumps(new_json, indent=2)
        current_week=new_json["values"][0]
        s3_hook.load_string(updated_json_data, 'week_info/week_info.json', 'hello00.net-airflow', replace=True)

    else:
        logging.info("*****파일 존재. 기존 파일 읽어옴*****")
        existing_data = s3_hook.read_key('week_info/week_info.json', 'hello00.net-airflow')
        existing_json = json.loads(existing_data)
        #이전 주차 정보 읽기
        current_week=existing_json["values"][0] + 1

        #기존 주차 정보 업데이트
        existing_json["values"] = [current_week]
        updated_json = json.dumps(existing_json, indent=2)
        s3_hook.load_string(updated_json, 'week_info/week_info.json', 'hello00.net-airflow', replace=True)
    
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
    vods = pd.read_sql(f'select * from vods_sumut where week(log_dt)<={current_week}', connection)
    conts = pd.read_sql(f'select * from contlog where week(log_dt)<={current_week}', connection)
    program_info_all = pd.read_sql('select * from vodinfo', connection)

    logging.info(users)
    logging.info(vods)
    logging.info(conts)
    logging.info(program_info_all)

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
    users = context["task_instance"].xcom_pull(task_ids="mysql_hook", key="users")
    vods = context["task_instance"].xcom_pull(task_ids="mysql_hook", key="vods")
    conts = context["task_instance"].xcom_pull(task_ids="mysql_hook", key="conts")
    program_info_all = context["task_instance"].xcom_pull(task_ids="mysql_hook", key="program_info_all")


    # 전처리 시작
    # e_bool == 0 인 데이터만 뽑기
    vod_log = vods[vods['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'use_tms', 'disp_rtm_sec', 'count_watch', 'month']]
    cont_log = conts[conts['e_bool']==0][['subsr_id', 'program_id', 'program_name', 'episode_num', 'log_dt', 'month']]
    vod_info = program_info_all[program_info_all['e_bool']==0][['program_id','program_name', 'ct_cl', 'program_genre', 'release_date', 'age_limit']]
    user_info = users.copy()

    vod_log['use_tms_ratio'] = vod_log['use_tms'] / vod_log['disp_rtm_sec']
    vod_log['log_dt'] = pd.to_datetime(vod_log['log_dt'])
    cont_log['log_dt'] = pd.to_datetime(cont_log['log_dt'])
    cont_log['recency'] = (cont_log['log_dt'].max() - cont_log['log_dt']).dt.days # 최근성

    log = pd.concat([vod_log[['subsr_id', 'program_id']], cont_log[['subsr_id', 'program_id']]]).drop_duplicates().reset_index(drop=True)
    log = log.merge(cont_log.groupby(['subsr_id', 'program_id'])[['program_name']].count().reset_index().rename(columns={'program_name':'click_cnt'}), how='left')
    log = log.merge(cont_log.groupby(['subsr_id', 'program_id'])[['recency']].min().reset_index().rename(columns={'recency':'click_recency'}), how='left')

    # (subsr_id, program_id) 쌍에 대해 하나의 평점만 남겨야 함
    # => use_tms_ratio 는 최대값으로 남기고, 시청 count도 해서 추가한다.
    log = log.merge(vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].max().reset_index().rename(columns={'use_tms_ratio':'watch_tms_max'}), how='left')
    log = log.merge(vod_log.groupby(['subsr_id', 'program_id'])[['use_tms_ratio']].count().reset_index().rename(columns={'use_tms_ratio':'watch_cnt'}), how='left')

    replace_value = log['click_cnt'].quantile(0.95) # 95% 로 최대값 고정
    log.loc[log[log['click_cnt'] > replace_value].index, 'click_cnt'] = replace_value

    log['click_recency'] = log['click_recency'].max() - log['click_recency']


    # 시청기록만 사용
    replace_value = log['watch_cnt'].quantile(0.95)  # 95% 로 최대값 고정
    log.loc[log[log['watch_cnt'] > replace_value].index, 'watch_cnt'] = replace_value

    log_rating = log.copy()
    log_rating['rating'] = log_rating['watch_tms_max']
    log_rating.loc[log_rating[log_rating['rating']>0].index, '시청여부'] = 1
    rating_df = log_rating[['subsr_id', 'program_id', 'rating', '시청여부']]
    vod_info['program_id'] = vod_info['program_id'].astype('int')
    rating_df['program_id'] = rating_df['program_id'].astype('int')
    vod_info = vod_info.merge(log_rating.groupby(['program_id', 'subsr_id'])[['click_cnt']].count().groupby('program_id').count().reset_index(), how='left')

    max_subsr_id = rating_df['subsr_id'].max()
    user_info = user_info[user_info['subsr_id'] < max_subsr_id]
    
    logging.info(vod_info)
    logging.info(rating_df)

    context["task_instance"].xcom_push(key="rating_df", value=rating_df)
    context["task_instance"].xcom_push(key="vod_info", value=vod_info)
    context["task_instance"].xcom_push(key="user_info", value=user_info)


# 모델 적용 및 성능추출
def model_running(**context):
    rating_df = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="rating_df")
    vod_info = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="vod_info")
    user_info = context["task_instance"].xcom_pull(task_ids="data_preprocessing", key="user_info")
    # 모델 함수
    logging.info("모델 적용 및 성능평가")

    class LightFM_Model:
        def __init__(self, data, vod_info, user_info, param_loss='logistic', param_components=10, param_epochs=10):
            self.vod_info = vod_info
            self.user_info = user_info
            self.train, self.test = self.split_evaluate(data)
            self.train_interactions, self.train_weights = self.dataset(self.train)
            self.score_matrix = self.create_score_matrix(data)
            self.score_matrix_evaluate = self.create_score_matrix(self.train)
            ####
            self.loss = param_loss
            self.components = param_components
            self.epoch = param_epochs
            self.precision, self.recall, self.map, self.mar, self.test_diversity, self.user_metrics = self.evaluate(self.train_interactions, self.train_weights, self.test)
            self.model = self.train_model(data)
            self.all_diversity = self.evaluate_all(self.model)

        def split_evaluate(self, data):
            #### 수정 완료
            train, test = train_test_split(data[data['시청여부'].notnull()][['subsr_id', 'program_id', 'rating']], test_size=0.25, random_state=0)
            train = data[['subsr_id', 'program_id', 'rating', '시청여부']].copy()
            train.loc[test.index, 'rating'] = np.nan
            return train, test

        #### 수정 완료
        def create_score_matrix(self, data):
            df = data.copy()
            df.loc[df[(df['시청여부']==1) & df['rating'].notnull()].index, 'score_matrix'] = 1
            score_matrix = df.pivot(columns='program_id', index='subsr_id', values='score_matrix')
            return score_matrix

        def dataset(self, train):
            dataset = Dataset()
            dataset.fit(users = train['subsr_id'].sort_values().unique(),
                        items = train['program_id'].sort_values().unique())
            num_users, num_vods = dataset.interactions_shape()
            ####
            train['interaction'] = train['rating'].apply(lambda x: 0 if np.isnan(x) else 1)
            train_dropna = train.dropna()
            train_interactions = coo_matrix((train_dropna['interaction'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(train['subsr_id'].max() + 1, train['program_id'].max() +1))
            train_weights = coo_matrix((train_dropna['rating'], (train_dropna['subsr_id'], train_dropna['program_id'])), shape=(train['subsr_id'].max() + 1, train['program_id'].max() +1))
            #(train_interactions, train_weights) = dataset.build_interactions(train[['subsr_id', 'program_id', 'rating']].dropna().values)
            return train_interactions, train_weights

        def fit(self, fitting_interactions, fitting_weights):
            model = LightFM(random_state=0, loss=self.loss, no_components=self.components)
            model.fit(interactions=fitting_interactions, sample_weight=fitting_weights, verbose=1, epochs=self.epoch)
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
            ####
            data['interaction'] = data['rating'].apply(lambda x: 0 if np.isnan(x) else 1)
            data_dropna = data.dropna()
            train_interactions = coo_matrix((data_dropna['interaction'], (data_dropna['subsr_id'], data_dropna['program_id'])))
            train_weights = coo_matrix((data_dropna['rating'], (data_dropna['subsr_id'], data_dropna['program_id'])))
            #(train_interactions, train_weights) = dataset.build_interactions(data[['subsr_id', 'program_id', 'rating']].dropna().values)
            # fitting
            model = self.fit(train_interactions, train_weights)
            return model
    # 모델 클래스 객체 생성하고, 성능 출력하는 코드
    lfm_model = LightFM_Model(rating_df, vod_info, user_info, 'warp', 30, 10)


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
    context["task_instance"].xcom_push(key="lfm_model", value=lfm_model)


# 성능 S3에 적재
def convert_to_json(**context):
    s3_hook = S3Hook(aws_conn_id='aws_default')
    current_week = context["task_instance"].xcom_pull(task_ids="week_info", key="current_week")
    current_year = datetime.now().year
    # 주차의 첫째날과 마지막날을 계산
    # first_day_of_week = datetime.strptime(f'{current_year}-W{current_week-1}-0', "%Y-W%W-%w").date()
    # last_day_of_week = first_day_of_week + timedelta(days=6)
    # current_week = f"{first_day_of_week.strftime('%Y.%m.%d')} ~ {last_day_of_week.strftime('%Y.%m.%d')}"
    logging.info(f"{current_week} 날짜 삽입")

    # Metrics to append
    metrics = [
        {"name": "Precision", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="Precision")},
        {"name": "Recall", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="Recall")},
        {"name": "MAP", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="MAP")},
        {"name": "MAR", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="MAR")},
        {"name": "test_Diversity", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="test_Diversity")},
        {"name": "all_Diversity", "value": context["task_instance"].xcom_pull(task_ids="model_running", key="all_Diversity")},
    ]



    for metric in metrics:
        metric_name = metric["name"]
        metric_value = metric["value"]

        # S3에 파일이 존재하는지 확인
        file_exists = s3_hook.check_for_key(f'model_accuracy/{metric_name.lower()}.json', 'hello00.net-airflow')
        
        start_date = '2023.08.01'
        # 파일이 존재하지 않을 경우 새 파일 생성, 존재할 경우에는 기존 파일에 새로운 데이터 추가
        if not file_exists:
            new_metrics = {"columns": [[f'{metric_name}({start_date})']], "values": [[metric_value]]}
            updated_json_data = json.dumps(new_metrics, indent=2)
        else:
            existing_data = s3_hook.read_key(f'model_accuracy/{metric_name.lower()}.json', 'hello00.net-airflow')
            existing_metrics = json.loads(existing_data)

            # 새로운 데이터 추가
            existing_metrics["columns"]=[[f'{metric_name}({start_date})']]
            existing_metrics["values"].append([metric_value])

            # Json으로 변환
            updated_json_data = json.dumps(existing_metrics, indent=2)

        # S3에 다시 저장
        s3_hook.load_string(updated_json_data, f'model_accuracy/{metric_name.lower()}.json', 'hello00.net-airflow', replace=True)


# 피클 파일로 S3에 모델 업데이트
def model_update(**context):
    lfm_model = context["task_instance"].xcom_pull(task_ids="model_running", key="lfm_model")

    pickle_file_path = 'model_lightfm.pkl'

    with open('pickle_file_path', 'wb') as f:
        pickle.dump(lfm_model, f)

    s3_hook = S3Hook(aws_conn_id='aws_default')

    s3_bucket_name = 'hello00.net-model'
    s3_key = 'model_lightfm.pkl'

    # S3에 피클 파일 업로드
    s3_hook.load_file(
        filename=pickle_file_path,
        key=s3_key,
        bucket_name=s3_bucket_name,
        replace=True  # 기존 파일 덮어쓰기
    )




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
    database = PythonOperator(
        task_id="mysql_hook",
        python_callable=mysql_hook
    )
    data_preprocess = PythonOperator(
    task_id="data_preprocessing",
    python_callable=data_preprocessing
    )
    model_apply = PythonOperator(
        task_id="model_running",
        python_callable=model_running
    )
    dashboard_data_save = PythonOperator(
        task_id="convert_to_json",
        python_callable=convert_to_json
    )
    # pickle =PythonOperator(
    #     task_id="model_update",
    #     python_callable=pickle_load
    # )

    week_info >> database >> data_preprocess >> model_apply >> dashboard_data_save
    # week_info >> database >> data_preprocess >> model_apply >> pickle

