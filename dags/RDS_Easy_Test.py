#RDS의 MySQL에서 쿼리문으로 Database목록을 조회
#조회된 목록을 CSV 파일로 S3 버킷 내부에 바로 저장

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.hooks.mysql_hook import MySqlHook
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'rds_to_s3_dag',
    default_args=default_args,
    description='DAG to extract RDS data and store it in S3',
    schedule_interval=timedelta(days=1),  # Adjust the schedule_interval as needed
)

def extract_rds_data(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id='mysql-01')  # Replace 'your_mysql_conn_id' with your MySQL connection ID

    # Query MySQL data
    connection = mysql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute('SHOW DATABASES')
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    # Convert the result to CSV format
    csv_content = '\n'.join([','.join(map(str, row)) for row in results])

    # Write CSV content to S3
    s3_hook = S3Hook(aws_conn_id='aws_default')
    s3_hook.load_string(csv_content, 'bucketname/databases.csv') # Replace 'bucketname' with your S3 bucket name


start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

extract_data_task = PythonOperator(
    task_id='extract_rds_data',
    python_callable=extract_rds_data,
    provide_context=True,
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define the task dependencies
start_task >> extract_data_task >> end_task
