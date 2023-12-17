#RDS 의 컬럼 내용을 csv파일로 로컬에 저장 및 S3에 복사해서 업로드
from datetime import datetime, timedelta
from airflow import DAG
from airflow.hooks.mysql_hook import MySqlHook
from airflow.operators.python_operator import PythonOperator
from airflow.operators.mysql_operator import MySqlOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CopyObjectOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
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
    'mysql_to_s3_dag',
    default_args=default_args,
    description='DAG to extract MySQL data and store it in S3',
    schedule_interval=timedelta(days=1),  # Adjust the schedule_interval as needed
)

# def test_mysql_connection(**kwargs):
#     mysql_hook = MySqlHook(mysql_conn_id='mysql-01')  # Replace 'your_mysql_conn_id' with your MySQL connection ID

#     # Test MySQL connection
#     connection = mysql_hook.get_conn()
#     cursor = connection.cursor()
#     cursor.execute("SELECT 1")
#     result = cursor.fetchall()
#     cursor.close()
#     connection.close()

#     if result:
#         return 'MySQL connection successful.'
#     else:
#         raise Exception('MySQL connection test failed.')

def extract_mysql_data_to_csv(**kwargs):
    mysql_hook = MySqlHook(mysql_conn_id='mysql-01')  # Replace 'your_mysql_conn_id' with your MySQL connection ID

    # Query MySQL data
    connection = mysql_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute("SHOW DATABASES")  # Replace 'your_table' with the table you want to export
    results = cursor.fetchall()
    cursor.close()
    connection.close()

    # Convert the result to CSV format
    csv_content = '\n'.join([','.join(map(str, row)) for row in results])

    # Write CSV content to a temporary file
    csv_filename = './mysql_data.csv'
    with open(csv_filename, 'w') as csv_file:
        csv_file.write(csv_content)

    return csv_filename


def upload_csv_to_s3(**kwargs):
    ti = kwargs['ti']
    csv_filename = ti.xcom_pull(task_ids='extract_mysql_data_to_csv')
    upload_task.execute(context=kwargs, source_bucket_key=csv_filename)  # S3 복사 실행



start_task = DummyOperator(
    task_id='start',
    dag=dag,
)

# test_mysql_connection_task = PythonOperator(
#     task_id='test_mysql_connection',
#     python_callable=test_mysql_connection,
#     provide_context=True,
#     dag=dag,
# )

extract_mysql_data_task = PythonOperator(
    task_id='extract_mysql_data_to_csv',
    python_callable=extract_mysql_data_to_csv,
    provide_context=True,
    dag=dag,
)

upload_task = S3CopyObjectOperator(
    task_id='upload_csv_to_s3',
    source_bucket_key='./mysql_data.csv',
    dest_bucket_key='table.csv',
    dest_bucket_name='airflowexample',
    aws_conn_id='aws_default',
    dag=dag,
)

end_task = DummyOperator(
    task_id='end',
    dag=dag,
)

# Define the task dependencies
# start_task >> test_mysql_connection_task >> extract_mysql_data_task >> upload_to_s3_task >> end_task
start_task >> extract_mysql_data_task >> upload_task >> end_task
