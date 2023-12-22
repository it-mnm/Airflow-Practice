import pickle
from airflow.providers.amazon.aws.hooks.s3 import S3Hook

def read_pickle_from_s3(bucket_name, key):
    s3_hook = S3Hook(aws_conn_id='aws_default')

    try:
        # S3에서 Pickle 파일 다운로드
        pickle_content = s3_hook.read_key(key, bucket_name)

        # Pickle 파일 읽기
        data = pickle.loads(pickle_content)

        return data
    except Exception as e:
        print(f"Error reading Pickle file from S3: {e}")
        return None

# 사용 예시
bucket_name = 'your_s3_bucket_name'
pickle_key = 'your/pickle/file/path.pickle'

pickle_data = read_pickle_from_s3(bucket_name, pickle_key)

if pickle_data:
    print("Pickle file contents:")
    print(pickle_data)
else:
    print("Failed to read Pickle file from S3.")
