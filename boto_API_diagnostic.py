import boto3
from dotenv import load_dotenv
load_dotenv()

client = boto3.client('sagemaker', region_name='us-east-1')

try:
    jobs = client.list_training_jobs(MaxResults=5)
    print("SageMaker API reachable")
    print(jobs['TrainingJobSummaries'])
except Exception as e:
    print(f"ERROR: {e}")