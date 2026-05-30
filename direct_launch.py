"""
direct_launch.py
----------------
Bypasses the SageMaker Python SDK and submits a training job
directly via boto3. Avoids the SDK session issues entirely.
"""

import boto3, tarfile, os, json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────
ROLE_ARN    = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role"
BUCKET = "sagemaker-us-east-1-730335300762"
REGION      = "us-east-1"
HF_TOKEN    = os.environ["HF_TOKEN"]
JOB_NAME    = f"edu-qlora-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
IMAGE_URI = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
# ──────────────────────────────────────────────────────────────────

def upload_source():
    """Package train.py + requirements.txt and upload to S3."""
    tarball = "sourcedir.tar.gz"
    with tarfile.open(tarball, "w:gz") as tar:
        tar.add("train.py")
        tar.add("requirements.txt")

    s3_key = f"source/{JOB_NAME}/sourcedir.tar.gz"
    s3 = boto3.client("s3", region_name=REGION)
    s3.upload_file(tarball, BUCKET, s3_key)
    os.remove(tarball)

    uri = f"s3://{BUCKET}/{s3_key}"
    print(f"Source uploaded: {uri}")
    return uri

def submit_job(source_uri):
    sm = boto3.client("sagemaker", region_name=REGION)

    response = sm.create_training_job(
        TrainingJobName=JOB_NAME,
        AlgorithmSpecification={
            "TrainingImage":     IMAGE_URI,
            "TrainingInputMode": "File",
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                "ChannelName":     "train",
                "DataSource":      {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://edu-qlora-art/edu-lora-dataset/train.jsonl", "S3DataDistributionType": "FullyReplicated"}},
                "ContentType":     "application/json",
                "InputMode":       "File",
            },
            {
                "ChannelName":     "eval",
                "DataSource":      {"S3DataSource": {"S3DataType": "S3Prefix", "S3Uri": "s3://edu-qlora-art/edu-lora-dataset/eval.jsonl", "S3DataDistributionType": "FullyReplicated"}},
                "ContentType":     "application/json",
                "InputMode":       "File",
            },
        ],
        OutputDataConfig={
            "S3OutputPath": f"s3://{BUCKET}/output/"
        },
        ResourceConfig={
            "InstanceType":   "ml.g5.xlarge",
            "InstanceCount":  1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 7200,"MaxWaitTimeInSeconds": 10800,},
        EnableManagedSpotTraining=True,
        CheckpointConfig={"S3Uri": f"s3://{BUCKET}/checkpoints/{JOB_NAME}/"},
        Environment={
            "HF_TOKEN":                    HF_TOKEN,
            "SAGEMAKER_SUBMIT_DIRECTORY":  source_uri,
            "SAGEMAKER_PROGRAM":           "train.py",
        },
    )

    print(f"\nJob submitted: {JOB_NAME}")
    print(f"Status: {response['ResponseMetadata']['HTTPStatusCode']}")
    print(f"\nWatch it here:")
    print(f"https://us-east-1.console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs/{JOB_NAME}")

def main():
    source_uri = upload_source()
    submit_job(source_uri)

if __name__ == "__main__":
    main()
