"""
run_eval.py
-----------
Launches a SageMaker Processing Job to evaluate the fine-tuned adapter.
Uses direct boto3 — same pattern as direct_launch.py.

Usage:
    python run_eval.py
"""

import boto3, os, tarfile
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG — edit if needed ───────────────────────────────────────────────────
ROLE_ARN   = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role"
BUCKET     = "sagemaker-us-east-1-730335300762"
REGION     = "us-east-1"
HF_TOKEN   = os.environ["HF_TOKEN"]

MODEL_S3_URI = "s3://sagemaker-us-east-1-730335300762/output/edu-qlora-20260531-183243/output/model.tar.gz"
EVAL_S3_URI  = "s3://edu-qlora-art/edu-lora-dataset/eval.jsonl"

IMAGE_URI  = "763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
JOB_NAME   = f"edu-qlora-eval-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
# ──────────────────────────────────────────────────────────────────────────────


def upload_script():
    """Upload eval.py to S3 so the processing container can access it."""
    s3     = boto3.client("s3", region_name=REGION)
    s3_key = f"eval-source/{JOB_NAME}/eval.py"
    s3.upload_file("eval.py", BUCKET, s3_key)
    s3_uri = f"s3://{BUCKET}/eval-source/{JOB_NAME}/"
    print(f"Script uploaded: {s3_uri}", flush=True)
    return s3_uri


def launch_processing_job(code_s3_uri):
    sm = boto3.client("sagemaker", region_name=REGION)

    response = sm.create_processing_job(
        ProcessingJobName=JOB_NAME,

        ProcessingInputs=[
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri":                   MODEL_S3_URI,
                    "LocalPath":               "/opt/ml/processing/input/model",
                    "S3DataType":              "S3Prefix",
                    "S3InputMode":             "File",
                    "S3DataDistributionType":  "FullyReplicated",
                },
            },
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri":                   EVAL_S3_URI,
                    "LocalPath":               "/opt/ml/processing/input/data",
                    "S3DataType":              "S3Prefix",
                    "S3InputMode":             "File",
                    "S3DataDistributionType":  "FullyReplicated",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri":                   code_s3_uri,
                    "LocalPath":               "/opt/ml/processing/input/code",
                    "S3DataType":              "S3Prefix",
                    "S3InputMode":             "File",
                    "S3DataDistributionType":  "FullyReplicated",
                },
            },
        ],

        ProcessingOutputConfig={
            "Outputs": [
                {
                    "OutputName": "results",
                    "S3Output": {
                        "S3Uri":         f"s3://{BUCKET}/eval-output/{JOB_NAME}/",
                        "LocalPath":     "/opt/ml/processing/output",
                        "S3UploadMode":  "EndOfJob",
                    },
                }
            ]
        },

        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount":    1,
                "InstanceType":     "ml.g5.xlarge",   # same quota we already have
                "VolumeSizeInGB":   30,
            }
        },

        AppSpecification={
            "ImageUri": IMAGE_URI,
            "ContainerEntrypoint": [
                "/bin/bash", "-c",
                "pip install -q peft datasets && "
                "python -u /opt/ml/processing/input/code/eval.py 2>&1"
            ],
        },

        RoleArn=ROLE_ARN,
        Environment={
            "HF_TOKEN":               HF_TOKEN,
            "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
        },
        StoppingCondition={"MaxRuntimeInSeconds": 3600},
    )

    print(f"\nProcessing job submitted: {JOB_NAME}")
    print(f"Status: {response['ResponseMetadata']['HTTPStatusCode']}")
    print(f"\nWatch it here:")
    print(f"https://us-east-1.console.aws.amazon.com/sagemaker/home?"
          f"region=us-east-1#/processing-jobs/{JOB_NAME}")
    print(f"\nResults will land at:")
    print(f"s3://{BUCKET}/eval-output/{JOB_NAME}/eval_results.json")


def main():
    print(f"Launching eval job: {JOB_NAME}")
    code_s3_uri = upload_script()
    launch_processing_job(code_s3_uri)


if __name__ == "__main__":
    main()
