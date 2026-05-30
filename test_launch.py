import boto3
import sagemaker
from sagemaker.huggingface import HuggingFace
from dotenv import load_dotenv
import os
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger('sagemaker').setLevel(logging.DEBUG)
load_dotenv()

ROLE_ARN   = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role"
REGION     = "us-east-1"

boto_session = boto3.Session(region_name=REGION, profile_name="default")
session      = sagemaker.Session(boto_session=boto_session)

estimator = HuggingFace(
    entry_point="train.py",
    source_dir=".",
    instance_type="ml.g5.xlarge",
    instance_count=1,
    role=ROLE_ARN,
    transformers_version="4.56.2",
    pytorch_version="2.8.0",
    py_version="py312",
    sagemaker_session=session,
    environment={"HF_TOKEN": os.environ["HF_TOKEN"]},
)

try:
    estimator.fit(
        {"train": "s3://edu-qlora-art/edu-lora-dataset/train.jsonl",
         "eval":  "s3://edu-qlora-art/edu-lora-dataset/eval.jsonl"},
        wait=False,
    )
    print(f"Job: {estimator.latest_training_job.name}")
except Exception as e:
    import traceback
    traceback.print_exc()