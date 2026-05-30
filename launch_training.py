"""
launch_training.py
------------------
Launches a SageMaker training job from your local machine.
Reads HF_TOKEN from .env file and passes it to the training job.

Usage:
    python launch_training.py\
        
Requiements:
    pip install sagemaker boto3 python-dotenv


"""

import os
import sagemaker
from sagemaker.huggingface import HuggingFace
from sagemaker.experiments.run import Run
from dotenv import load_dotenv

load_dotenv()  # loads .env from current directory

# ---CONFIG - edit these as needed---
ROLE_ARN = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role" 
TRAIN_PATH = "s3://edu-qlora-art/edu-lora-dataset/train.jsonl"
EVAL_PATH = "s3://edu-qlora-art/edu-lora-dataset/eval.jsonl"
REGION = "us-east-1"
HF_TOKEN = os.environ["HF_TOKEN"]  # loaded from .env
# ---END CONFIG---

# ---Session setup---
def get_session():
    # boto3 session uses credentials from ~/.aws/credentials
    # configured via 'aws configure' = never stored in this file
    import boto3
    boto_session = boto3.Session(region_name=REGION)
    return sagemaker.Session(boto_session=boto_session)


# ---HuggingFace Estimator setup---

def get_estimator(session):
    # HuggingFace estimator is a SageMaker wrapper that pulls the
    # official HuggingFace deep learning container - PyTorch +
    # transformers pre-installed. requirements.txt adds peft/bitsandbytes/trl.
    #
    # use spot_instances=True requests AWS Spot capacity - up to 70% cheaper.
    # max_wait must be >= max_run; SageMaker waits this long for Spot.
    # source_dir="." uploads all files in the current directory to the
    # container, including train.py and requirements.txt
    return HuggingFace(
        entry_point="train.py",
        source_dir=".",
        instance_type="ml.g5.xlarge",
        instance_count=1,
        role=ROLE_ARN,
        transformers_version="4.56.2",
        pytorch_version="2.8.0",
        py_version="py312",
        use_spot_instances=True,
        max_wait=10800,   # max wait time for spot capacity (3 hours) 
        max_run=7200,     # max training time in seconds (2 hours)
        sagemaker_session=session,
        environment={"HF_TOKEN": HF_TOKEN}
    )

# ---MAIN---

def main():
    session = get_session()
    estimator = get_estimator(session)
    
    # SageMaker Experiments tracks hyperparameters and metrics
    # so runs are repoducible and comparable in the Studio UI.
    print("Launching Sagemaker Training Job...")
    print(f" Training data: {TRAIN_PATH}")
    print(f" Eval data: {EVAL_PATH}")
    print(f" Instance: ml.g5.xlarge (Spot)")

    with Run(
        experiment_name="edu-qlora-llama",
        run_name="llama-1b-openbookqa-sciq",
        sagemaker_session=session,
    ) as run:

        run.log_parameters({
            "base model": "Llama-3.2-1B-Instruct",
            "dataset": "OpenBookQA + SciQ",
            "lora_r": 16,
            "lora_alpha": 32,
            "epochs": 3,
            "batch_size": 4,
            "learning_rate": 2e-4,
            
        })

        estimator.fit(
            {
                "train": TRAIN_PATH,
                "eval": EVAL_PATH
            },
            wait=True,   # streams logs to your terminal while training runs
        )
        if estimator.latest_training_job:
            run.log_metric("training_job_id", estimator.latest_training_job.name)
            print("\nTraining Job Complete")
            print(f"Model artifact: {estimator.model_data}")
        else:
            print("Warning: no training job was started")

if __name__ == "__main__":
    main()

