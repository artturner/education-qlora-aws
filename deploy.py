"""
deploy.py
---------
Standalone script to create a SageMaker endpoint from the
already-merged model artifact produced by the Dagster pipeline.

The merged model is already in S3 — no need to re-merge.
Just creates the SageMaker Model, EndpointConfig, and Endpoint.

Usage:
    python deploy.py
"""

import boto3, os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROLE_ARN   = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role"
REGION     = "us-east-1"
BUCKET     = "sagemaker-us-east-1-730335300762"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")

# Already-merged model from the Dagster pipeline run
MERGED_MODEL_S3 = (
    "s3://sagemaker-us-east-1-730335300762/"
    "merged-models/edu-qlora-endpoint-20260601-175456/model.tar.gz"
)

# Inference image — NOT the training image
INFERENCE_IMAGE_URI = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-inference:2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04"
)
# ──────────────────────────────────────────────────────────────────────────────


def main():
    sm        = boto3.client("sagemaker", region_name=REGION)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    model_name  = f"edu-qlora-model-{timestamp}"
    config_name = f"edu-qlora-config-{timestamp}"
    ep_name     = f"edu-qlora-endpoint-{timestamp}"

    # ── Create Model ──────────────────────────────────────────────────────────
    print(f"Creating model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        INFERENCE_IMAGE_URI,
            "ModelDataUrl": MERGED_MODEL_S3,
            "Environment": {
                "HF_TASK":                "text-generation",
                "HF_TOKEN":               HF_TOKEN,
                "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
            },
        },
        ExecutionRoleArn=ROLE_ARN,
    )
    print("  Model created.")

    # ── Create Endpoint Config ────────────────────────────────────────────────
    print(f"Creating endpoint config: {config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":          "AllTraffic",
            "ModelName":            model_name,
            "InstanceType":         "ml.g5.xlarge",
            "InitialInstanceCount": 1,
        }],
    )
    print("  Endpoint config created.")

    # ── Create Endpoint ───────────────────────────────────────────────────────
    print(f"Creating endpoint: {ep_name}")
    sm.create_endpoint(
        EndpointName=ep_name,
        EndpointConfigName=config_name,
    )
    print("  Endpoint submitted — waiting for InService (~10 min) ...")

    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(
        EndpointName=ep_name,
        WaiterConfig={"Delay": 30, "MaxAttempts": 40},  # up to 20 min
    )

    print(f"\nEndpoint is live: {ep_name}")
    print(f"Invocation URL:")
    print(f"  https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/{ep_name}/invocations")
    print(f"\n*** DELETE endpoint when done — ~$1/hr billing ***")
    print(f"    aws sagemaker delete-endpoint --endpoint-name {ep_name} --region {REGION}")


if __name__ == "__main__":
    main()
