"""
dagster_pipeline.py
-------------------
Dagster asset pipeline orchestrating the full education-domain
LoRA fine-tuning workflow:

  prepared_dataset → trained_model → evaluation_report → deployed_endpoint

Each asset submits work to AWS (S3, SageMaker Training/Processing)
and materializes metadata for tracking. The deployed_endpoint asset
includes a quality gate — deployment is aborted if the fine-tuned
model does not outperform the base model on Exact Match.

Usage:
    dagster dev -f dagster_pipeline.py
    # Then open http://localhost:3000 and materialize assets
"""

import boto3, json, os, time, tarfile, tempfile
from datetime import datetime
from dotenv import load_dotenv
from dagster import (
    asset, AssetExecutionContext, define_asset_job,
    AssetSelection, Definitions, MetadataValue,
)

load_dotenv()

# ── CONFIG ────────────────────────────────────────────────────────────────────
ROLE_ARN   = "arn:aws:iam::730335300762:role/sagemaker_edu_qlora-role"
REGION     = "us-east-1"
BUCKET     = "sagemaker-us-east-1-730335300762"
DATA_BUCKET = "edu-qlora-art"
DATA_PREFIX = "edu-lora-dataset"
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
IMAGE_URI  = (
    "763104351884.dkr.ecr.us-east-1.amazonaws.com/"
    "huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04"
)
# ──────────────────────────────────────────────────────────────────────────────


def sm_client():
    return boto3.client("sagemaker", region_name=REGION)


def s3_client():
    return boto3.client("s3", region_name=REGION)


def wait_for_job(client, job_name: str, job_type: str, context: AssetExecutionContext):
    """Poll SageMaker until a Training or Processing job completes or fails."""
    describe = (
        client.describe_training_job
        if job_type == "training"
        else client.describe_processing_job
    )
    key = "TrainingJobStatus" if job_type == "training" else "ProcessingJobStatus"

    while True:
        response = describe(**{
            ("TrainingJobName" if job_type == "training" else "ProcessingJobName"): job_name
        })
        status = response[key]
        context.log.info(f"{job_type.capitalize()} job {job_name}: {status}")

        if status == "Completed":
            return response
        if status in ("Failed", "Stopped"):
            reason = response.get("FailureReason", "Unknown")
            raise RuntimeError(f"{job_type.capitalize()} job failed: {reason}")

        time.sleep(30)


# ── Asset 1: prepared_dataset ─────────────────────────────────────────────────

@asset
def prepared_dataset(context: AssetExecutionContext) -> dict:
    """
    Run data_prep.py logic: download OpenBookQA + SCIQ, format to Alpaca
    instruction style with combined text field, upload train/eval JSONL to S3.
    """
    from datasets import load_dataset
    import random, json
    from pathlib import Path

    random.seed(42)

    context.log.info("Loading OpenBookQA and SCIQ datasets ...")

    def format_openbookqa(example):
        choices = "\n".join([
            f"{label}. {text}"
            for label, text in zip(
                example["choices"]["label"],
                example["choices"]["text"]
            )
        ])
        instruction = (
            f"Answer the following elementary science question by selecting the correct choice.\n\n"
            f"Question: {example['question_stem']}\n\nChoices:\n{choices}"
        )
        output = (
            f"The correct answer is {example['answerKey']}."
            f"{example['choices']['text'][example['choices']['label'].index(example['answerKey'])]}"
        )
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "text": f"{instruction}\n\n{output}",
        }

    def format_sciq(example):
        choices = [
            example["distractor1"], example["distractor2"],
            example["distractor3"], example["correct_answer"],
        ]
        random.shuffle(choices)
        correct_letter = "ABCD"[choices.index(example["correct_answer"])]
        choice_str = "\n".join([f"{l}. {c}" for l, c in zip("ABCD", choices)])
        support = f"\n\nContext: {example['support']}" if example.get("support") else ""
        instruction = (
            f"Answer the following science question by selecting the correct choice.{support}\n\n"
            f"Question: {example['question']}\n\nChoices:\n{choice_str}"
        )
        output = (
            f"The correct answer is {correct_letter}."
            f"{example['correct_answer']}"
        )
        return {
            "instruction": instruction,
            "input": "",
            "output": output,
            "text": f"{instruction}\n\n{output}",
        }

    obqa = load_dataset("allenai/openbookqa", "main")
    obqa_all = [format_openbookqa(ex)
                for split in ["train", "validation", "test"]
                for ex in obqa[split]]

    sciq = load_dataset("allenai/sciq")
    sciq_all = [format_sciq(ex)
                for split in ["train", "validation", "test"]
                for ex in sciq[split]]

    target_sciq = int(len(obqa_all) * 0.43)
    sciq_sampled = random.sample(sciq_all, min(target_sciq, len(sciq_all)))

    combined = obqa_all + sciq_sampled
    random.shuffle(combined)

    split_idx  = int(len(combined) * 0.9)
    train_data = combined[:split_idx]
    eval_data  = combined[split_idx:]

    context.log.info(f"Train: {len(train_data)} | Eval: {len(eval_data)}")

    # Save locally then upload
    with tempfile.TemporaryDirectory() as tmp:
        train_path = os.path.join(tmp, "train.jsonl")
        eval_path  = os.path.join(tmp, "eval.jsonl")

        with open(train_path, "w") as f:
            for ex in train_data:
                f.write(json.dumps(ex) + "\n")
        with open(eval_path, "w") as f:
            for ex in eval_data:
                f.write(json.dumps(ex) + "\n")

        s3 = s3_client()
        s3.upload_file(train_path, DATA_BUCKET, f"{DATA_PREFIX}/train.jsonl")
        s3.upload_file(eval_path,  DATA_BUCKET, f"{DATA_PREFIX}/eval.jsonl")

    train_uri = f"s3://{DATA_BUCKET}/{DATA_PREFIX}/train.jsonl"
    eval_uri  = f"s3://{DATA_BUCKET}/{DATA_PREFIX}/eval.jsonl"

    context.add_output_metadata({
        "train_examples": MetadataValue.int(len(train_data)),
        "eval_examples":  MetadataValue.int(len(eval_data)),
        "train_s3_uri":   MetadataValue.text(train_uri),
        "eval_s3_uri":    MetadataValue.text(eval_uri),
    })

    return {"train_uri": train_uri, "eval_uri": eval_uri}


# ── Asset 2: trained_model ────────────────────────────────────────────────────

@asset
def trained_model(
    context: AssetExecutionContext,
    prepared_dataset: dict,
) -> dict:
    """
    Upload train.py to S3 and submit a SageMaker Training Job.
    Waits for completion and returns the model artifact S3 URI.
    """
    job_name = f"edu-qlora-dagster-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    context.log.info(f"Submitting training job: {job_name}")

    # Upload training script
    s3 = s3_client()
    for fname in ["train.py", "requirements.txt"]:
        s3.upload_file(fname, BUCKET, f"source/{job_name}/{fname}")
    source_uri = f"s3://{BUCKET}/source/{job_name}/"

    sm = sm_client()
    sm.create_training_job(
        TrainingJobName=job_name,
        AlgorithmSpecification={
            "TrainingImage":     IMAGE_URI,
            "TrainingInputMode": "File",
            "ContainerEntrypoint": [
                "/bin/bash", "-c",
                "pip install -r /opt/ml/input/data/code/requirements.txt && "
                "python -u /opt/ml/input/data/code/train.py 2>&1"
            ],
        },
        RoleArn=ROLE_ARN,
        InputDataConfig=[
            {
                "ChannelName": "train",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prepared_dataset["train_uri"],
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "ContentType": "application/json",
                "InputMode": "File",
            },
            {
                "ChannelName": "eval",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": prepared_dataset["eval_uri"],
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "ContentType": "application/json",
                "InputMode": "File",
            },
            {
                "ChannelName": "code",
                "DataSource": {"S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": source_uri,
                    "S3DataDistributionType": "FullyReplicated",
                }},
                "InputMode": "File",
            },
        ],
        OutputDataConfig={"S3OutputPath": f"s3://{BUCKET}/output/"},
        ResourceConfig={
            "InstanceType": "ml.g5.xlarge",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        StoppingCondition={
            "MaxRuntimeInSeconds": 7200,
            "MaxWaitTimeInSeconds": 10800,
        },
        EnableManagedSpotTraining=True,
        CheckpointConfig={"S3Uri": f"s3://{BUCKET}/checkpoints/{job_name}/"},
        Environment={
            "HF_TOKEN":               HF_TOKEN,
            "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
        },
    )

    response = wait_for_job(sm, job_name, "training", context)
    model_uri = f"s3://{BUCKET}/output/{job_name}/output/model.tar.gz"

    context.add_output_metadata({
        "job_name":       MetadataValue.text(job_name),
        "model_artifact": MetadataValue.text(model_uri),
        "train_loss":     MetadataValue.text(
            str(response.get("FinalMetricDataList", [{}])[0].get("Value", "N/A"))
        ),
    })

    return {"job_name": job_name, "model_uri": model_uri}


# ── Asset 3: evaluation_report ────────────────────────────────────────────────

@asset
def evaluation_report(
    context: AssetExecutionContext,
    trained_model: dict,
    prepared_dataset: dict,
) -> dict:
    """
    Submit a SageMaker Processing Job to run Exact Match evaluation.
    Waits for completion, downloads results, and applies quality gate.
    """
    job_name = f"edu-qlora-eval-dagster-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    context.log.info(f"Submitting eval job: {job_name}")

    # Upload eval script
    s3 = s3_client()
    s3.upload_file("eval.py", BUCKET, f"eval-source/{job_name}/eval.py")
    code_uri = f"s3://{BUCKET}/eval-source/{job_name}/"

    sm = sm_client()
    sm.create_processing_job(
        ProcessingJobName=job_name,
        ProcessingInputs=[
            {
                "InputName": "model",
                "S3Input": {
                    "S3Uri":                  trained_model["model_uri"],
                    "LocalPath":              "/opt/ml/processing/input/model",
                    "S3DataType":             "S3Prefix",
                    "S3InputMode":            "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "data",
                "S3Input": {
                    "S3Uri":                  prepared_dataset["eval_uri"],
                    "LocalPath":              "/opt/ml/processing/input/data",
                    "S3DataType":             "S3Prefix",
                    "S3InputMode":            "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
            {
                "InputName": "code",
                "S3Input": {
                    "S3Uri":                  code_uri,
                    "LocalPath":              "/opt/ml/processing/input/code",
                    "S3DataType":             "S3Prefix",
                    "S3InputMode":            "File",
                    "S3DataDistributionType": "FullyReplicated",
                },
            },
        ],
        ProcessingOutputConfig={
            "Outputs": [{
                "OutputName": "results",
                "S3Output": {
                    "S3Uri":        f"s3://{BUCKET}/eval-output/{job_name}/",
                    "LocalPath":    "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }]
        },
        ProcessingResources={
            "ClusterConfig": {
                "InstanceCount":  1,
                "InstanceType":   "ml.g5.xlarge",
                "VolumeSizeInGB": 30,
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

    wait_for_job(sm, job_name, "processing", context)

    # Download results
    results_key = f"eval-output/{job_name}/eval_results.json"
    obj = s3.get_object(Bucket=BUCKET, Key=results_key)
    metrics = json.loads(obj["Body"].read())

    base_em = metrics["base_em"]
    ft_em   = metrics["ft_em"]
    delta   = metrics["improvement"]

    context.log.info(f"Base EM: {base_em}% | FT EM: {ft_em}% | Delta: +{delta}%")

    # ── Quality gate ──────────────────────────────────────────────────────────
    if ft_em <= base_em:
        raise RuntimeError(
            f"Quality gate FAILED: fine-tuned EM ({ft_em}%) did not exceed "
            f"base EM ({base_em}%). Aborting deployment."
        )
    context.log.info(f"Quality gate PASSED: +{delta}% improvement.")

    context.add_output_metadata({
        "base_em":   MetadataValue.float(base_em),
        "ft_em":     MetadataValue.float(ft_em),
        "delta":     MetadataValue.float(delta),
        "results_s3": MetadataValue.text(f"s3://{BUCKET}/{results_key}"),
    })

    return {"base_em": base_em, "ft_em": ft_em, "delta": delta,
            "model_uri": trained_model["model_uri"]}


# ── Asset 4: deployed_endpoint ────────────────────────────────────────────────

@asset
def deployed_endpoint(
    context: AssetExecutionContext,
    evaluation_report: dict,
) -> dict:
    """
    Download model.tar.gz, merge LoRA adapter with base model,
    re-upload merged model, and create a SageMaker real-time endpoint.
    Skips deployment if quality gate already failed upstream.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch

    model_uri = evaluation_report["model_uri"]
    job_name  = f"edu-qlora-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    model_id  = "meta-llama/Llama-3.2-1B-Instruct"

    context.log.info(f"Downloading model artifact: {model_uri}")

    s3 = s3_client()
    bucket_name = model_uri.split("/")[2]
    key         = "/".join(model_uri.split("/")[3:])

    with tempfile.TemporaryDirectory() as tmp:
        tar_path     = os.path.join(tmp, "model.tar.gz")
        extract_dir  = os.path.join(tmp, "extracted")
        merged_dir   = os.path.join(tmp, "merged")
        os.makedirs(extract_dir, exist_ok=True)
        os.makedirs(merged_dir,  exist_ok=True)

        s3.download_file(bucket_name, key, tar_path)
        context.log.info("Extracting adapter ...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)

        adapter_path = os.path.join(extract_dir, "lora_adapter")
        context.log.info("Loading base model and merging adapter ...")

        base = AutoModelForCausalLM.from_pretrained(
            model_id, token=HF_TOKEN,
            torch_dtype=torch.bfloat16, device_map="cpu",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
        model_with_adapter = PeftModel.from_pretrained(base, adapter_path)
        merged = model_with_adapter.merge_and_unload()

        context.log.info("Saving merged model ...")
        merged.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

        # Package and upload
        merged_tar = os.path.join(tmp, "merged_model.tar.gz")
        with tarfile.open(merged_tar, "w:gz") as tar:
            tar.add(merged_dir, arcname=".")

        merged_key = f"merged-models/{job_name}/model.tar.gz"
        s3.upload_file(merged_tar, BUCKET, merged_key)
        merged_s3 = f"s3://{BUCKET}/{merged_key}"
        context.log.info(f"Merged model uploaded: {merged_s3}")

    # Create SageMaker endpoint
    sm = sm_client()

    model_name = f"edu-qlora-model-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image":        IMAGE_URI,
            "ModelDataUrl": merged_s3,
            "Environment": {
                "HF_TOKEN":               HF_TOKEN,
                "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
            },
        },
        ExecutionRoleArn=ROLE_ARN,
    )

    config_name = f"{model_name}-config"
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName":         "AllTraffic",
            "ModelName":           model_name,
            "InstanceType":        "ml.g5.xlarge",
            "InitialInstanceCount": 1,
        }],
    )

    endpoint_name = f"edu-qlora-endpoint-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name,
    )
    context.log.info(f"Endpoint creating: {endpoint_name} (takes ~5 min)")

    # Wait for endpoint
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name, WaiterConfig={"Delay": 30, "MaxAttempts": 20})
    context.log.info(f"Endpoint live: {endpoint_name}")

    context.add_output_metadata({
        "endpoint_name": MetadataValue.text(endpoint_name),
        "merged_model":  MetadataValue.text(merged_s3),
        "base_em":       MetadataValue.float(evaluation_report["base_em"]),
        "ft_em":         MetadataValue.float(evaluation_report["ft_em"]),
        "WARNING":       MetadataValue.text(
            "DELETE endpoint when done to avoid ongoing charges (~$1/hr)"
        ),
    })

    return {"endpoint_name": endpoint_name, "merged_model_s3": merged_s3}


# ── Job and Definitions ───────────────────────────────────────────────────────

edu_pipeline = define_asset_job(
    name="edu_finetune_pipeline",
    selection=AssetSelection.all(),
)

defs = Definitions(
    assets=[prepared_dataset, trained_model, evaluation_report, deployed_endpoint],
    jobs=[edu_pipeline],
)
