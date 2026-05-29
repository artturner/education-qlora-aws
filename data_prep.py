"""
data_prep.py
------------
Downloads OpenBookQA + SCIQ, formats both into Alpaca-style instruction-following data, and saves the result as a JSONL file. 
Blends the two datasets together, with a 70:30 ratio of OpenBookQA to SCIQ examples.
Uploads train/eval splits to S3 for use in training and evaluation.

Run locally before spinning up any paid AWS compute.

Usage:
    python data_prep.py

Requirements:
    pip install datasets boto3 sagemaker
    
AWS credentials must be configured locally to allow uploading to S3.
    aws configure (or set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables)

"""

import json
import random
import boto3
from pathlib import Path
from datasets import load_dataset

# ---CONFIG - edit these as needed---
SE_Bucket = 'edu-qlora-art'  # S3 bucket to upload the prepared data to
SE_Prefix = 'edu-lora-dataset'  # S3 prefix (folder) to upload the prepared data to
AWS_REGION = "us-east-1"  # AWS region for S3
OUTPUT_DIR = Path('./data')  # Local directory to save prepared data before uploading
RANDOM_SEED = 42  # For reproducibility of the blended dataset
EVAL_SPLIT = 0.1  # Ratio of data to use for evaluation
# ---END CONFIG---

random.seed(RANDOM_SEED)

# ---Formatters---
def format_openbookqa(example):
    """Converts an OpenBookQA example into Alpaca-style instruction-following format."""
    choices = "\n".join([
        f"{label}. {text}"
        for label, text in zip(
            example['choices']['label'],
            example['choices']['text']
        )
    ])

    instruction = (
        f"Answer the following elementary science question based on selecting the correct choices.\n\n"
        f"Question: {example['question_stem']}\n\n"
        f"Choices:\n{choices}"
    )
    output = (
        f"The correct answer is {example['answerKey']}."
        f"{example['choices']['text'][example['choices']['label'].index(example['answerKey'])]}"
    )
    
    return {"instruction": instruction, "input": "", "output": output}

def format_sciq(example):
    """Convert SCIQ example to instruction/output format with distractor choices."""
    choices = [
        example["distractor1"],
        example["distractor2"],
        example["distractor3"],
        example["correct_answer"],   
    ]
    random.shuffle(choices)  # Shuffle choices to avoid position bias
    correct_letter = "ABCD"[choices.index(example['correct_answer'])]  # Get the letter corresponding to the correct answer
    choice_str = "\n".join([f"{letter}. {choice}" for letter, choice in zip("ABCD", choices)])

    # Include support passage if present
    support = f"\n\nContext: {example['support']}" if example.get('support') else ""

    instruction = (
        f"Answer the following science question by selecting the correct choice.{support}\n\n"
        f"Question: {example['question']}\n\n"
        f"Choices:\n{choice_str}"
    )
    output = (
        f"The correct answer is {correct_letter}."
        f"{example['correct_answer']}"
    )
    return {"instruction": instruction, "input": "", "output": output}

# ---Load & Format Datasets---

def load_and_format():
    # Load datasets
    print("\n-- Loading OpenBookQA ---")
    obqa = load_dataset("allenai/openbookqa", "main")
    obqa_all = []
    for split in ["train", "validation", "test"]:
        for ex in obqa[split]:
            obqa_all.append(format_openbookqa(ex))
    print(f"   OpenBookQA examples: {len(obqa_all)}")

    print("\n-- Loading SCIQ ---")

    sciq = load_dataset("allenai/sciq")
    sciq_all = []
    for split in ["train", "validation", "test"]:
        for ex in sciq[split]:
            sciq_all.append(format_sciq(ex))
    print(f"   SCIQ examples: {len(sciq_all)}")

    # Blend datasets (70% OBQA, 30% SCIQ). Sample SCXIQ to -43% of OBQA to maintain the ratio.
    target_sciq = int(len(obqa_all) * 0.43)  # 30% of total blended dataset
    sciq_sampled = random.sample(sciq_all, min(target_sciq, len(sciq_all)))
    print(f"   SCIQ sampled to {len(sciq_sampled)} examples to maintain 70:30 ratio with OpenBookQA.")
          
    combined = obqa_all + sciq_sampled
    random.shuffle(combined)  # Shuffle the combined dataset
    print(f"\nTotal blended dataset size: {len(combined)} (70% OpenBookQA, 30% SCIQ)")

    return combined

# ---Split & Save---

def split_and_save(examples):
    
    OUTPUT_DIR.mkdir(exist_ok=True)

    split_idx = int(len(examples) * (1 - EVAL_SPLIT))
    train_data = examples[:split_idx]
    eval_data = examples[split_idx:]
    
    train_path = OUTPUT_DIR / "train.jsonl"
    eval_path = OUTPUT_DIR / "eval.jsonl"\
    
    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")

    print(f"\n---Saved locally---")
    print(f"\nSaved {len(train_data)} training examples to {train_path}")
    print(f"Saved {len(eval_data)} evaluation examples to {eval_path}")

    return train_path, eval_path

# ---Upload to S3---

def ensure_bucket(s3_client, bucket, region):
    """Ensure the S3 bucket exists, create if it doesn't."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        print(f"S3 bucket '{bucket}' already exists: s3://{bucket}")
    except s3_client.exceptions.ClientError as e:
        error_code = e.response['Error']['Code']
        if error_code == '404':
            print(f"S3 bucket '{bucket}' does not exist. Creating...")
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket)
            else:
                s3_client.create_bucket(Bucket=bucket, CreateBucketConfiguration={'LocationConstraint': region})
        else:
            raise
            
def upload_to_s3(train_path, eval_path):
    s3 = boto3.client('s3', region_name=AWS_REGION)

    print(f"\n---Uploading to S3---")
    ensure_bucket(s3, SE_Bucket, AWS_REGION)
    
    s3_train = f"{SE_Prefix}/train.jsonl"
    s3_eval = f"{SE_Prefix}/eval.jsonl"

    s3.upload_file(str(train_path), SE_Bucket, s3_train)
    print(f"Uploaded training data to s3://{SE_Bucket}/{s3_train}")
    s3.upload_file(str(eval_path), SE_Bucket, s3_eval)
    print(f"Uploaded evaluation data to s3://{SE_Bucket}/{s3_eval}")

    train_url = f"s3://{SE_Bucket}/{s3_train}"
    eval_url = f"s3://{SE_Bucket}/{s3_eval}"

    return train_url, eval_url


#---Preview---

def preview(examples, n=2):
    print(f"\n---Previewing {n} examples---")
    for i, ex in enumerate(random.sample(examples,n)):
        print(f" [{i+1}] Instruction:\n{ex['instruction']}")
        print(f"Output:\n{ex['output']}")
        print("-" * 60)

# ---Main---

def main():
    print("=" * 65)
    print("Education-Domain QLoRa Data Prep")
    print("OpenBookQA + SCIQ + Alpaca JSONL to S3")
    print("=" * 65)

    examples = load_and_format()
    preview(examples)
    train_path, eval_path = split_and_save(examples)
    train_url, eval_url = upload_to_s3(train_path, eval_path)

    print("\n" + "=" * 65)
    print("Data Prep Complete. Paste these URLs into your training job script:")

    print(f"\n   TRAIN_PATH = \"{train_url}\"")
    print(f"   EVAL_PATH = \"{eval_url}\"")
    print("\n" + "=" * 65)
if __name__ == "__main__":
    main()
