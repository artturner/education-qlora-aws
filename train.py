"""
train.py
--------
QLoRA fine-tuning of Phi-3-mini-4k-instruct on education domain Q&A data.

Usage:
    python train.py

Requirements:
    pip install peft bitsandbytes trl
"""

import os
import glob
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


# ── LoRA config ────────────────────────────────────────────────────────────────
def get_lora_config():
    # LoRA injects small trainable matrices into attention layers.
    # r=16: adapter rank — higher = more capacity, more VRAM.
    # lora_alpha=32: scaling factor (conventionally 2x rank).
    # target_modules: attention + MLP projection layers in Phi-3.
    # lora_dropout: regularization to prevent overfitting on small datasets.
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


# ── Model and tokenizer ────────────────────────────────────────────────────────
def load_model_and_tokenizer(model_id, hf_token=""):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=hf_token,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer


# ── Training arguments ─────────────────────────────────────────────────────────
def get_training_args(output_dir):
    return SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=True,
        seed=42,
        optim="adamw_torch",
        group_by_length=True,
        gradient_checkpointing=False,
        dataset_text_field="text",
        packing=True,
    )


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    import sys

    print("=== MAIN STARTED ===", flush=True)

    # model_id = "microsoft/Phi-3-mini-4k-instruct"
    model_id = "meta-llama/Llama-3.2-1B-Instruct"

    output_dir      = os.environ.get("SM_MODEL_DIR",     "/opt/ml/model")
    train_data_path = os.path.join(
        os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"),
        "train.jsonl",
    )
    eval_data_path = os.path.join(
        os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/input/data/eval"),
        "eval.jsonl",
    )
    hf_token = os.environ.get("HF_TOKEN", "")

    print(f"HF_TOKEN set:   {bool(hf_token)}", flush=True)
    print(f"Train path:     {train_data_path}", flush=True)
    print(f"Eval path:      {eval_data_path}", flush=True)
    print(f"Output dir:     {output_dir}", flush=True)

    lora_config = get_lora_config()

    try:
        print("Loading model...", flush=True)
        model, tokenizer = load_model_and_tokenizer(model_id, hf_token)
        print("Model loaded!", flush=True)
    except Exception as e:
        print(f"MODEL LOAD FAILED: {e}", flush=True)
        sys.exit(1)

    # Apply LoRA adapters once — do NOT also pass peft_config to SFTTrainer.
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    eval_dataset  = load_dataset("json", data_files=eval_data_path,  split="train")
    print(f"Train examples: {len(train_dataset)}", flush=True)
    print(f"Eval examples:  {len(eval_dataset)}",  flush=True)

    training_args = get_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
    )

    print("Starting training...", flush=True)
    trainer.train()
    print("Training complete!", flush=True)

    # Save only the LoRA adapter (~50 MB vs ~8 GB for the full model).
    # We merge with the base model at deployment time.
    adapter_path = os.path.join(output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}", flush=True)


if __name__ == "__main__":
    main()
