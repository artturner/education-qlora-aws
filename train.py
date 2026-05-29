"""
train.py
------------
Usage:
    python train.py

Requirements:
    pip install peft bitsandbytes trl
    
"""

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_dataset

# ---Quantization config---
def get_bnb_config():
    # BitsAndBytes config enables QLoRA:
    # load_in_4bit loads the base model in 4-bit precision,
    # massively reducing VRAM usage so a 3.8B model fits on one A10G GPU.
    # nf4 (NormalFloat4) is the quantization type — better than int4 for LLMs.
    # double_quant quantizes the quantization constants themselves for extra savings.
    # compute_dtype stays in bfloat16 for stable training arithmetic.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

#---LoRA config---
def get_lora_config():
    # LoRA config: LoRA is a parameter-efficient fine-tuning method that adds small trainable matrices to the model.
    # r is the rank of the matrices, alpha is the scaling factor, and dropout is the dropout rate.
    # target_modules are the layers to apply LoRA to.
    # bias is set to none to avoid adding extra parameters.
    # task_type is set to CAUSAL_LM for language modeling.
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

#---Load model and Tokenizer---
def load_model_and_tokenizer(model_id, bnb_config):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # automatically places layers across available GPUs
        trust_remote_code=True, # required for Phi-3 custom modeling code
        torch_dtype=torch.bfloat16,
    )

    # prepare_model_for_kbit_training enables gradient checkpointing and 
    # casts layer norms to float32 - both required for stable QLoRA training.

    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Phi-3 has no native pad token - uses EOS token for padding, so we set it here.
    # padding_side="right" prevents attention mask warnings during training
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

# ---Training arguments---
def get_training_args():
    # TrainingArguments config: TrainingArguments is a class that contains all the hyperparameters for training.
    # These hyperparameters are tuned for a single A10G GPU (24 GB VRAM)
    # output_dir is the directory where the model checkpoints will be saved.
    # per_device_train_batch_size is the batch size per GPU.
    # gradient_accumulation_steps is the number of steps to accumulate gradients before updating the model.
    # gradient_accumulation_steps=2 simulates a batch size of 8 (4 x 2)
    # without storing 8 batches in VRAM simultaneously.
    # cosine lr_swcheduler wamrs up then decays smoothly - better than linear for LLM.
    # bf16=True uses bfloat16 precision; faster and more stable than fp16 on A10G.
    # optim is the optimizer to use.
    # group_by_length=True groups sequences of similar length together to reduce padding.

    return TrainingArguments(
        output_dir="output_dir",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        bf16=True,
        seed=42,
        optim="paged_adamw_32bit",
        group_by_length=True,
        gradient_checkpointing=True,
    )

#---Main---
def main():

    model_id = "meta-llama/Llama-3.2-1B-Instruct"
    output_dir = os.environ["SM_MODEL_DIR"] # Sagemaker injects this path

    # SM_CHANNEL_<NAME> env vars point to the S3 data SageMaker downloaded
    # into the container before training starts
    train_data_path = os.environ["SM_CHANNEL_TRAIN"]
    eval_data_path = os.environ["SM_CHANNEL_EVAL"]

    bnb_config = get_bnb_config()
    lora_config = get_lora_config()
    model, tokenizer = load_model_and_tokenizer(model_id, bnb_config)
    model = get_peft_model(model, lora_config)

    # Print trainable vs total parameters - good to log
    model.print_trainable_parameters()

    train_dataset = load_dataset("json", data_files=train_data_path, split="train")
    eval_dataset = load_dataset("json", data_files=eval_data_path, split="train")

    training_args = get_training_args(output_dir)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        peft_config=lora_config,
        max_seq_length=512,               # truncate longer examples to save VRAM
        dataset_text_field="instruction", # the field SFTTrainer trains on
        packing=True,                     # concatenates multiple short examples into single 512-token sequences
    )

    trainer.train()

    # Save only the LoRA adapter - not the full model
    # The Adapter is ~50 MB vs ~1.5 GB for the full model
    # We merge at deployment time in a separate step.
    adapter_path = os.path.join(output_dir, "lora_adapter")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

    if __name__ == "__main__":
        main()
