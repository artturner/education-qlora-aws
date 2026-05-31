"""
eval.py
-------
SageMaker Processing Job script.
Computes Exact Match on 100 eval examples:
  - Base Llama 3.2 1B (no fine-tuning)
  - Fine-tuned Llama 3.2 1B (LoRA adapter from training)

Inputs  (injected by SageMaker Processing):
  /opt/ml/processing/input/model/  — contains model.tar.gz
  /opt/ml/processing/input/data/   — contains eval.jsonl
  /opt/ml/processing/input/code/   — contains this script

Output:
  /opt/ml/processing/output/eval_results.json
"""

import os, json, tarfile, torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_INPUT = "/opt/ml/processing/input/model"
DATA_INPUT  = "/opt/ml/processing/input/data"
OUTPUT_DIR  = "/opt/ml/processing/output"

MODEL_ID        = "meta-llama/Llama-3.2-1B-Instruct"
HF_TOKEN        = os.environ.get("HF_TOKEN", "")
EXTRACT_DIR     = "/tmp/model_extracted"
N_EVAL_SAMPLES  = 100


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def extract_adapter():
    """Extract model.tar.gz and return path to lora_adapter folder."""
    tar_path = os.path.join(MODEL_INPUT, "model.tar.gz")
    print(f"Extracting {tar_path} ...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(EXTRACT_DIR)
    adapter_path = os.path.join(EXTRACT_DIR, "lora_adapter")
    print(f"Adapter path: {adapter_path}", flush=True)
    print(f"Adapter contents: {os.listdir(adapter_path)}", flush=True)
    return adapter_path


def load_base_model():
    print("Loading base Llama 3.2 1B ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        token=HF_TOKEN,
        quantization_config=get_bnb_config(),
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, token=HF_TOKEN, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    print("Base model loaded.", flush=True)
    return model, tokenizer


def generate(model, tokenizer, prompt, max_new_tokens=64):
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=512
    ).to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
    ).strip()

def extract_answer_letter(text):
    # Match "correct answer is X" or "answer is X" pattern first
    match = re.search(r'(?:correct\s+)?answer\s+is\s+([A-D])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    # Fallback: standalone letter followed by period or space
    match = re.search(r'\b([A-D])[.\s]', text)
    if match:
        return match.group(1).upper()
    return ""


def run_inference(model, tokenizer, samples, label):
    print(f"Running {label} inference on {len(samples)} examples ...", flush=True)
    predictions, correct = [], 0
    for i, ex in enumerate(samples):
        prompt = ex["instruction"]
        gold   = ex["output"]
        pred   = generate(model, tokenizer, prompt)
        gold_letter = extract_answer_letter(gold)
        pred_letter = extract_answer_letter(pred)
        ok = gold_letter != "" and gold_letter == pred_letter
        correct += int(ok)
        predictions.append({
            "idx":         i,
            "prompt":      prompt[:300],
            "gold":        gold,
            "gold_letter": gold_letter,
            "pred":        pred,
            "pred_letter": pred_letter,
            "correct":     ok,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{label}] {i+1}/{len(samples)} done", flush=True)

    em = correct / len(samples) * 100
    print(f"{label} Exact Match: {correct}/{len(samples)} = {em:.1f}%", flush=True)
    return predictions, em


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== EVAL STARTED ===", flush=True)

    # Load eval dataset
    eval_path = os.path.join(DATA_INPUT, "eval.jsonl")
    dataset   = load_dataset("json", data_files=eval_path, split="train")
    samples   = list(dataset.select(range(min(N_EVAL_SAMPLES, len(dataset)))))
    print(f"Loaded {len(samples)} eval examples.", flush=True)

    # ── Base model evaluation ──────────────────────────────────────────────
    base_model, tokenizer = load_base_model()
    base_preds, base_em   = run_inference(base_model, tokenizer, samples, "BASE")

    # ── Fine-tuned model evaluation ────────────────────────────────────────
    adapter_path = extract_adapter()
    print("Loading LoRA adapter onto base model ...", flush=True)
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()
    ft_preds, ft_em = run_inference(ft_model, tokenizer, samples, "FINE-TUNED")

    # ── Merge results ──────────────────────────────────────────────────────
    results = []
    for b, f in zip(base_preds, ft_preds):
        results.append({**b,
                        "ft_pred":   f["pred"],
                        "ft_letter": f["pred_letter"],
                        "ft_correct": f["correct"]})

    summary = {
        "base_em":      base_em,
        "ft_em":        ft_em,
        "improvement":  round(ft_em - base_em, 1),
        "n_samples":    len(samples),
        "base_correct": sum(1 for r in results if r["correct"]),
        "ft_correct":   sum(1 for r in results if r["ft_correct"]),
        "results":      results,
    }

    # ── Save ──────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nResults saved to {out_path}", flush=True)
    print(f"\n{'='*50}", flush=True)
    print(f"  Base EM:       {base_em:.1f}%", flush=True)
    print(f"  Fine-tuned EM: {ft_em:.1f}%", flush=True)
    print(f"  Improvement:   +{ft_em - base_em:.1f}%", flush=True)
    print(f"{'='*50}", flush=True)
    print("=== EVAL COMPLETE ===", flush=True)


if __name__ == "__main__":
    main()
