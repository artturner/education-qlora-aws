# Education-Domain LoRA Fine-Tuning on AWS

End-to-end MLOps pipeline for fine-tuning Llama 3.2 1B Instruct on K-12 science
education Q&A data, with full AWS infrastructure, Dagster orchestration, and a live
SageMaker real-time inference endpoint.

**Results:** 54% → 76% Exact Match (+22pp) on held-out science questions.  
**LLM-as-Judge:** Overall score 2.20 → 3.27 / 5 (Correctness, Pedagogy, Conciseness).  
**Cost:** Full pipeline including failed iterations — under $6.

---

## Architecture Overview

```
data_prep.py          →  S3 (train/eval JSONL)
      ↓
train.py              →  SageMaker Training Job (ml.g5.xlarge Spot)
      ↓                   LoRA fine-tuning, adapter saved to S3
eval.py               →  SageMaker Processing Job (ml.g5.xlarge Spot)
      ↓                   Exact Match: base vs fine-tuned
llm_judge.py          →  Local (Anthropic API)
      ↓                   Correctness / Pedagogy / Conciseness scoring
deploy.py             →  SageMaker Real-Time Endpoint
      ↓                   Merged model, HuggingFace inference container
test_endpoint.py      →  Live inference validation

dagster_pipeline.py   →  Orchestrates all of the above as Dagster assets
```

---

## Repository Structure

| File | Purpose |
|------|---------|
| `data_prep.py` | Download OpenBookQA + SCIQ, format to Alpaca instruction style, upload to S3 |
| `train.py` | LoRA fine-tuning script — runs inside SageMaker Training container |
| `direct_launch.py` | Submit SageMaker Training Job via direct boto3 (bypasses SDK) |
| `eval.py` | Exact Match evaluation script — runs inside SageMaker Processing container |
| `run_eval.py` | Submit SageMaker Processing Job for evaluation |
| `llm_judge.py` | LLM-as-Judge scoring via Claude API, pulls results from S3 |
| `deploy.py` | Merge LoRA adapter, create SageMaker endpoint |
| `test_endpoint.py` | Send test questions to live endpoint and print responses |
| `dagster_pipeline.py` | Full Dagster asset pipeline — runs everything end-to-end |
| `requirements.txt` | Python dependencies for the training container |
| `boto_API_diagnostic.py` | Diagnostic utility for SageMaker API debugging |

---

## Model & Training Details

| Parameter | Value |
|-----------|-------|
| Base model | `meta-llama/Llama-3.2-1B-Instruct` |
| Method | LoRA (bfloat16, no quantization) |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target modules | q, k, v, o, gate, up, down projections |
| Trainable parameters | 8,912,896 (0.23% of 3.8B total) |
| Training epochs | 3 |
| Effective batch size | 8 (4 × gradient accumulation 2) |
| Learning rate | 2e-4 (cosine schedule) |
| Optimizer | adamw_torch |
| Instance | ml.g5.xlarge Spot (NVIDIA A10G 24 GB) |
| Training time | ~7.5 minutes |

### Why bfloat16 LoRA instead of QLoRA?

Llama 3.2 1B fits comfortably in 24 GB VRAM at bfloat16 (~2.5 GB). Applying 4-bit
quantization to this model caused a silent gradient blocking interaction between
`paged_adamw_32bit` and the quantized backward pass, causing `lora_B` matrices to
remain at zero throughout training. QLoRA is the right choice for 7B+ models where
memory is genuinely constrained. For 1B models, standard bfloat16 LoRA is faster,
simpler, and more reliable.

---

## Dataset

| Source | Examples | Split |
|--------|----------|-------|
| OpenBookQA | ~5,957 | 70% of blend |
| SCIQ (sampled) | ~2,561 | 30% of blend |
| **Total train** | **7,666** | 90% |
| **Total eval** | **852** | 10% |

Both datasets are formatted to Alpaca instruction style with a combined `text` field:

```
Answer the following elementary science question by selecting the correct choice.

Question: {question}

Choices:
A. {choice_a}
B. {choice_b}
C. {choice_c}
D. {choice_d}

The correct answer is {letter}.{correct_answer_text}
```

> **Important:** The `text` field must contain the full instruction + output sequence.
> Training only on `instruction` teaches the model to generate questions, not answers.

---

## AWS Infrastructure

### Prerequisites

- AWS account with SageMaker access
- IAM role with `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`
- Service quotas approved (check before running):
  - `ml.g5.xlarge for training job usage` — 1
  - `ml.g5.xlarge for training job usage (Spot)` — 1
  - `ml.g5.xlarge for processing job usage` — 1
- HuggingFace account with Llama 3.2 license accepted

### S3 Buckets

| Bucket | Purpose |
|--------|---------|
| `edu-qlora-art` | Training and evaluation data |
| `sagemaker-us-east-1-{account_id}` | Model artifacts, checkpoints, eval output |

### Container Images

| Use | Image |
|-----|-------|
| Training | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training:2.8.0-transformers4.56.2-gpu-py312-cu129-ubuntu22.04` |
| Inference | `763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-inference:2.6.0-transformers5.5.3-gpu-py312-cu124-ubuntu22.04` |

> The training image does not support inference (`docker run <image> serve`). Always
> use a separate inference image for SageMaker endpoints.

### Why direct boto3 instead of the SageMaker Python SDK?

The SageMaker Python SDK's `HuggingFace.fit()` silently failed without submitting a
training job under Python 3.13 (the SDK assumed older Python internals). This was
diagnosed via CloudTrail audit logs. The fix was to call `boto3.client("sagemaker").create_training_job()` directly, bypassing the SDK entirely. All launcher scripts use
this pattern.

---

## Quickstart

### 1. Clone and set up environment

```bash
git clone https://github.com/artturner/education-qlora-aws.git
cd education-qlora-aws
python3 -m venv .venv
source .venv/bin/activate
pip install dagster dagster-webserver boto3 python-dotenv anthropic datasets \
            transformers peft torch awscli sagemaker
```

### 2. Configure credentials

```bash
aws configure   # enter edu-qlora-dev credentials
```

Create `.env` in the project root:

```
HF_TOKEN=hf_xxxxxxxxxxxx
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxx
```

### 3. Update configuration

Edit `dagster_pipeline.py` (or individual scripts) and set:

```python
ROLE_ARN = "arn:aws:iam::<your-account-id>:role/sagemaker_edu_qlora-role"
```

### 4. Run the full pipeline with Dagster

```bash
dagster dev -f dagster_pipeline.py
```

Open `http://localhost:3000`, select `edu_finetune_pipeline`, and click **Materialize**.

The pipeline runs four assets in dependency order:

1. **`prepared_dataset`** — formats and uploads data to S3
2. **`trained_model`** — submits and monitors SageMaker Training Job
3. **`evaluation_report`** — submits and monitors SageMaker Processing Job; applies
   quality gate (aborts if fine-tuned EM ≤ base EM)
4. **`deployed_endpoint`** — merges adapter locally, uploads merged model,
   creates SageMaker real-time endpoint

> **Cost warning:** The endpoint bills at ~$1/hr. Delete it immediately after testing.

### 5. Run steps individually

```bash
# Data prep (run locally, no GPU needed)
python data_prep.py

# Submit training job
python direct_launch.py

# Submit evaluation job
python run_eval.py

# LLM-as-Judge (local, requires ANTHROPIC_API_KEY)
python llm_judge.py

# Deploy endpoint
python deploy.py

# Test endpoint
python test_endpoint.py

# Delete endpoint when done
aws sagemaker delete-endpoint \
    --endpoint-name <endpoint-name> \
    --region us-east-1
```

---

## Evaluation Results

### Exact Match (100 examples)

| Model | Score |
|-------|-------|
| Base Llama 3.2 1B | 54.0% |
| Fine-tuned adapter | **76.0%** |
| Improvement | **+22.0 pp (+40.7% relative)** |

| Category | Count |
|----------|-------|
| FT only correct | 29 |
| Base only correct | 7 |
| Both correct | 47 |
| Both wrong | 17 |

### LLM-as-Judge (20 examples, Claude evaluator)

| Dimension | Base | Fine-tuned | Delta |
|-----------|------|------------|-------|
| Correctness | 2.15 | 3.60 | +1.45 |
| Pedagogy | 2.15 | 2.40 | +0.25 |
| Conciseness | 2.30 | 3.80 | +1.50 |
| **Overall** | **2.20** | **3.27** | **+1.07** |

**Note on pedagogy:** Both models score low because the training format
(`"The correct answer is X."`) is concise but non-explanatory. A future iteration
should train on chain-of-thought outputs to improve pedagogical quality.

### Live Endpoint Test (3 questions)

| Question | Correct? |
|----------|----------|
| Main energy source for ecosystems? | ✓ |
| Force pulling objects to Earth? | ✓ |
| What do plants need for photosynthesis? | ✓ |

---

## Key Lessons Learned

**QLoRA vs LoRA for small models** — 4-bit quantization broke gradient flow silently
on a 1B model. For models ≤ 3B that fit in bfloat16, skip quantization.

**Evaluation metric bugs can be invisible** — A naive character scan returned 'C' from
the word "correct" before finding the actual answer letter, collapsing EM to 24% for
both models. Always verify your metric on a few examples manually before trusting it.

**Training target matters** — `dataset_text_field="instruction"` trained the model to
predict questions, not answers. The field must reference a combined instruction + output
sequence.

**Loss curves lie without gradient monitoring** — Two training runs showed decreasing
loss while gradients were entirely blocked, due to cosine warmup schedule effects on
the loss calculation independent of weight updates.

**Training vs inference containers** — SageMaker training images do not expose a
`serve` endpoint. Endpoint deployment requires a separate inference image.

**Pin your library versions** — TRL 0.23.0 renamed `evaluation_strategy`, `tokenizer`,
`max_seq_length`, and changed the `SFTConfig`/`TrainingArguments` split. Version
mismatches caused multiple failed runs that would have been avoided with pinned deps.

---

## Future Work

- **Explanatory outputs** — retrain with chain-of-thought format to improve pedagogy scores
- **Question generation** — use fine-tuned model as SME-assisted question bank generator
- **Smaller base model** — demonstrate dramatic improvement on Qwen2.5-0.5B or SmolLM
- **Larger dataset** — add ARC, MMLU-elementary, and SciQ for broader coverage
- **RLHF** — apply DPO or PPO with teacher feedback to further improve correctness

---

## AWS Certifications

This project was built by Art Turner, AWS certified across seven credentials:
Generative AI Developer – Professional, Machine Learning – Specialty,
Machine Learning Engineer – Associate, AI Practitioner,
Data Engineer – Associate, Solutions Architect – Associate, and Cloud Practitioner.

---

## License

MIT License — see `LICENSE` for details.
