"""
test_endpoint.py
----------------
Fires a few test questions at the deployed SageMaker endpoint
and prints the responses side by side.

Usage:
    python test_endpoint.py
"""

import boto3, json, os
from dotenv import load_dotenv

load_dotenv()

ENDPOINT_NAME = "edu-qlora-endpoint-20260601-193404"
REGION        = "us-east-1"

TEST_QUESTIONS = [
    {
        "question": "What is the main source of energy for most ecosystems on Earth?",
        "choices":  "A. Wind\nB. The sun\nC. Water\nD. Soil",
    },
    {
        "question": "What force pulls objects toward the center of the Earth?",
        "choices":  "A. Magnetism\nB. Friction\nC. Gravity\nD. Inertia",
    },
    {
        "question": "What do plants need to perform photosynthesis?",
        "choices":  "A. Oxygen and water\nB. Sunlight, water, and carbon dioxide\nC. Soil and nitrogen\nD. Heat and darkness",
    },
]

def build_prompt(q):
    return (
        f"Answer the following elementary science question by selecting the correct choice.\n\n"
        f"Question: {q['question']}\n\n"
        f"Choices:\n{q['choices']}"
    )

def main():
    runtime = boto3.client("sagemaker-runtime", region_name=REGION)

    print(f"Testing endpoint: {ENDPOINT_NAME}\n")
    print("=" * 60)

    for i, q in enumerate(TEST_QUESTIONS):
        prompt = build_prompt(q)
        payload = json.dumps({"inputs": prompt, "parameters": {"max_new_tokens": 64, "do_sample": False}})

        response = runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Body=payload,
        )

        result = json.loads(response["Body"].read())
        if isinstance(result, list):
            generated = result[0].get("generated_text", str(result))
        else:
            generated = str(result)

        # Strip the prompt from the output if echoed back
        if generated.startswith(prompt):
            generated = generated[len(prompt):].strip()

        print(f"Q{i+1}: {q['question']}")
        print(f"     {q['choices'].replace(chr(10), ' | ')}")
        print(f"→  {generated[:200]}")
        print()

    print("=" * 60)
    print("Done. Remember to delete the endpoint:")
    print(f"  aws sagemaker delete-endpoint --endpoint-name {ENDPOINT_NAME} --region {REGION}")

if __name__ == "__main__":
    main()
