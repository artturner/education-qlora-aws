python3 << 'EOF'
import json
with open('/opt/conda/lib/python3.12/site-packages/sagemaker/core/image_uri_config/huggingface.json') as f:
    d = json.load(f)

def search(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if any(x in str(k).lower() for x in ["tag", "container", "2.8.0", "uri"]):
                print(f"{path}.{k} = {v}")
            search(v, f"{path}.{k}")

search(d)
EOF
