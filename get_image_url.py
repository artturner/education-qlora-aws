from sagemaker.huggingface import HuggingFace
import sagemaker, boto3
session = sagemaker.Session(boto_session=boto3.Session(region_name='us-east-1'))
est = HuggingFace(entry_point="train.py", instance_type="ml.g5.xlarge", instance_count=1, role="placeholder", transformers_version="4.56.2", pytorch_version="2.8.0", py_version="py312", sagemaker_session=session)
print(est.training_image_uri())
