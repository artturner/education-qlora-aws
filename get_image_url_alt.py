from sagemaker import image_uris
uri = image_uris.retrieve(
    framework="huggingface",
    region="us-east-1",
    version="4.56.2",
    py_version="py312",
    instance_type="ml.g5.xlarge",
    image_scope="training",
    base_framework_version="pytorch2.8.0"
)
print(uri)
