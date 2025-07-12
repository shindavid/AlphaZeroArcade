import boto3
import os


def upload_folder(local_folder_path, bucket_name, s3_prefix=""):
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path)

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)

