import boto3
import os
import zipfile

def zip_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)
    print(f"✅ Zipped '{folder_path}' to '{output_zip}'")


def upload_file_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded '{file_path}' to 's3://{bucket}/{key}'")


def upload_folder(local_folder_path, bucket_name, s3_prefix=""):
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(local_folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_folder_path)
            s3_key = os.path.join(s3_prefix, relative_path)

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)

