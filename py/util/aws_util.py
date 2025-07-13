from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import boto3
import json
import os
import requests
import time
from urllib.parse import urlencode, urlparse, urlunparse
import zipfile


def zip_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)
    print(f"Zipped '{folder_path}' to '{output_zip}'")


def upload_file_to_s3(file_path, bucket, key):
    s3 = boto3.client('s3')
    s3.upload_file(file_path, bucket, key)
    print(f"Uploaded '{file_path}' to 's3://{bucket}/{key}'")


def generate_signed_url(url, private_key_path, key_pair_id, expire_minutes=10):
    parsed_url = urlparse(url)
    expire_time = int(time.time()) + expire_minutes * 60

    policy_dict = {
        "Statement": [
            {
                "Resource": url,
                "Condition": {
                    "DateLessThan": {
                        "AWS:EpochTime": expire_time
                    }
                }
            }
        ]
    }

    policy = json.dumps(policy_dict, separators=(',', ':'))

    # Load private key
    with open(private_key_path, 'rb') as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None,
        )

    # Sign the policy
    signature = private_key.sign(
        policy.encode('utf-8'),
        padding.PKCS1v15(),
        hashes.SHA1()
    )

    # Base64-url encode signature
    signature_encoded = base64.b64encode(signature).decode('utf-8')
    signature_encoded = signature_encoded.replace('+', '-').replace('=', '_').replace('/', '~')

    # Create query params
    query_params = {
        'Policy': base64.b64encode(policy.encode('utf-8')).decode('utf-8').replace('+', '-').replace('=', '_').replace('/', '~'),
        'Signature': signature_encoded,
        'Key-Pair-Id': key_pair_id
    }

    signed_url = urlunparse(parsed_url._replace(query=urlencode(query_params)))
    return signed_url


def download_file_from_s3_via_cloudfront(cloudfront_url, private_key_path, key_pair_id, destination_path):
    signed_url = generate_signed_url(cloudfront_url, private_key_path, key_pair_id)
    response = requests.get(signed_url)

    if response.status_code == 200:
        with open(destination_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ File downloaded to: {destination_path}")
    else:
        raise Exception(f"❌ Failed to download file: HTTP {response.status_code} - {response.text}")

if __name__ == "__main__":
    download_file_from_s3_via_cloudfront(
    cloudfront_url="https://download.alphazeroarcade.io/benchmarks/mcts/c4.zip",
    private_key_path="/workspace/aws-creds/cloudfront-private.pem",
    key_pair_id="KDHZFFYK6PB1L",
    destination_path="/workspace/c4.zip"
)

