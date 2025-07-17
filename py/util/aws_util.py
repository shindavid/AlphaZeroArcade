from alphazero.servers.loop_control.base_dir import Workspace

import configparser
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
import base64
import boto3
from dataclasses import dataclass
import json
import os
import requests
import time
from urllib.parse import urlencode, urlparse, urlunparse
import zipfile


CREDENTIALS = os.path.join(Workspace.aws_dir, "credentials")
CONFIG = os.path.join(Workspace.aws_dir, "config")
PRIVATE_KEY = os.path.join(Workspace.aws_dir, "cloudfront-private.pem")
PROFILE = 'default'

@dataclass
class Bucket:
    name: str
    cloudfront_url: str
    key_pair_id: str

    def upload_file_to_s3(self, file_path: str, key: str):
        cred = configparser.ConfigParser()
        cred.read(CREDENTIALS)
        aws_access_key_id = cred[PROFILE]['aws_access_key_id']
        aws_secret_access_key = cred[PROFILE]['aws_secret_access_key'] 
        
        conf = configparser.ConfigParser()
        conf.read(CONFIG)
        region = conf[PROFILE]['region'] 
        
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )
        s3 = session.client('s3')
        
        s3.upload_file(file_path, self.name, key)
        print(f"Uploaded '{file_path}' to 's3://{self.name}/{key}'")

    def generate_signed_url(self, key: str, expire_minutes=10):
        parsed_url = urlparse(os.path.join(self.cloudfront_url, key))
        expire_time = int(time.time()) + expire_minutes * 60

        policy_dict = {
            "Statement": [
                {
                    "Resource": self.cloudfront_url,
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
        with open(PRIVATE_KEY, 'rb') as key_file:
            private_key = serialization.load_pem_private_key(key_file.read(), password=None)

        # Sign the policy
        signature = private_key.sign(policy.encode('utf-8'), padding.PKCS1v15(), hashes.SHA1())

        # Base64-url encode signature
        signature_encoded = base64.b64encode(signature).decode('utf-8')
        signature_encoded = signature_encoded.replace('+', '-').replace('=', '_').replace('/', '~')

        # Create query params
        query_params = {
            'Policy': base64.b64encode(policy.encode('utf-8')).decode('utf-8').replace('+', '-').replace('=', '_').replace('/', '~'),
            'Signature': signature_encoded,
            'Key-Pair-Id': bucket.key_pair_id
        }

        signed_url = urlunparse(parsed_url._replace(query=urlencode(query_params)))
        return signed_url

    def download_from_s3(self, key: str, destination_path: str):
        signed_url = self.generate_signed_url(key)
        response = requests.get(signed_url)

        if response.status_code == 200:
            with open(destination_path, 'wb') as f:
                f.write(response.content)
            print(f"✅ File downloaded to: {destination_path}")
        else:
            raise Exception(f"❌ Failed to download file: HTTP {response.status_code} - {response.text}")


BUCKET = Bucket(name='alphazeroarcade',
                cloudfront_url='https://download.alphazeroarcade.io',
                key_pair_id='KDHZFFYK6PB1L')


def zip_folder(folder_path, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, folder_path)
                zipf.write(full_path, arcname=rel_path)
    print(f"Zipped '{folder_path}' to '{output_zip}'")


