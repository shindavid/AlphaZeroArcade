#!/usr/bin/env python3

import boto3
import os
from pathlib import Path


def upload_folder(folder_path, bucket_name, s3_prefix=""):
    s3 = boto3.client("s3")

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, folder_path)
            s3_key = os.path.join(s3_prefix, relative_path)

            print(f"Uploading {local_path} to s3://{bucket_name}/{s3_key}")
            s3.upload_file(local_path, bucket_name, s3_key)


if __name__ == "__main__":
    REPO_DIR = os.path.join(Path.home(), 'projects/AlphaZeroArcade')
    bucket_name = 'alphazeroarcade'
    upload_folder(os.path.join(REPO_DIR, 'benchmarks/mcts'), bucket_name, s3_prefix='mcts')