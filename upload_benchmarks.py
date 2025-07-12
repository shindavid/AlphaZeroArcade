#!/usr/bin/env python3
from py.util.aws_util import upload_folder

import os


if __name__ == "__main__":
    repo_dir = '/workspace/repo/'
    bucket_name = 'alphazeroarcade'
    folder_to_upload = os.path.join(repo_dir, 'benchmarks/mcts')
    s3_prefix = 'benchmarks/mcts'

    upload_folder(folder_to_upload, bucket_name, s3_prefix=s3_prefix)

