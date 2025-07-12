#!/usr/bin/env python3
from alphazero.logic.run_params import RunParams
from games.game_spec import GameSpec
from py.util.aws_util import upload_file_to_s3, zip_folder
from util.py_util import CustomHelpFormatter

import argparse
import os


def load_args():
    parser = argparse.ArgumentParser(formatter_class=CustomHelpFormatter)
    RunParams.add_args(parser)
    return parser.parse_args()


def main():
    args = load_args()

    assert args.game is not None, "Must specify --game"

    key = 'benchmarks/mcts'
    folder_path = os.path.join('/workspace/repo', key)
    bucket = 'alphazeroarcade'
    folder_to_upload = os.path.join(folder_path, args.game)
    zip_file_name = args.game + '.zip'
    output_zip = os.path.join(folder_path, zip_file_name)

    zip_folder(folder_to_upload, output_zip)
    upload_file_to_s3(output_zip, bucket, key + '/' + zip_file_name)


if __name__ == "__main__":
    main()

