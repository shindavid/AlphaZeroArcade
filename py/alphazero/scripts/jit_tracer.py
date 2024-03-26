"""
Called from Model.save_model() in net_modules.py
"""

import pickle
import sys
import torch

pickle_filename = sys.argv[1]
output_filename = sys.argv[2]
with open(pickle_filename, 'rb') as f:
    clone, example_input = pickle.load(f)

torch.jit.trace(clone, example_input).save(output_filename)
