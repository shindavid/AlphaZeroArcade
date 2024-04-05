"""
Called from Model.save_model() in net_modules.py

Calling torch.jit.trace(...).save(...) directly in net_modules.py results in a memory leak. Hence
this workaround.

See: https://github.com/pytorch/pytorch/issues/35600
"""

import pickle
import sys
import torch

pickle_filename = sys.argv[1]
output_filename = sys.argv[2]
with open(pickle_filename, 'rb') as f:
    clone, example_input = pickle.load(f)

torch.jit.trace(clone, example_input).save(output_filename)
