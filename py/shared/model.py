from shared.basic_types import ShapeInfoDict
from shared.model_config import ModelConfig
from shared.net_modules import Head
from util.graph_util import AdjMatrix, topological_sort

import numpy as np
import onnx
import torch
from torch import nn as nn

import hashlib
import io
import logging
import os
from typing import Any, Dict, List, Set


logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()

        self._config = config
        self._n = len(config.parts)
        self._module_dict = nn.ModuleDict({k: v.to_module() for k, v in config.parts.items()})
        self._module_list = list(self._module_dict.values())
        self._parent_indices = self._compute_parent_indices()
        self._adj_matrix = self._compute_adj_matrix()
        self._dag_indices = topological_sort(self._adj_matrix)
        self._head_indices = [i for i, v in enumerate(self._module_list) if isinstance(v, Head)]
        self._head_names = [k for k, v in self._module_dict.items() if isinstance(v, Head)]

        self._clone_dict = {}
        self._validate()
        self._model_architecture_signature = self._compute_model_architecture_signature()

    def _compute_parent_indices(self) -> List[List[int]]:
        parent_indices = [list() for _ in range(self._n)]
        inv_module_dict = {k: i for i, k in enumerate(self._module_dict.keys())}
        for i, c in enumerate(self._config.parts.values()):
            for parent in c.parents:
                j = inv_module_dict[parent]
                parent_indices[i].append(j)
        return parent_indices

    def _compute_adj_matrix(self) -> AdjMatrix:
        adj_matrix: AdjMatrix = np.zeros((self._n, self._n), dtype=bool)

        for i, ps in enumerate(self._parent_indices):
            for p in ps:
                adj_matrix[p, i] = True

        return adj_matrix

    def _compute_model_architecture_signature(self):
        s = str(self) + '\n\n' + str(self._dag_indices)
        logger.debug('Computing model architecture signature: %s', s)
        return hashlib.md5(s.encode()).hexdigest()

    def _validate(self):
        head_names = set()
        for head in self._head_names:
            assert head not in head_names, f'Head with name {head} already exists'
            head_names.add(head)

        # TODO: rm these asserts here, and instead do dynamic index assignment in the c++
        assert self._head_names[0] == 'policy', 'The first head must be policy'
        assert self._head_names[1] == 'value', 'The second head must be value'
        assert self._head_names[2] == 'action_value', 'The third head must be action_value'

    @property
    def heads(self) -> List[nn.Module]:
        return [self._module_list[i] for i in self._head_indices]

    @property
    def head_names(self) -> List[str]:
        return self._head_names

    def get_head(self, name: str) -> Head:
        assert name in self._module_dict, f'No module named {name}'
        head = self._module_dict[name]
        assert isinstance(head, Head), f'Module {name} is not a Head'
        return head

    def get_parameter_counts(self) -> Dict[str, int]:
        """
        Returns a dictionary mapping module names to the number of parameters in that module.
        """
        return {k: sum(p.numel() for p in v.parameters()) for k, v in self._module_dict.items()}

    def forward(self, x):
        y = [None] * self._n
        for i in self._dag_indices:
            parents = self._parent_indices[i]
            if not parents:
                y[i] = self._module_list[i](x)
            else:
                args = [y[p] for p in parents]
                y[i] = self._module_list[i](*args)

        return tuple(y[i] for i in self._head_indices)

    def save_model(self, filename: str, input_shape_info_dict: ShapeInfoDict, head_names: Set[str]):
        """
        Saves this network to disk in ONNX format.
        """
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 1) clone, strip extra heads, freeze
        clone_dict_key = frozenset(head_names)
        clone = self._clone_dict.get(clone_dict_key, None)
        if clone is None:
            trimmed_config = self._config.trim(head_names)
            clone = Model(trimmed_config)
            self._clone_dict[clone_dict_key] = clone
        clone.load_state_dict(self.state_dict(), strict=False)
        clone.cpu().eval()

        input_names = ["input"]
        output_names = clone._head_names
        dynamic_axes = {k:{0: "batch"} for k in input_names + output_names}

        # 2) make an example‐input and ONNX‐export it
        batch_size = 1
        example_shape = (batch_size, *input_shape_info_dict['input'].shape)
        example_input = torch.zeros(example_shape, dtype=torch.float32)

        # 3) Export to a temporary in-memory buffer
        buf = io.BytesIO()
        torch.onnx.export(
            clone, example_input, buf,
            export_params=True,
            opset_version=18,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

        # 4) Add metadata
        model = onnx.load_from_string(buf.getvalue())
        kv = model.metadata_props.add()
        kv.key = 'model-architecture-signature'
        kv.value = clone._model_architecture_signature

        onnx.save(model, filename)

    @staticmethod
    def load_from_checkpoint(checkpoint: Dict[str, Any]) -> 'Model':
        """
        Load a model from a checkpoint. Inverse of add_to_checkpoint().
        """
        model_state_dict = checkpoint['model.state_dict']
        config = checkpoint['model.config']
        model = Model(config)
        model.load_state_dict(model_state_dict)
        return model

    def add_to_checkpoint(self, checkpoint: Dict[str, Any]):
        """
        Save the current state of this neural net to a checkpoint, so that it can be loaded later
        via load_from_checkpoint().
        """
        checkpoint.update({
            'model.state_dict': self.state_dict(),
            'model.config': self._config,
        })
