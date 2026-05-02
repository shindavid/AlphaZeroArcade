from shared.basic_types import ShapeInfoCollection
from shared.model_config import ModelConfig
from shared.net_modules import Head
from util.graph_util import AdjMatrix, topological_sort
from util.logging_util import mute_everything

import numpy as np
import onnx
import torch
from torch import nn as nn

import hashlib
import io
import logging
import os
from typing import Any, Dict, List, Optional


logger = logging.getLogger(__name__)


class Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Model, self).__init__()

        self._config = config
        self._n = len(config.parts)
        self._module_dict = nn.ModuleDict({k: v.to_module() for k, v in config.parts.items()})
        self._module_list = list(self._module_dict.values())
        self._external_inputs: List[str] = self._compute_external_inputs()
        self._parent_indices = self._compute_parent_indices()
        self._adj_matrix = self._compute_adj_matrix()
        self._dag_indices = topological_sort(self._adj_matrix)
        self._head_indices = [i for i, v in enumerate(self._module_list) if isinstance(v, Head)]
        self._head_names = [k for k, v in self._module_dict.items() if isinstance(v, Head)]

        self._clone_dict = {}
        self._validate()
        self._model_architecture_signature = self._compute_model_architecture_signature()

    def _compute_external_inputs(self) -> List[str]:
        """
        Returns the list of external Model input names referenced by `parents` of any spec but
        not themselves present in `config.parts`. A spec with empty `parents` defaults to
        `['input']` (so a model with all parents-less stem still has 'input' as an external).
        Order is first-seen-in-parts-iteration to keep ONNX export deterministic.
        """
        seen = set()
        ordered: List[str] = []
        parts = self._config.parts
        for spec in parts.values():
            effective_parents = spec.parents if spec.parents else ['input']
            for p in effective_parents:
                if p not in parts and p not in seen:
                    seen.add(p)
                    ordered.append(p)
        return ordered

    def _compute_parent_indices(self) -> List[List[int]]:
        """
        Returns parent index lists for each module. Indices [0, n) refer to module outputs;
        indices [n, n+m) refer to external inputs (in self._external_inputs order).
        """
        parts = self._config.parts
        name_to_index: Dict[str, int] = {name: i for i, name in enumerate(parts)}
        for j, ext_name in enumerate(self._external_inputs):
            name_to_index[ext_name] = self._n + j

        parent_indices: List[List[int]] = []
        for spec in parts.values():
            effective_parents = spec.parents if spec.parents else ['input']
            parent_indices.append([name_to_index[p] for p in effective_parents])
        return parent_indices

    def _compute_adj_matrix(self) -> AdjMatrix:
        # Module-only adjacency; external inputs are roots not in the topo-sort.
        adj_matrix: AdjMatrix = np.zeros((self._n, self._n), dtype=bool)
        for i, ps in enumerate(self._parent_indices):
            for p in ps:
                if p < self._n:
                    adj_matrix[p, i] = True
        return adj_matrix

    def _compute_model_architecture_signature(self):
        components = [self, self._dag_indices, torch.__version__, onnx.__version__]
        s = '\n'.join(str(c) for c in components)
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

    def forward(self, *inputs):
        assert len(inputs) == len(self._external_inputs), (
            f'expected {len(self._external_inputs)} positional inputs '
            f'({self._external_inputs}), got {len(inputs)}')
        n = self._n
        y: List[Any] = [None] * (n + len(self._external_inputs))
        for j, val in enumerate(inputs):
            y[n + j] = val
        for i in self._dag_indices:
            parents = self._parent_indices[i]
            args = [y[p] for p in parents]
            y[i] = self._module_list[i](*args)

        return tuple(y[i] for i in self._head_indices)

    def save_model(self, filename: str, shape_info_collection: ShapeInfoCollection):
        """
        Saves this network to disk in ONNX format.
        """
        output_dir = os.path.split(filename)[0]
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        input_shapes = shape_info_collection.input_shapes
        head_shapes = shape_info_collection.head_shapes
        head_names = frozenset(head_shapes.keys())

        # 1) clone, strip extra heads, freeze
        clone_dict_key = head_names
        clone = self._clone_dict.get(clone_dict_key, None)
        if clone is None:
            trimmed_config = self._config.trim(head_names)
            clone = Model(trimmed_config)
            self._clone_dict[clone_dict_key] = clone
        clone.load_state_dict(self.state_dict(), strict=False)
        clone.cpu().eval()

        input_names = list(clone._external_inputs)
        output_names = clone._head_names

        # 2) make example inputs (one per external input of the trimmed clone) and ONNX-export
        batch_size = 1
        example_inputs = tuple(
            torch.zeros((batch_size, *input_shapes[name].shape), dtype=torch.float32)
            for name in input_names
        )
        dynamic_axes = {name: {0: "batch"} for name in input_names}

        # 3) Export to a temporary in-memory buffer
        buf = io.BytesIO()
        with mute_everything():
            torch.onnx.export(
                clone, example_inputs, buf,
                export_params=True,
                opset_version=18,
                input_names=input_names,
                output_names=output_names,
                dynamo=False,
                dynamic_axes=dynamic_axes,
                do_constant_folding=False,
            )

        # 4) Add metadata
        model = onnx.load_from_string(buf.getvalue())
        kv = model.metadata_props.add()
        kv.key = 'model-architecture-signature'
        kv.value = clone._model_architecture_signature

        # 5) Embed any auxiliary NNUE weights as orphan initializers in the ONNX graph.
        #
        # For paradigms like BetaZero, the model contains modules whose weights need to ship to
        # the C++ side by name (e.g. BackupNet's CPU-side dense layers, ChildEmbeddingHead's
        # `child_embed` weights for NNUE-style subtract-add updates). Each such module exposes
        # `collect_graph_initializers(out)` which inserts NumPy arrays into `out`. We walk the
        # live (un-trimmed) Model so that modules dropped by trim() (e.g. BackupNet, which is
        # not part of the inference graph) still contribute their weights.
        #
        # The C++ side reads them by name via core::parse_received_model
        # (see cpp/src/core/ReceivedModel.cpp).
        nnue_initializers: Dict[str, np.ndarray] = {}
        for module in self.modules():
            collector = getattr(module, 'collect_graph_initializers', None)
            if collector is None:
                continue
            collector(nnue_initializers)
        for name, arr in nnue_initializers.items():
            tensor = onnx.numpy_helper.from_array(np.ascontiguousarray(arr.astype(np.float32)),
                                                  name=f'nnue/{name}')
            model.graph.initializer.append(tensor)

        onnx.save(model, filename)

    def _collect_nnue_initializers(self) -> Dict[str, np.ndarray]:
        """
        Returns a flat dict of NNUE auxiliary weight tensors that would be embedded as orphan
        ONNX initializers (sans the `nnue/` prefix). Used by tests; production code should rely
        on save_model's walk over self.modules() instead.
        """
        out: Dict[str, np.ndarray] = {}
        for module in self.modules():
            collector = getattr(module, 'collect_graph_initializers', None)
            if collector is None:
                continue
            collector(out)
        return out


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
