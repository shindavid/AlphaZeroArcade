from shared.basic_types import SearchParadigm, ShapeInfoDict
from shared.loss_term import LossTerm
from shared.net_modules import Head, MODULE_MAP

from torch import nn as nn
from torch import optim

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class ModuleSpec:
    type: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)
    repeat: int = 1  # number of times to repeat this module sequentially
    parents: List[str] = field(default_factory=list)  # names of parent modules

    def to_module(self) -> nn.Module:
        assert self.type in MODULE_MAP, f'Unknown module type {self.type}'
        cls = MODULE_MAP[self.type]
        modules = [cls(*self.args, **self.kwargs) for _ in range(self.repeat)]
        if self.repeat == 1:
            return modules[0]
        else:
            return nn.Sequential(*modules)


@dataclass
class ModelConfig:
    """
    A ModelConfig specifies an arbitrary directed acyclic graph (DAG) of modules, where some
    modules are designated as heads.

    The key method to create a ModelConfig is the static create() method, which takes
    keyword arguments mapping module names to ModuleSpec's.

    The topological structure is specified by the parents field of each ModuleSpec.
    """
    parts: Dict[str, ModuleSpec]

    def __post_init__(self):
        self._validate()

    @staticmethod
    def create(**parts: ModuleSpec) -> 'ModelConfig':
        return ModelConfig(dict(parts))

    def trim(self, keep_set: Set[str]) -> 'ModelConfig':
        """
        Returns a copy of this ModelConfig with only the parts in in keep_set retained, along with
        all ancestors of those parts. All other parts are removed.
        """
        assert keep_set, 'keep_set is empty'
        for part in keep_set:
            assert part in self.parts, f'Unknown part {part}'

        keep_set = set(keep_set)

        # add all ancestors of parts in keep_set
        for part in list(keep_set):
            parents = self.parts[part].parents
            while parents:
                new_parents = []
                for parent in parents:
                    if parent not in keep_set:
                        keep_set.add(parent)
                        new_parents.extend(self.parts[parent].parents)
                parents = new_parents

        trimmed_parts = {k: v for k, v in self.parts.items() if k in keep_set}
        return ModelConfig.create(**trimmed_parts)

    def _validate(self):
        input_seen = False
        parents = set()
        for key, value in self.parts.items():
            assert isinstance(value, ModuleSpec), f'{key}={type(value)}'
            assert value.type in MODULE_MAP, f'Unknown module type {value.type} for {key}'
            assert value.repeat >= 1, f'Invalid repeat {value.repeat} for {key}'
            if not value.parents:
                input_seen = True
            else:
                parents.update(value.parents)
                for parent in value.parents:
                    assert parent in self.parts, f'Unknown parent {parent} for {key}'

        assert input_seen, 'No modules process the input tensor!'

        heads = set(self.parts.keys()) - parents
        for head_name in heads:
            head_type = MODULE_MAP[self.parts[head_name].type]
            if not issubclass(head_type, Head):
                raise Exception(f'Module {head_name} has no children, but is of '
                                f'type {head_type}, which is not derived from Head')


class ModelConfigGenerator(abc.ABC):
    search_paradigm: SearchParadigm = SearchParadigm.AlphaZero

    @staticmethod
    @abc.abstractmethod
    def generate(shape_info_dict: ShapeInfoDict) -> ModelConfig:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_terms() -> List[LossTerm]:
        pass

    @staticmethod
    @abc.abstractmethod
    def optimizer(params) -> optim.Optimizer:
        pass
