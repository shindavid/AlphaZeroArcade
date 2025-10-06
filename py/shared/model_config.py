from shared.basic_types import SearchParadigm, ShapeInfoCollection
from shared.loss_term import LossTerm
from shared.net_modules import Head, MODULE_MAP

from torch import nn as nn
from torch import optim

import abc
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


class ModuleSpecBase(abc.ABC):
    def __init__(self, type: Optional[str], args=None, kwargs=None, repeat=1, parents=None):
        self.type = type
        self.args = args if args is not None else []
        self.kwargs = kwargs if kwargs is not None else {}
        self.repeat = repeat
        self.parents = parents if parents is not None else []

    @abc.abstractmethod
    def to_module(self) -> nn.Module:
        pass


class ModuleSpec(ModuleSpecBase):
    def __init__(self, type: str, args=None, kwargs=None, repeat=1, parents=None):
        super().__init__(type, args, kwargs, repeat, parents)

    def to_module(self) -> nn.Module:
        assert self.type in MODULE_MAP, f'Unknown module type {self.type}'
        cls = MODULE_MAP[self.type]
        modules = [cls(*self.args, **self.kwargs) for _ in range(self.repeat)]
        if self.repeat == 1:
            return modules[0]
        else:
            return nn.Sequential(*modules)


class ModuleSequenceSpec(ModuleSpecBase):
    def __init__(self, *specs: ModuleSpecBase, parents=None):
        super().__init__(None, parents=parents)
        self.specs = list(specs)

    def to_module(self) -> nn.Module:
        return nn.Sequential(*(spec.to_module() for spec in self.specs))


@dataclass
class ModelConfig:
    """
    A ModelConfig specifies an arbitrary directed acyclic graph (DAG) of modules, where some
    modules are designated as heads.

    The key method to create a ModelConfig is the static create() method, which takes
    keyword arguments mapping module names to ModuleSpec's.

    The topological structure is specified by the parents field of each ModuleSpec.
    """
    parts: Dict[str, ModuleSpecBase]

    def __post_init__(self):
        self._validate()

    @staticmethod
    def create(**parts: ModuleSpecBase) -> 'ModelConfig':
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
        types = set()
        parents = set()

        specs = set(self.parts.values())
        while specs:
            spec = specs.pop()
            if isinstance(spec, ModuleSequenceSpec):
                specs.update(spec.specs)
            else:
                assert isinstance(spec, ModuleSpec), f'Invalid spec {spec}'
                assert spec.repeat >= 1, f'Invalid repeat {spec.repeat} for {spec}'
                assert spec.type is not None, f'ModuleSpec {spec} has no type'
                types.add(spec.type)

        for spec_type in types:
            assert spec_type in MODULE_MAP, f'Unknown module type {spec_type}'

        for key, value in self.parts.items():
            if not value.parents:
                input_seen = True
            else:
                parents.update(value.parents)
                for parent in value.parents:
                    assert parent in self.parts, f'Unknown parent {parent} for {key}'

        assert input_seen, 'No modules process the input tensor!'

        heads = set(self.parts.keys()) - parents
        for head_name in heads:
            head_type_str = self.parts[head_name].type
            assert head_type_str is not None, f'Head {head_name} has no type'
            head_type = MODULE_MAP[head_type_str]
            if not issubclass(head_type, Head):
                raise Exception(f'Module {head_name} has no children, but is of '
                                f'type {head_type}, which is not derived from Head')


class ModelConfigGenerator(abc.ABC):
    search_paradigm: SearchParadigm = SearchParadigm.AlphaZero

    @staticmethod
    @abc.abstractmethod
    def generate(head_shape_info_collection: ShapeInfoCollection) -> ModelConfig:
        pass

    @staticmethod
    @abc.abstractmethod
    def loss_terms() -> List[LossTerm]:
        pass

    @staticmethod
    @abc.abstractmethod
    def optimizer(params) -> optim.Optimizer:
        pass
