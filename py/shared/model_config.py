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
    keyword arguments mapping module names to ModuleSpec's, plus an optional `external_inputs`
    list naming the additional Model inputs (beyond the universal `'input'`) that get fed in
    at training time (typically sourced from FFI training-target tensors).

    The topological structure is specified by the `parents` field of each ModuleSpec. Each
    parent name is interpreted by membership: it must be either (a) a key of `parts` (sibling
    module output) or (b) a member of the effective external-input set, which is
    `{'input'} | external_inputs`. A spec with `parents=[]` (or unset) defaults to having
    `['input']` as its sole parent.
    """
    parts: Dict[str, ModuleSpecBase]
    external_inputs: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._validate()

    @staticmethod
    def create(*, external_inputs: Optional[List[str]] = None,
               **parts: ModuleSpecBase) -> 'ModelConfig':
        return ModelConfig(dict(parts), external_inputs=list(external_inputs or []))

    def effective_external_inputs(self) -> List[str]:
        """
        Returns the ordered list of effective external inputs: `'input'` (always implicitly
        available) prepended to any user-declared `external_inputs`, with duplicates removed.
        """
        seen = set()
        out: List[str] = []
        for name in ('input', *self.external_inputs):
            if name not in seen:
                seen.add(name)
                out.append(name)
        return out

    def trim(self, keep_set: Set[str]) -> 'ModelConfig':
        """
        Returns a copy of this ModelConfig with only the parts in keep_set retained, along with
        all ancestors of those parts. All other parts are removed. The `external_inputs` list is
        filtered to those still referenced by the trimmed graph.
        """
        assert keep_set, 'keep_set is empty'
        for part in keep_set:
            assert part in self.parts, f'Unknown part {part}'

        keep_set = set(keep_set)

        # add all ancestors of parts in keep_set (external inputs are skipped naturally
        # since they are not in self.parts)
        for part in list(keep_set):
            parents = self.parts[part].parents
            while parents:
                new_parents = []
                for parent in parents:
                    if parent in self.parts and parent not in keep_set:
                        keep_set.add(parent)
                        new_parents.extend(self.parts[parent].parents)
                parents = new_parents

        trimmed_parts = {k: v for k, v in self.parts.items() if k in keep_set}

        # Filter external_inputs to those still referenced by some surviving spec.
        referenced: Set[str] = set()
        for spec in trimmed_parts.values():
            for p in (spec.parents if spec.parents else ['input']):
                if p not in trimmed_parts:
                    referenced.add(p)
        new_externals = [n for n in self.external_inputs if n in referenced]
        return ModelConfig(trimmed_parts, external_inputs=new_externals)

    def _validate(self):
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

        # No name may serve as both a module key and a declared external input.
        for name in self.external_inputs:
            assert name not in self.parts, (
                f'Name {name!r} appears in both `parts` and `external_inputs`')

        effective_externals = set(self.effective_external_inputs())

        for key, value in self.parts.items():
            effective_parents = value.parents if value.parents else ['input']
            parents.update(effective_parents)
            for parent in effective_parents:
                assert parent in self.parts or parent in effective_externals, (
                    f'Module {key!r} parent {parent!r} is neither a sibling module nor a '
                    f'declared external input. Available modules: {sorted(self.parts.keys())}; '
                    f'effective external inputs: {sorted(effective_externals)}.')

        # Every declared external input must be referenced by at least one module.
        for name in self.external_inputs:
            assert name in parents, (
                f'Declared external_input {name!r} is not referenced by any module')

        assert parents & effective_externals, 'No modules reference any external input!'

        # Leaves (parts not referenced as anyone's parent) are typically Head modules whose
        # outputs are returned by Model.forward. Non-Head leaves are allowed: they are
        # training-only modules (e.g. BetaZero's BackupNet) whose outputs are consumed by loss
        # terms outside the inference graph.
        heads = set(self.parts.keys()) - parents
        for head_name in heads:
            head_type_str = self.parts[head_name].type
            assert head_type_str is not None, f'Head {head_name} has no type'


class ModelConfigGenerator(abc.ABC):
    spec_name: str = 'alpha0'
    paradigm: SearchParadigm = SearchParadigm.AlphaZero

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
