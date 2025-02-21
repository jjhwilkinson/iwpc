from abc import ABC, abstractmethod
from typing import List

from torch.nn import Module

from .group_action_element import GroupActionElement


class GroupAction(ABC):
    @abstractmethod
    def batch(self) -> List[GroupActionElement]:
        pass

    def symmetrize(self, base_model: Module) -> "SymmetrizedModel":
        from .symmetrized_model import SymmetrizedModel
        return SymmetrizedModel(self, base_model)

    def complement(self, base_model: Module) -> "ComplementModel":
        from .complement_model import ComplementModel
        return ComplementModel(self, base_model)
