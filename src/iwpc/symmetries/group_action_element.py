from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class GroupActionElement(Module, ABC):
    @abstractmethod
    def input_space_action(self, x: Tensor) -> Tensor:
        pass

    @abstractmethod
    def output_space_action(self, x: Tensor) -> Tensor:
        pass


class Identity(GroupActionElement):
    def input_space_action(self, x: Tensor) -> Tensor:
        return x

    def output_space_action(self, x: Tensor) -> Tensor:
        return x
