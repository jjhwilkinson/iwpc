from typing import Iterable, Callable

from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement, Identity


class FiniteGroupAction(GroupAction):
    def __init__(self, elements: Iterable[GroupActionElement]):
        super().__init__()
        self.elements = set(elements)

    def batch(self):
        for element in self.elements:
            yield element

    def __len__(self):
        return len(self.elements)


class Z2GroupAction(FiniteGroupAction):
    def __init__(self, involution: Callable[[Tensor], Tensor]):
        super().__init__([Identity(), involution])
        self.involution = involution
