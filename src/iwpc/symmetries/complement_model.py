from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.symmetrized_model import SymmetrizedModel


class ComplementModel(Module):
    def __init__(self, group: GroupAction, base_model):
        super().__init__()
        self.group = group
        self.base_model = base_model
        self.symmetrized_model = SymmetrizedModel(group, base_model)

    def forward(self, input: Tensor) -> Tensor:
        return self.base_model(input) - self.symmetrized_model(input)
