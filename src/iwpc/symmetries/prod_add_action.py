import torch
from torch import Tensor

from iwpc.symmetries.finite_group_action import Z2GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement


class ProdAddAction(GroupActionElement):
    def input_space_action(self, x: Tensor) -> Tensor:
        return x * self.input_prod + self.input_add

    def output_space_action(self, x: Tensor) -> Tensor:
        return x * self.output_prod + self.output_add

    def __init__(
        self,
        input_prod,
        input_add,
        output_prod,
        output_add,
    ):
        super().__init__()
        self.register_buffer('input_prod', torch.as_tensor(input_prod, dtype=torch.float)[None, :])
        self.register_buffer('input_add', torch.as_tensor(input_add, dtype=torch.float)[None, :])
        self.register_buffer('output_prod', torch.as_tensor(output_prod, dtype=torch.float)[None, :])
        self.register_buffer('output_add', torch.as_tensor(output_add, dtype=torch.float)[None, :])


class ProdAddZ2GroupAction(Z2GroupAction):
    def __init__(
        self,
        input_prod,
        input_add,
        output_prod,
        output_add,
    ):
        super().__init__(ProdAddAction(input_prod, input_add, output_prod, output_add))
