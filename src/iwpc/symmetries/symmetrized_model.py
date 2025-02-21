import torch
from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction


class SymmetrizedModel(Module):
    def __init__(self, group: GroupAction, base_model):
        super().__init__()
        self.group = group
        self.base_model = base_model

    def forward(self, input: Tensor) -> Tensor:
        full_input = []
        actions = self.group.batch()
        for action in actions:
            full_input.append(action.input_space_action(input))
        full_inputs = torch.stack(full_input, dim=0).reshape((-1, *input.shape[1:]))
        base_output = self.base_model(full_inputs).reshape((-1, *input.shape))

        final_output = None
        for action, output in zip(actions, base_output):
            output = action.output_space_action(output)
            if final_output is None:
                final_output = output
            else:
                final_output = final_output + output

        return final_output / len(actions)
