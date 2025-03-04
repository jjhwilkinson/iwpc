from typing import Union, List

import torch
from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction


class SymmetrizedModel(Module):
    def __init__(self, group: GroupAction, base_model, return_haar_batch_results=False):
        super().__init__()
        self.group = group
        self.base_model = base_model
        self.return_haar_batch_results = return_haar_batch_results

    def forward(self, input: Tensor) -> Union[Tensor, List[Tensor]]:
        full_input = []
        actions = list(self.group.batch())
        for action in actions:
            full_input.append(action.input_space_action(input))
        full_inputs = torch.stack(full_input, dim=0).reshape((-1, *input.shape[1:]))
        base_output = self.base_model(full_inputs)
        base_output = base_output.reshape((len(full_input), input.shape[0], *base_output.shape[1:]))

        final_outputs = []
        for action, output in zip(actions, base_output):
            final_outputs.append(action.output_space_action(output))

        if self.return_haar_batch_results:
            return final_outputs

        return sum(final_outputs) / len(final_outputs)
