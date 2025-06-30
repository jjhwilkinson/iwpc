import torch
from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import InputSpaceInvariantException


class SymmetrizedModel(Module):
    """
    Group actions, G, define a projection operator S_G where S_Gf(x) = E_G[gf(x)] and expectation is taken with
    respect to the Haar measure on G. This wrapper module implements the complement projection operator on the
    base_model. The resulting module is invariant under the action of G. Note that the averaging procedure can
    significantly increase model evaluation time.
    """
    def __init__(self, group: GroupAction, base_model: Module):
        """
        Parameters
        ----------
        group
            A group action over which the base_model should be averaged
        base_model
            Some base model
        """
        super().__init__()
        self.group = group
        self.base_model = base_model

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the average of base_model under the group action, i.e. S_G base_model. Implementation will re-use model
        calls if a group action element does not affect the input space and raises an InputSpaceInvariantException

        Parameters
        ----------
        input
            An input tensor

        Returns
        -------
        Tensor
            The average of base_model under the group action for the batch
        """
        full_input = []
        actions = list(self.group.batch())
        original_input_idx = None
        output_indices = []
        max_output_idx = -1

        for action in actions:
            try:
                full_input.append(action.input_space_action(input))
                max_output_idx += 1
                output_indices.append(max_output_idx)
            except InputSpaceInvariantException:
                if original_input_idx is None:
                    full_input.append(input)
                    max_output_idx += 1
                    original_input_idx = max_output_idx
                output_indices.append(original_input_idx)

        full_inputs = torch.stack(full_input, dim=0).reshape((-1, *input.shape[1:]))
        base_output = self.base_model(full_inputs)
        base_output = base_output.reshape((len(full_input), input.shape[0], *base_output.shape[1:]))

        final_outputs = []
        for action, output_idx in zip(actions, output_indices):
            final_outputs.append(action.output_space_action(base_output[output_idx]))

        return sum(final_outputs) / len(final_outputs)
