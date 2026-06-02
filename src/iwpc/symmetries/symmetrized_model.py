from typing import Callable, List, Tuple

import torch
from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import InputSpaceInvariantException


def evaluate_group_action_outputs(
    group: GroupAction,
    base_function: Callable[..., Tensor],
    input: Tensor,
) -> Tuple[List[Tensor], Tensor]:
    """
    Evaluates base_function on the input transformed by every group element in a single batched forward pass, applies
    each element's output space action to the corresponding slice, and additionally returns the slice corresponding to
    the identity action (i.e. base_function(input)).

    Performing every evaluation as one batched forward call ensures that any input-dependent layers in base_function
    (running normalisation, batch normalisation, dropout, ...) see a single, consistent batch of activations. This is
    relied on by SymmetrizedModel and ComplementModel below: evaluating the base_function on each group-action input in
    a separate call would otherwise apply different normalisation to each branch, breaking the projection identities

    Parameters
    ----------
    group
        The group action over which to evaluate base_function
    base_function
        A function (typically a model) to evaluate on each transformed input
    input
        The input tensor on which the group acts

    Returns
    -------
    Tuple[List[Tensor], Tensor]
        - per_action_outputs: for each group element, the base_function output for its transformed input passed through
          the element's output_space_action. Same length as group.batch()
        - base_input_output: the un-transformed base_function(input) slice, taken from the same batched forward pass
    """
    full_input: List[Tensor] = []
    actions = list(group.batch())
    original_input_idx = None
    output_indices: List[int] = []
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

    if original_input_idx is None:
        # No identity-action shortcut hit yet (no element raised InputSpaceInvariantException). Append the original
        # input so base_function(input) shares the same forward pass as the transformed inputs
        full_input.append(input)
        original_input_idx = max_output_idx + 1

    full_inputs = torch.stack(full_input, dim=0).reshape((-1, *input.shape[1:]))
    base_output = base_function(full_inputs)
    base_output = base_output.reshape((len(full_input), input.shape[0], *base_output.shape[1:]))

    per_action_outputs = [
        action.output_space_action(base_output[idx])
        for action, idx in zip(actions, output_indices)
    ]
    base_input_output = base_output[original_input_idx]
    return per_action_outputs, base_input_output


class SymmetrizedModel(Module):
    """
    Group actions, G, define a projection operator S_G where S_Gf(x) = E_G[gf(x)] and expectation is taken with
    respect to the Haar measure on G. This wrapper module implements the symmetrisation projection operator on the
    base_function. The resulting module is invariant under the action of G. Note that the averaging procedure can
    significantly increase model evaluation time.
    """
    def __init__(self, group: GroupAction, base_function: Callable[[...], Tensor]):
        """
        Parameters
        ----------
        group
            A group action over which the base_model should be averaged
        base_function
            Some function
        """
        super().__init__()
        self.group = group
        self.base_model = base_function

    def forward(self, input: Tensor) -> Tensor:
        """
        Computes the average of base_model under the group action, i.e. S_G base_model. Implementation evaluates every
        action's transformed input in a single batched forward pass through base_model so any input-dependent layers
        (normalisation, dropout, ...) apply consistently across actions

        Parameters
        ----------
        input
            An input tensor

        Returns
        -------
        Tensor
            The average of base_model under the group action for the batch
        """
        per_action_outputs, _ = evaluate_group_action_outputs(self.group, self.base_model, input)
        return sum(per_action_outputs) / len(per_action_outputs)
