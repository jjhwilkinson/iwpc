from typing import Callable

from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.symmetrized_model import evaluate_group_action_outputs


class ComplementModel(Module):
    """
    Group actions, G, define a projection operator S_G where S_Gf(x) = E_G[gf(x)] and expectation is taken with
    respect to the Haar measure on G. This wrapper module implements the complement projection operator on the
    base_function, 1 - S_G. Note that the averaging procedure can significantly increase model evaluation time.
    """
    def __init__(self, group: GroupAction, base_function: Callable[[...], Tensor]):
        """
        Parameters
        ----------
        group
            A group action for which the resulting module should live in the symmetrized complement
        base_function
            A function
        """
        super().__init__()
        self.group = group
        self.base_model = base_function

    def forward(self, input: Tensor) -> Tensor:
        """
        Evaluates (1 - S_G) base_model. The transformed inputs for every group element and the original input are all
        passed through base_model in a single batched forward pass, so any input-dependent layers in base_model
        (running normalisation, batch normalisation, dropout, ...) apply consistently across the symmetrised and the
        original branches. Evaluating these branches in separate forward passes (as a naive
        `base_model(input) - SymmetrizedModel(input)` implementation would) breaks the antisymmetry identity in train
        mode

        Parameters
        ----------
        input
            An input Tensor

        Returns
        -------
        Tensor
        """
        per_action_outputs, base_input_output = evaluate_group_action_outputs(
            self.group, self.base_model, input
        )
        symmetrised = sum(per_action_outputs) / len(per_action_outputs)
        return base_input_output - symmetrised
