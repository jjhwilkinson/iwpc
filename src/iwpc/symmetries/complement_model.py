from typing import Callable

from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.symmetrized_model import SymmetrizedModel


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
        self.symmetrized_model = SymmetrizedModel(group, base_function)

    def forward(self, input: Tensor) -> Tensor:
        """
        Evaluates (1 - S_G) base_model

        Parameters
        ----------
        input
            An input Tensor

        Returns
        -------
        Tensor
        """
        return self.base_model(input) - self.symmetrized_model(input)
