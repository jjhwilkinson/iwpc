from typing import Callable

from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.separable_group_action import SeparableGroupAction
from iwpc.symmetries.symmetrized_model import SymmetrizedModel


class ComplementModel(Module):
    """
    Separable group actions, G, define a projection operator S_G where S_G f(x) = E_G[g.f(x)] and expectation is taken
    with respect to the Haar measure on G. This wrapper module implements the orthogonal complement projection,
    1 - S_G, on ``base_function``. Note that the averaging procedure can significantly increase model evaluation time
    """

    def __init__(self, group: SeparableGroupAction, base_function: Callable[[...], Tensor]):
        """
        Parameters
        ----------
        group
            A separable group action for which the resulting module should live in the symmetrized complement
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
