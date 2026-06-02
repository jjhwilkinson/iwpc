from typing import Callable

from torch import Tensor

from iwpc.symmetries.group_action_element import GroupActionElement


class LambdaAction(GroupActionElement):
    """
    Group action element that acts on R^dim using an arbitrary provided function
    """

    def __init__(self, dim: int, fn: Callable[[Tensor], Tensor]):
        """
        Parameters
        ----------
        dim
            The dimensionality of the vector space this element acts on
        fn
            A callable mapping a tensor in R^dim to a tensor in R^dim
        """
        super().__init__(dim=dim)
        self.fn = fn

    def action(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            An input tensor with last dimension self.dim

        Returns
        -------
        Tensor
            self.fn(x)
        """
        return self.fn(x)
