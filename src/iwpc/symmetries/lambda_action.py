from typing import Callable, Optional

from torch import Tensor

from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class LambdaAction(GroupActionElement):
    """
    Group action that acts using two arbitrary provided functions. Note that input_fn should be set to None if the
    action does not affect the input space rather than using an identity function so model calls can be re-used
    """
    def __init__(
        self,
        input_fn: Optional[Callable[[Tensor], Tensor]] = None,
        output_fn: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        input_fn
            A callable that acts on the input space. If this action is trivial, you should provide input_fn=None rather
            than an identity function like lambda x: x
        output_fn
            A callable that acts on the output space
        """
        super().__init__()

        if output_fn is None:
            output_fn = lambda x: x

        self.input_fn = input_fn
        self.output_fn = output_fn

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Applies the specified function to the input space. Raises an InputSpaceInvariantException if no input_fn was
            provided
        """
        if self.input_fn is None:
            raise InputSpaceInvariantException()
        return self.input_fn(x)

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Applies the specified function to the output space.
        """
        return self.output_fn(x)
