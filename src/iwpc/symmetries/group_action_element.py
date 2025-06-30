from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class InputSpaceInvariantException(Exception):
    """
    Special exception that may be raised in the implementation of the input_space_action method of a GroupActionElement
    if the action does not affect the input space. It is recommended to raise this exception rather than returning the
    input tensor as various implementations can use this fact to speed up execution and prevent re-evaluating models
    on duplicate inputs
    """
    def __init__(self):
        super().__init__("Input space is invariant under group element action")


class GroupActionElement(Module, ABC):
    """
    Abstract interface for the action of a particular group element, g, on the function space accessible to a NN from
    R^M -> R^N. We restrict ourselves to actions that act separately on the input and output spaces, that is group
    actions that can be expressed in the form [g⋅f](x) = g⋅(f(g⋅x)) for some action of G on R^M and R^N separately
    """
    @abstractmethod
    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Performs the action of the group element on the input space, R^M, of the function. If the action does not affect
        the input space, then this function should raise an InputSpaceInvariantException to inform the caller that it
        may re-use previous model evaluations of the original inputs

        Parameters
        ----------
        x
            An input tensor in R^M

        Returns
        -------
        Tensor
            The action of g input tensor, gx
        """

    @abstractmethod
    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Performs the action of the group element on the output space, R^N, of the function

        Parameters
        ----------
        x
            An input tensor of output values in R^N

        Returns
        -------
        Tensor
            The action of g input tensor, gx
        """

    def to_group(self) -> "FiniteGroupAction":
        """
        Constructs a group action containing the identity and this group action element. Warning, this method should only
        be used if this group action element is an involution. In other words, this action undoes itself. It is your
        responsibility to check this

        Returns
        -------
        FiniteGroupAction
            A FiniteGroupAction containing only this element and the identity
        """
        from iwpc.symmetries.finite_group_action import FiniteGroupAction
        return FiniteGroupAction([self])


class Identity(GroupActionElement):
    """
    Convenience implementation of the action of the identity.
    """
    def input_space_action(self, x: Tensor) -> Tensor:
        raise InputSpaceInvariantException()

    def output_space_action(self, x: Tensor) -> Tensor:
        return x
