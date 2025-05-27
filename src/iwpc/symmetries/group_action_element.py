from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class GroupActionElement(Module, ABC):
    """
    Abstract interface for the action of a particular group element, g, on the function space accessible to a NN from
    R^M -> R^N. We restrict ourselves to actions that act separately on the input and output spaces, that is group
    actions that can be expressed in the form [g⋅f](x) = g⋅(f(g⋅x)) for some action of G on R^M and R^N separately
    """
    @abstractmethod
    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Performs the action of the group element on the input space, R^M, of the function

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


class Identity(GroupActionElement):
    """
    Convenience implementation of the action of the identity.
    """
    def input_space_action(self, x: Tensor) -> Tensor:
        return x

    def output_space_action(self, x: Tensor) -> Tensor:
        return x
