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
    actions that can be expressed in the form [g⋅f](x) = g⋅(f(g⋅x)) for some action of G on R^M and R^N separately.

    GroupActionElements support declarative composition via Python operators

    >>> # Group multiplication: (g1 * g2)(x) = g1(g2(x)) for both input and output spaces
    >>> composed = g1 * g2
    >>> # Direct product on disjoint dim ranges: (g1 & g2)(concat(x1, x2)) = concat(g1(x1), g2(x2))
    >>> product = g1 & g2

    Nested compositions are automatically un-curried, so g1 * g2 * g3 yields a single ComposedActionElement with three
    sub-elements rather than a binary tree
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Parameters
        ----------
        input_dim
            The dimensionality of the input space this element acts on
        output_dim
            The dimensionality of the output space this element acts on
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

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
        return FiniteGroupAction([self], input_dim=self.input_dim, output_dim=self.output_dim)

    def __mul__(self, other: "GroupActionElement") -> "ComposedActionElement":
        """
        Composes two GroupActionElements via group multiplication. The resulting element acts on the input space as
        (g1 * g2).input_space_action(x) = g1.input_space_action(g2.input_space_action(x)), and likewise for the output
        space. Nested ComposedActionElement instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupActionElement to compose with

        Returns
        -------
        ComposedActionElement
            The composed group element
        """
        from iwpc.symmetries.composed_action_element import ComposedActionElement
        return ComposedActionElement.merge(self, other)

    def __and__(self, other: "GroupActionElement") -> "ProductActionElement":
        """
        Forms the direct product of two GroupActionElements acting on disjoint dim ranges. The resulting element acts on
        the concatenation of input feature vectors of length self.input_dim + other.input_dim by applying self to the
        first slice and other to the second slice, and likewise for the output space. Both operands must declare their
        input_dim and output_dim. Nested ProductActionElement instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupActionElement to take the direct product with

        Returns
        -------
        ProductActionElement
            The direct product element
        """
        from iwpc.symmetries.product_action_element import ProductActionElement
        return ProductActionElement.merge(self, other)


class Identity(GroupActionElement):
    """
    Convenience implementation of the action of the identity.
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Parameters
        ----------
        input_dim
            The dimensionality of the input space this identity acts on
        output_dim
            The dimensionality of the output space this identity acts on
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim)

    def input_space_action(self, x: Tensor) -> Tensor:
        raise InputSpaceInvariantException()

    def output_space_action(self, x: Tensor) -> Tensor:
        return x
