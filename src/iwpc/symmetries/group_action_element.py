from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module


class GroupActionElement(Module, ABC):
    """
    Abstract interface for the action of a particular group element, g, on a single vector space R^dim. Concrete
    subclasses implement :py:meth:`action` plus any structure-preserving algebra they support. The separable function
    space wrappers in :py:mod:`iwpc.symmetries.separable_group_action_element` combine two of these — one acting on the
    input space and one on the output space — to recover the original function-space action

    GroupActionElements support declarative composition via Python operators

    >>> # Group multiplication: (g1 * g2)(x) = g1(g2(x))
    >>> composed = g1 * g2
    >>> # Direct product on disjoint dim ranges: (g1 & g2)(concat(x1, x2)) = concat(g1(x1), g2(x2))
    >>> product = g1 & g2

    Nested compositions are automatically un-curried, so g1 * g2 * g3 yields a single ComposedActionElement with three
    sub-elements rather than a binary tree

    Subclasses may override the :py:attr:`is_identity` class attribute, or set it as an instance attribute, to advertise
    that the element is the identity transformation. The separable wrapper layer reads this flag to skip duplicate model
    evaluations of unchanged inputs
    """

    is_identity: bool = False

    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim
            The dimensionality of the vector space this element acts on
        """
        super().__init__()
        self.dim = dim

    @abstractmethod
    def action(self, x: Tensor) -> Tensor:
        """
        Performs the action of the group element on a tensor in R^dim

        Parameters
        ----------
        x
            An input tensor with last dimension self.dim

        Returns
        -------
        Tensor
            The action of g on the input tensor, gx
        """

    def __mul__(self, other: "GroupActionElement") -> "GroupActionElement":
        """
        Composes two GroupActionElements via group multiplication. The resulting element acts as
        (g1 * g2).action(x) = g1.action(g2.action(x)). Nested ComposedActionElement instances are automatically
        un-curried

        Parameters
        ----------
        other
            A GroupActionElement to compose with

        Returns
        -------
        GroupActionElement
            The composed group element
        """
        from iwpc.symmetries.composed_action_element import ComposedActionElement
        return ComposedActionElement.merge(self, other)

    def __and__(self, other: "GroupActionElement") -> "GroupActionElement":
        """
        Forms the direct product of two GroupActionElements acting on disjoint dim ranges. The resulting element acts on
        the concatenation of feature vectors of length self.dim + other.dim by applying self to the first slice and
        other to the second slice. Nested ProductActionElement instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupActionElement to take the direct product with

        Returns
        -------
        GroupActionElement
            The direct product element
        """
        from iwpc.symmetries.product_action_element import ProductActionElement
        return ProductActionElement.merge(self, other)


class Identity(GroupActionElement):
    """
    Convenience implementation of the identity action on R^dim
    """

    is_identity = True

    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim
            The dimensionality of the vector space this identity acts on
        """
        super().__init__(dim=dim)

    def action(self, x: Tensor) -> Tensor:
        """
        Returns the input tensor unchanged

        Parameters
        ----------
        x
            An input tensor

        Returns
        -------
        Tensor
            x, unchanged
        """
        return x


class InputSpaceInvariantException(Exception):
    """
    Legacy sentinel signalling that the input space side of a separable action is the identity. Retained for
    back-compat: :py:class:`iwpc.symmetries.symmetrized_model.SymmetrizedModel` still catches this so user-defined
    elements that raise it continue to dedupe correctly. New code should prefer setting
    :py:attr:`GroupActionElement.is_identity` on the input-side element
    """

    def __init__(self):
        super().__init__("Input space is invariant under group element action")
