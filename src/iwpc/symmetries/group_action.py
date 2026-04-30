from abc import ABC, abstractmethod
from typing import Tuple, Callable

from torch import Tensor
from torch.nn import Module

from .group_action_element import GroupActionElement


class GroupAction(ABC, Module):
    """
    Abstract interface for group actions acting on the function space accessible to a NN from R^M -> R^N. We restrict
    ourselves to actions that act separately on the input and output spaces, that is group actions that can be expressed
    in the form [g⋅f](x) = g⋅(f(g⋅x)) for some action of G on R^M and R^N separately. In particular, provides the batch
    method enabling averaging over the group by averaging over batches of its action.

    GroupActions support declarative composition via Python operators

    >>> # Direct product on disjoint dim ranges. For finite groups, the full direct product is enumerated
    >>> product_group = G1 & G2
    >>> # Joint action on the same space. For finite groups, the full Cartesian product is enumerated
    >>> joint_group = G1 * G2

    Nested compositions are automatically un-curried, so G1 & G2 & G3 yields a single ProductGroupAction with three
    sub-groups rather than a binary tree
    """
    def __init__(self, input_dim: int, output_dim: int):
        """
        Parameters
        ----------
        input_dim
            The dimensionality of the input space this group acts on
        output_dim
            The dimensionality of the output space this group acts on
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def batch(self) -> Tuple[GroupActionElement]:
        """
        Provides a batch of group action elements sampled from the Haar measure of the group. Small finite groups
        should return all elements in every batch, but larger and even infinite groups should return a batch of samples
        from the Haar measure of the group.

        Returns
        -------
        Tuple[GroupActionElement]
        """

    def symmetrize(self, base_function: Callable[..., Tensor]) -> "SymmetrizedModel":
        """
        Helper function to wrap a function in a SymmetrizedModel resulting in a function symmetric with respect to this group
        action

        Parameters
        ----------
        base_function
            A function to symmetrize

        Returns
        -------
        SymmetrizedModel
            A symmetrized function
        """
        from .symmetrized_model import SymmetrizedModel
        return SymmetrizedModel(self, base_function)

    def complement(self, base_function: Callable[..., Tensor]) -> "ComplementModel":
        """
        Helper function to wrap a function in a ComplementModel resulting in a function in the complement of the symmetrization
        projection of this group action

        Parameters
        ----------
        base_function
            A function to symmetrize

        Returns
        -------
        SymmetrizedModel
            A function in the complement of the symmetrization projection of this group action
        """
        from .complement_model import ComplementModel
        return ComplementModel(self, base_function)

    def __and__(self, other: "GroupAction") -> "ProductGroupAction":
        """
        Forms the direct product of two GroupActions acting on disjoint input and output dim ranges. Both operands must
        declare their input_dim and output_dim. When both operands are FiniteGroupActions, the full direct product is
        enumerated as |self| * |other| ProductActionElements. Otherwise, batches are drawn jointly by zipping
        self.batch() with other.batch(). Nested ProductGroupAction instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupAction to take the direct product with

        Returns
        -------
        ProductGroupAction
            The direct product action
        """
        from .product_group_action import ProductGroupAction
        return ProductGroupAction.merge(self, other)

    def __mul__(self, other: "GroupAction") -> "JointGroupAction":
        """
        Forms the joint action of two GroupActions acting on the same input and output space. When both operands are
        FiniteGroupActions, the full Cartesian product of elements is enumerated as |self| * |other| ComposedActionElements.
        Otherwise, batches are drawn jointly by zipping self.batch() with other.batch() and composing each pair. Nested
        JointGroupAction instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupAction to compose jointly with

        Returns
        -------
        JointGroupAction
            The joint action
        """
        from .joint_group_action import JointGroupAction
        return JointGroupAction.merge(self, other)
