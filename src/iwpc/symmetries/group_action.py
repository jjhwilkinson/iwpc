from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import Module

from .group_action_element import GroupActionElement


class GroupAction(ABC, Module):
    """
    Abstract interface for group actions acting on a single vector space R^dim. Provides the batch method enabling
    averaging over the group by averaging over batches of its action. Concrete subclasses must implement :py:meth:`batch`

    Two of these may be combined into a :py:class:`SeparableGroupAction` to recover the original separable
    function-space action, where one GroupAction acts on the input space and the other on the output space

    GroupActions support declarative composition via Python operators

    >>> # Direct product on disjoint dim ranges. For finite groups, the full direct product is enumerated
    >>> product_group = G1 & G2
    >>> # Joint action on the same space. For finite groups, the full Cartesian product is enumerated
    >>> joint_group = G1 * G2

    Nested compositions are automatically un-curried, so G1 & G2 & G3 yields a single ProductGroupAction with three
    sub-groups rather than a binary tree
    """

    def __init__(self, dim: int):
        """
        Parameters
        ----------
        dim
            The dimensionality of the vector space this group acts on
        """
        super().__init__()
        self.dim = dim

    @abstractmethod
    def batch(self) -> Tuple[GroupActionElement, ...]:
        """
        Provides a batch of group action elements sampled from the Haar measure of the group. Small finite groups
        should return all elements in every batch, but larger and even infinite groups should return a batch of samples
        from the Haar measure of the group

        Returns
        -------
        Tuple[GroupActionElement, ...]
        """

    def __and__(self, other: "GroupAction") -> "GroupAction":
        """
        Forms the direct product of two GroupActions acting on disjoint dim ranges. When both operands are
        FiniteGroupActions the full direct product is enumerated as |self| * |other| ProductActionElements. Otherwise
        batches are drawn jointly by zipping self.batch() with other.batch(). Nested ProductGroupAction instances are
        automatically un-curried

        Parameters
        ----------
        other
            A GroupAction to take the direct product with

        Returns
        -------
        GroupAction
            The direct product action
        """
        from .product_group_action import ProductGroupAction
        return ProductGroupAction.merge(self, other)

    def __mul__(self, other: "GroupAction") -> "GroupAction":
        """
        Forms the joint action of two GroupActions acting on the same space. When both operands are
        FiniteGroupActions, the full Cartesian product of elements is enumerated as |self| * |other|
        ComposedActionElements. Otherwise batches are drawn jointly by zipping self.batch() with other.batch() and
        composing each pair. Nested JointGroupAction instances are automatically un-curried

        Parameters
        ----------
        other
            A GroupAction to compose jointly with

        Returns
        -------
        GroupAction
            The joint action
        """
        from .joint_group_action import JointGroupAction
        return JointGroupAction.merge(self, other)
