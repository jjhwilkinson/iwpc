from typing import Iterable, Tuple, Optional

from torch.nn import ModuleList

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement, Identity


class FiniteGroupAction(GroupAction):
    """
    Generic implementation of a finite group action. Overrides the '&' and '*' operators so that when both operands
    are FiniteGroupActions the result is a FiniteGroupAction enumerating the full direct product or full Cartesian
    product of elements respectively. When the other operand is not finite, the operators fall back to the generic
    ProductGroupAction or JointGroupAction wrappers
    """
    def __init__(
        self,
        non_id_elements: Iterable[GroupActionElement],
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        non_id_elements
            An iterable of the non-identity GroupActionElements in the group action
        input_dim
            The dimensionality of the input space this group acts on. Optional in general but required for use with the
            '&' direct-product operator. The prepended Identity element is constructed with this dim
        output_dim
            The dimensionality of the output space this group acts on. Optional in general but required for use with the
            '&' direct-product operator. The prepended Identity element is constructed with this dim
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        self.elements = ModuleList([Identity(input_dim=input_dim, output_dim=output_dim), *non_id_elements])

    def batch(self) -> Tuple[GroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[GroupActionElement, ...]
            All the elements in the group action, including the identity element
        """
        return self.elements

    def __len__(self):
        """
        Returns
        -------
        int
            The number of elements in the group action including the identity element
        """
        return len(self.elements)

    def __and__(self, other: GroupAction) -> GroupAction:
        """
        Specialised direct product. When other is also a FiniteGroupAction, returns a FiniteGroupAction enumerating
        the full direct product |self| * |other| of elements as ProductActionElements. Otherwise falls back to the
        generic ProductGroupAction wrapper

        Parameters
        ----------
        other
            A GroupAction to take the direct product with

        Returns
        -------
        GroupAction
            A FiniteGroupAction if other is a FiniteGroupAction, otherwise a ProductGroupAction
        """
        if isinstance(other, FiniteGroupAction):
            from iwpc.symmetries.product_group_action import _build_finite_product
            return _build_finite_product([self, other])
        return super().__and__(other)

    def __mul__(self, other: GroupAction) -> GroupAction:
        """
        Specialised joint action on the same space. When other is also a FiniteGroupAction, returns a
        FiniteGroupAction enumerating the full Cartesian product |self| * |other| of elements as
        ComposedActionElements. Otherwise falls back to the generic JointGroupAction wrapper

        Parameters
        ----------
        other
            A GroupAction to compose jointly with

        Returns
        -------
        GroupAction
            A FiniteGroupAction if other is a FiniteGroupAction, otherwise a JointGroupAction
        """
        if isinstance(other, FiniteGroupAction):
            from iwpc.symmetries.joint_group_action import _build_finite_joint
            return _build_finite_joint([self, other])
        return super().__mul__(other)
