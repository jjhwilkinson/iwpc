import itertools
from typing import Iterable, List, Tuple

from torch.nn import ModuleList

from iwpc.symmetries.group_action_element import Identity
from iwpc.symmetries.separable_group_action import (
    SeparableGroupAction,
    _separable_joint_compose,
    _separable_product_compose,
    _validate_consistent_separable_dims,
)
from iwpc.symmetries.separable_group_action_element import SeparableGroupActionElement


class SeparableFiniteGroupAction(SeparableGroupAction):
    """
    Generic implementation of a finite separable group action on the function space R^input_dim -> R^output_dim. Stores
    an explicit list of :py:class:`SeparableGroupActionElement` pairs with the identity pair prepended. Overrides the
    ``&`` and ``*`` operators so that when both operands are SeparableFiniteGroupActions the result is a
    SeparableFiniteGroupAction enumerating the full direct or full Cartesian product of element pairs respectively
    """

    def __init__(
        self,
        non_id_elements: Iterable[SeparableGroupActionElement],
        input_dim: int,
        output_dim: int,
    ):
        """
        Parameters
        ----------
        non_id_elements
            An iterable of the non-identity SeparableGroupActionElements in the group
        input_dim
            The dimensionality of the input space. The prepended identity pair acts on this dim
        output_dim
            The dimensionality of the output space. The prepended identity pair acts on this dim
        """
        super().__init__(input_dim=input_dim, output_dim=output_dim)
        identity_pair = SeparableGroupActionElement(
            input_action=Identity(dim=input_dim),
            output_action=Identity(dim=output_dim),
        )
        self.elements = ModuleList([identity_pair, *non_id_elements])

    def batch(self) -> Tuple[SeparableGroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[SeparableGroupActionElement, ...]
            All elements in the group action, including the identity pair
        """
        return self.elements

    def __len__(self):
        return len(self.elements)

    def __and__(self, other: SeparableGroupAction) -> SeparableGroupAction:
        """
        Specialised direct product. When other is also a SeparableFiniteGroupAction, enumerates the full direct
        product as a SeparableFiniteGroupAction. Otherwise falls back to the generic SeparableProductGroupAction
        wrapper
        """
        if isinstance(other, SeparableFiniteGroupAction):
            return _build_finite_separable_product([self, other])
        return super().__and__(other)

    def __mul__(self, other: SeparableGroupAction) -> SeparableGroupAction:
        """
        Specialised joint action on the same space. When other is also a SeparableFiniteGroupAction, enumerates the
        full Cartesian product as a SeparableFiniteGroupAction. Otherwise falls back to the generic
        SeparableJointGroupAction wrapper
        """
        if isinstance(other, SeparableFiniteGroupAction):
            return _build_finite_separable_joint([self, other])
        return super().__mul__(other)


def _build_finite_separable_product(
    sub_groups: List[SeparableFiniteGroupAction],
) -> SeparableFiniteGroupAction:
    """
    Enumerates the full direct product of a list of SeparableFiniteGroupActions as a SeparableFiniteGroupAction whose
    non-identity elements are SeparableGroupActionElements combined per-side via ``&``
    """
    if len(sub_groups) == 0:
        raise ValueError('SeparableProductGroupAction requires at least one sub-group')

    input_dim = sum(g.input_dim for g in sub_groups)
    output_dim = sum(g.output_dim for g in sub_groups)
    sub_element_lists = [list(g.batch()) for g in sub_groups]
    all_tuples = list(itertools.product(*sub_element_lists))
    non_id_elements = [_separable_product_compose(tup) for tup in all_tuples[1:]]
    return SeparableFiniteGroupAction(non_id_elements, input_dim=input_dim, output_dim=output_dim)


def _build_finite_separable_joint(
    sub_groups: List[SeparableFiniteGroupAction],
) -> SeparableFiniteGroupAction:
    """
    Enumerates the full Cartesian product of a list of SeparableFiniteGroupActions as a SeparableFiniteGroupAction
    whose non-identity elements are SeparableGroupActionElements composed per-side via ``*``
    """
    if len(sub_groups) == 0:
        raise ValueError('SeparableJointGroupAction requires at least one sub-group')
    _validate_consistent_separable_dims(sub_groups)

    sub_element_lists = [list(g.batch()) for g in sub_groups]
    all_tuples = list(itertools.product(*sub_element_lists))
    non_id_elements = [_separable_joint_compose(tup) for tup in all_tuples[1:]]
    return SeparableFiniteGroupAction(
        non_id_elements,
        input_dim=sub_groups[0].input_dim,
        output_dim=sub_groups[0].output_dim,
    )
