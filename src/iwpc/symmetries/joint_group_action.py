import itertools
from typing import List, Tuple

from torch.nn import ModuleList

from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement


class JointGroupAction(GroupAction):
    """
    A wrapper group action representing the joint action of a list of GroupActions on the same space R^dim. The
    elements of the joint action are the group multiplications g1 * g2 * ... * gN of one element drawn from each
    sub-group, applied right-to-left as

        [g1 * g2 * ... * gN](x) = g1(g2(...gN(x)))

    Nested JointGroupAction instances are automatically un-curried when constructed using the multiplication operator
    ``*`` or :py:meth:`merge`. Batches are drawn jointly by zipping the sub-group batches and composing each tuple.
    When all sub-groups are FiniteGroupActions the full Cartesian product is enumerated instead via the finite
    fast-path on :py:meth:`FiniteGroupAction.__mul__`

    All sub-groups must agree on ``dim`` since they act on the same space
    """

    def __init__(self, sub_groups: List[GroupAction]):
        """
        Parameters
        ----------
        sub_groups
            A list of GroupActions sharing the same vector space
        """
        if len(sub_groups) == 0:
            raise ValueError('JointGroupAction requires at least one sub-group')
        _validate_consistent_dims(sub_groups)

        super().__init__(dim=sub_groups[0].dim)
        self.sub_groups = ModuleList(sub_groups)

    def batch(self) -> Tuple[GroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[GroupActionElement, ...]
            A tuple of ComposedActionElements drawn by zipping a fresh batch from each sub-group. The batch length is
            the minimum of the sub-group batch lengths
        """
        sub_batches = [list(g.batch()) for g in self.sub_groups]
        return tuple(_joint_compose(tup) for tup in zip(*sub_batches))

    @classmethod
    def merge(cls, a: GroupAction, b: GroupAction) -> "JointGroupAction":
        """
        Constructs a JointGroupAction from a and b, splicing in the sub_groups of any operand that is itself a
        JointGroupAction so that nested joint actions are flattened into a single un-curried list
        """
        a_groups = list(a.sub_groups) if isinstance(a, JointGroupAction) else [a]
        b_groups = list(b.sub_groups) if isinstance(b, JointGroupAction) else [b]
        return JointGroupAction(a_groups + b_groups)


def _joint_compose(elements: Tuple[GroupActionElement, ...]) -> GroupActionElement:
    """
    Composes a tuple of GroupActionElements via group multiplication, automatically un-currying via the
    ComposedActionElement.merge factory
    """
    composed = elements[0]
    for e in elements[1:]:
        composed = composed * e
    return composed


def _validate_consistent_dims(sub_groups: List[GroupAction]) -> None:
    """
    Checks that every sub-group declares the same dim
    """
    dims = {g.dim for g in sub_groups}
    if len(dims) != 1:
        raise ValueError(
            f'All sub-groups of a JointGroupAction must agree on dim. Got dims={dims}'
        )


def _build_finite_joint(sub_groups: List[FiniteGroupAction]) -> FiniteGroupAction:
    """
    Enumerates the full Cartesian product of a list of FiniteGroupActions as a FiniteGroupAction whose non-identity
    elements are ComposedActionElements
    """
    if len(sub_groups) == 0:
        raise ValueError('JointGroupAction requires at least one sub-group')
    _validate_consistent_dims(sub_groups)

    sub_element_lists = [list(g.batch()) for g in sub_groups]
    all_tuples = list(itertools.product(*sub_element_lists))
    non_id_elements = [_joint_compose(tup) for tup in all_tuples[1:]]
    return FiniteGroupAction(non_id_elements, dim=sub_groups[0].dim)
