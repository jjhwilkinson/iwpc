import itertools
from typing import List, Tuple

from torch.nn import ModuleList

from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.group_action_element import GroupActionElement
from iwpc.symmetries.product_action_element import ProductActionElement


class ProductGroupAction(GroupAction):
    """
    A wrapper group action representing the direct product of a list of GroupActions acting on disjoint dim ranges of
    the input and output spaces. Given sub-groups with input dimensions d1, ..., dN and output dimensions o1, ..., oN,
    the product acts on a feature vector of length d1+...+dN by applying each sub-group's elements to the corresponding
    slice. Every sub-group must declare both input_dim and output_dim. Nested ProductGroupAction instances are
    automatically un-curried when constructed using the bitwise and operator, '&', or ProductGroupAction.merge.
    Batches are drawn jointly by zipping the sub-group batches. When all sub-groups are FiniteGroupActions the full
    direct product is enumerated instead via the finite fast-path on FiniteGroupAction.__and__
    """
    def __init__(self, sub_groups: List[GroupAction]):
        """
        Parameters
        ----------
        sub_groups
            A list of GroupActions to take the direct product of. Every sub-group must declare both input_dim and
            output_dim
        """
        if len(sub_groups) == 0:
            raise ValueError('ProductGroupAction requires at least one sub-group')

        super().__init__(
            input_dim=sum(g.input_dim for g in sub_groups),
            output_dim=sum(g.output_dim for g in sub_groups),
        )
        self.sub_groups = ModuleList(sub_groups)

    def batch(self) -> Tuple[GroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[GroupActionElement, ...]
            A tuple of ProductActionElements drawn by zipping a fresh batch from each sub-group. The batch length is
            the minimum of the sub-group batch lengths
        """
        sub_batches = [list(g.batch()) for g in self.sub_groups]
        return tuple(
            _product_compose(tup) for tup in zip(*sub_batches)
        )

    @classmethod
    def merge(cls, a: GroupAction, b: GroupAction) -> "ProductGroupAction":
        """
        Constructs a ProductGroupAction from a and b, splicing in the sub_groups of any operand that is itself a
        ProductGroupAction so that nested products are flattened into a single un-curried list

        Parameters
        ----------
        a
            A GroupAction
        b
            A GroupAction

        Returns
        -------
        ProductGroupAction
            The flattened direct product a & b
        """
        a_groups = list(a.sub_groups) if isinstance(a, ProductGroupAction) else [a]
        b_groups = list(b.sub_groups) if isinstance(b, ProductGroupAction) else [b]
        return ProductGroupAction(a_groups + b_groups)


def _product_compose(elements: Tuple[GroupActionElement, ...]) -> GroupActionElement:
    """
    Composes a tuple of GroupActionElements into a single ProductActionElement, automatically un-currying via the
    ProductActionElement.merge factory

    Parameters
    ----------
    elements
        A tuple of GroupActionElements

    Returns
    -------
    GroupActionElement
        A single (un-curried) ProductActionElement representing the direct product of the inputs
    """
    composed = elements[0]
    for e in elements[1:]:
        composed = composed & e
    return composed


def _build_finite_product(sub_groups: List[FiniteGroupAction]) -> FiniteGroupAction:
    """
    Enumerates the full direct product of a list of FiniteGroupActions as a FiniteGroupAction whose non-identity
    elements are ProductActionElements

    Parameters
    ----------
    sub_groups
        A list of FiniteGroupActions

    Returns
    -------
    FiniteGroupAction
        A FiniteGroupAction with |sub_groups[0]| * ... * |sub_groups[-1]| elements
    """
    if len(sub_groups) == 0:
        raise ValueError('ProductGroupAction requires at least one sub-group')

    input_dim = sum(g.input_dim for g in sub_groups)
    output_dim = sum(g.output_dim for g in sub_groups)
    sub_element_lists = [list(g.batch()) for g in sub_groups]
    all_tuples = list(itertools.product(*sub_element_lists))
    non_id_elements = [_product_compose(tup) for tup in all_tuples[1:]]
    return FiniteGroupAction(non_id_elements, input_dim=input_dim, output_dim=output_dim)
