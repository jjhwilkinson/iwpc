from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from torch import Tensor
from torch.nn import Module, ModuleList

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.separable_group_action_element import SeparableGroupActionElement


class SeparableGroupAction(ABC, Module):
    """
    Abstract interface for a separable group action on the function space of maps R^input_dim -> R^output_dim. Each
    element is represented by a :py:class:`SeparableGroupActionElement` pairing an input-space vector action with an
    output-space vector action: [g.f](x) = output_action(f(input_action(x))). Provides :py:meth:`batch` for Haar
    averaging plus the :py:meth:`symmetrize` / :py:meth:`complement` model wrappers

    SeparableGroupActions support declarative composition via Python operators

    >>> product_group = G1 & G2  # direct product on disjoint dim ranges
    >>> joint_group = G1 * G2    # joint action on the same space

    Nested compositions are automatically un-curried, so G1 & G2 & G3 yields a single SeparableProductGroupAction with
    three sub-groups rather than a binary tree. When all operands are :py:class:`SeparableFiniteGroupAction`, the
    finite fast paths enumerate the full direct / Cartesian product instead
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
    def batch(self) -> Tuple[SeparableGroupActionElement, ...]:
        """
        Provides a batch of SeparableGroupActionElements sampled from the Haar measure of the group. Small finite
        groups should return all elements in every batch; larger and infinite groups return a fresh sample on every
        call

        Returns
        -------
        Tuple[SeparableGroupActionElement, ...]
        """

    def symmetrize(self, base_function: Callable[..., Tensor]) -> "SymmetrizedModel":
        """
        Wraps ``base_function`` in a :py:class:`SymmetrizedModel` so the result is invariant under this group action

        Returns
        -------
        SymmetrizedModel
        """
        from .symmetrized_model import SymmetrizedModel
        return SymmetrizedModel(self, base_function)

    def complement(self, base_function: Callable[..., Tensor]) -> "ComplementModel":
        """
        Wraps ``base_function`` in a :py:class:`ComplementModel` so the result lives in the orthogonal complement of
        the symmetrisation projection under this group action

        Returns
        -------
        ComplementModel
        """
        from .complement_model import ComplementModel
        return ComplementModel(self, base_function)

    def __and__(self, other: "SeparableGroupAction") -> "SeparableGroupAction":
        """
        Forms the direct product of two SeparableGroupActions acting on disjoint dim ranges. When both operands are
        :py:class:`SeparableFiniteGroupAction`, the full direct product is enumerated as |self| * |other|
        SeparableGroupActionElements. Otherwise the result is a :py:class:`SeparableProductGroupAction` zipping fresh
        batches from each sub-group
        """
        return SeparableProductGroupAction.merge(self, other)

    def __mul__(self, other: "SeparableGroupAction") -> "SeparableGroupAction":
        """
        Forms the joint action of two SeparableGroupActions on the same space. When both operands are
        :py:class:`SeparableFiniteGroupAction`, the full Cartesian product is enumerated as |self| * |other|
        SeparableGroupActionElements. Otherwise the result is a :py:class:`SeparableJointGroupAction` zipping fresh
        batches and composing each pair
        """
        return SeparableJointGroupAction.merge(self, other)

    @classmethod
    def from_pair_of_vector_groups(
        cls,
        input_group: GroupAction,
        output_group: GroupAction,
    ) -> "SeparableGroupAction":
        """
        Constructs a direct-product separable group action from two independent vector-space groups. At each
        :py:meth:`batch` call the two underlying groups are sampled independently and their elements are zipped into
        SeparableGroupActionElements. This represents the homomorphism G_in x G_out -> G_in x G_out — i.e. a separable
        action where the input and output sides are sampled independently

        Parameters
        ----------
        input_group
            A vector-space :py:class:`GroupAction` acting on the input space
        output_group
            A vector-space :py:class:`GroupAction` acting on the output space
        """
        return _PairedSeparableGroupAction(input_group=input_group, output_group=output_group)


class _PairedSeparableGroupAction(SeparableGroupAction):
    """
    A separable group action built from two independent vector-space groups acting on the input and output spaces
    respectively. ``batch()`` zips fresh batches from each
    """

    def __init__(self, input_group: GroupAction, output_group: GroupAction):
        super().__init__(input_dim=input_group.dim, output_dim=output_group.dim)
        self.input_group = input_group
        self.output_group = output_group

    def batch(self) -> Tuple[SeparableGroupActionElement, ...]:
        return tuple(
            SeparableGroupActionElement(input_action=in_elem, output_action=out_elem)
            for in_elem, out_elem in zip(self.input_group.batch(), self.output_group.batch())
        )


class SeparableProductGroupAction(SeparableGroupAction):
    """
    A wrapper representing the direct product of a list of SeparableGroupActions acting on disjoint dim ranges. Given
    sub-groups with input dims di_1, ..., di_N and output dims do_1, ..., do_N, the product acts on input vectors of
    length di_1+...+di_N and output vectors of length do_1+...+do_N. Batches are drawn jointly by zipping the
    sub-group batches and combining each tuple per-side via the vector-space ``&`` operator on input and output
    actions independently. Nested SeparableProductGroupAction instances are auto-flattened via :py:meth:`merge`
    """

    def __init__(self, sub_groups: List[SeparableGroupAction]):
        """
        Parameters
        ----------
        sub_groups
            A list of SeparableGroupActions to take the direct product of
        """
        if len(sub_groups) == 0:
            raise ValueError('SeparableProductGroupAction requires at least one sub-group')

        super().__init__(
            input_dim=sum(g.input_dim for g in sub_groups),
            output_dim=sum(g.output_dim for g in sub_groups),
        )
        self.sub_groups = ModuleList(sub_groups)

    def batch(self) -> Tuple[SeparableGroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[SeparableGroupActionElement, ...]
            A tuple of SeparableGroupActionElements drawn by zipping a fresh batch from each sub-group and combining
            each tuple per-side via ``&``
        """
        sub_batches = [list(g.batch()) for g in self.sub_groups]
        return tuple(_separable_product_compose(tup) for tup in zip(*sub_batches))

    @classmethod
    def merge(cls, a: SeparableGroupAction, b: SeparableGroupAction) -> "SeparableGroupAction":
        """
        Constructs a SeparableProductGroupAction from a and b, splicing in the sub_groups of any operand that is itself
        a SeparableProductGroupAction so that nested products are flattened. When both operands are
        :py:class:`SeparableFiniteGroupAction`, returns a SeparableFiniteGroupAction enumerating the full direct product
        instead
        """
        from iwpc.symmetries.separable_finite_group_action import SeparableFiniteGroupAction, _build_finite_separable_product
        if isinstance(a, SeparableFiniteGroupAction) and isinstance(b, SeparableFiniteGroupAction):
            return _build_finite_separable_product([a, b])

        a_groups = list(a.sub_groups) if isinstance(a, SeparableProductGroupAction) else [a]
        b_groups = list(b.sub_groups) if isinstance(b, SeparableProductGroupAction) else [b]
        return SeparableProductGroupAction(a_groups + b_groups)


class SeparableJointGroupAction(SeparableGroupAction):
    """
    A wrapper representing the joint action of a list of SeparableGroupActions on the same space. Batches are drawn
    jointly by zipping the sub-group batches and composing each tuple per-side via the vector-space ``*`` operator on
    input and output actions independently. Nested SeparableJointGroupAction instances are auto-flattened via
    :py:meth:`merge`. All sub-groups must agree on ``input_dim`` and ``output_dim``
    """

    def __init__(self, sub_groups: List[SeparableGroupAction]):
        """
        Parameters
        ----------
        sub_groups
            A list of SeparableGroupActions sharing the same input and output spaces
        """
        if len(sub_groups) == 0:
            raise ValueError('SeparableJointGroupAction requires at least one sub-group')
        _validate_consistent_separable_dims(sub_groups)

        super().__init__(
            input_dim=sub_groups[0].input_dim,
            output_dim=sub_groups[0].output_dim,
        )
        self.sub_groups = ModuleList(sub_groups)

    def batch(self) -> Tuple[SeparableGroupActionElement, ...]:
        """
        Returns
        -------
        Tuple[SeparableGroupActionElement, ...]
            A tuple of SeparableGroupActionElements drawn by zipping a fresh batch from each sub-group and composing
            each tuple per-side via ``*``
        """
        sub_batches = [list(g.batch()) for g in self.sub_groups]
        return tuple(_separable_joint_compose(tup) for tup in zip(*sub_batches))

    @classmethod
    def merge(cls, a: SeparableGroupAction, b: SeparableGroupAction) -> "SeparableGroupAction":
        """
        Constructs a SeparableJointGroupAction from a and b, splicing in the sub_groups of any operand that is itself a
        SeparableJointGroupAction so that nested joint actions are flattened. When both operands are
        :py:class:`SeparableFiniteGroupAction`, returns a SeparableFiniteGroupAction enumerating the full Cartesian
        product instead
        """
        from iwpc.symmetries.separable_finite_group_action import SeparableFiniteGroupAction, _build_finite_separable_joint
        if isinstance(a, SeparableFiniteGroupAction) and isinstance(b, SeparableFiniteGroupAction):
            return _build_finite_separable_joint([a, b])

        a_groups = list(a.sub_groups) if isinstance(a, SeparableJointGroupAction) else [a]
        b_groups = list(b.sub_groups) if isinstance(b, SeparableJointGroupAction) else [b]
        return SeparableJointGroupAction(a_groups + b_groups)


def _separable_product_compose(
    elements: Tuple[SeparableGroupActionElement, ...],
) -> SeparableGroupActionElement:
    """
    Combines a tuple of SeparableGroupActionElements via per-side ``&`` (direct product on disjoint dim ranges)
    """
    composed = elements[0]
    for e in elements[1:]:
        composed = composed & e
    return composed


def _separable_joint_compose(
    elements: Tuple[SeparableGroupActionElement, ...],
) -> SeparableGroupActionElement:
    """
    Combines a tuple of SeparableGroupActionElements via per-side ``*`` (group multiplication on the same space)
    """
    composed = elements[0]
    for e in elements[1:]:
        composed = composed * e
    return composed


def _validate_consistent_separable_dims(sub_groups: List[SeparableGroupAction]) -> None:
    """
    Checks that every sub-group declares the same input_dim and output_dim
    """
    input_dims = {g.input_dim for g in sub_groups}
    output_dims = {g.output_dim for g in sub_groups}
    if len(input_dims) != 1 or len(output_dims) != 1:
        raise ValueError(
            'All sub-groups of a SeparableJointGroupAction must agree on input_dim and output_dim. '
            f'Got input_dims={input_dims}, output_dims={output_dims}'
        )
