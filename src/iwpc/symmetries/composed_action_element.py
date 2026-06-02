from typing import List

from torch import Tensor
from torch.nn import ModuleList

from iwpc.symmetries.group_action_element import GroupActionElement


class ComposedActionElement(GroupActionElement):
    """
    A wrapper element representing the group multiplication of a list of GroupActionElements on R^dim. The action is
    the right-to-left composition of each sub-element's action, that is

        [g1 * g2 * ... * gN](x) = g1(g2(...gN(x)))

    Nested ComposedActionElement instances are automatically un-curried when constructed using the multiplication
    operator ``*`` or ``merge``. The composed element is the identity iff every sub-element is the identity
    """

    def __init__(self, sub_elements: List[GroupActionElement]):
        """
        Parameters
        ----------
        sub_elements
            A list of GroupActionElements to compose. The composition is applied right-to-left. All sub-elements must
            share the same ``dim``
        """
        if len(sub_elements) == 0:
            raise ValueError('ComposedActionElement requires at least one sub-element')

        dims = {e.dim for e in sub_elements}
        if len(dims) != 1:
            raise ValueError(
                f'All sub-elements of a ComposedActionElement must agree on dim. Got dims={dims}'
            )

        super().__init__(dim=sub_elements[0].dim)
        self.sub_elements = ModuleList(sub_elements)
        self.is_identity = all(e.is_identity for e in sub_elements)

    def action(self, x: Tensor) -> Tensor:
        """
        Composes the actions of the sub-elements right-to-left

        Parameters
        ----------
        x
            An input tensor with last dimension self.dim

        Returns
        -------
        Tensor
            The composed action applied to x
        """
        for element in reversed(self.sub_elements):
            x = element.action(x)
        return x

    @classmethod
    def merge(cls, a: GroupActionElement, b: GroupActionElement) -> "ComposedActionElement":
        """
        Constructs a ComposedActionElement from a and b. If either is itself a ComposedActionElement, its sub_elements
        are spliced in so that nested compositions are flattened into a single un-curried list

        Parameters
        ----------
        a
            A GroupActionElement
        b
            A GroupActionElement

        Returns
        -------
        ComposedActionElement
            The flattened composition a * b
        """
        a_elements = list(a.sub_elements) if isinstance(a, ComposedActionElement) else [a]
        b_elements = list(b.sub_elements) if isinstance(b, ComposedActionElement) else [b]
        return ComposedActionElement(a_elements + b_elements)
