from typing import List

from torch import Tensor
from torch.nn import ModuleList

from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class ComposedActionElement(GroupActionElement):
    """
    A wrapper element representing the group multiplication of a list of GroupActionElements. The action on both the
    input and output spaces is the right-to-left composition of each sub-element's action, that is

    [g1 * g2 * ... * gN](x) = g1(g2(...gN(x)))

    Nested ComposedActionElement instances are automatically un-curried when constructed using the multiplication
    operator, '*', or ComposedActionElement.merge. If every sub-element raises InputSpaceInvariantException for the
    input space action, the composed element also raises InputSpaceInvariantException so callers can re-use model
    evaluations
    """
    def __init__(self, sub_elements: List[GroupActionElement]):
        """
        Parameters
        ----------
        sub_elements
            A list of GroupActionElements to compose. The composition is applied right-to-left
        """
        if len(sub_elements) == 0:
            raise ValueError('ComposedActionElement requires at least one sub-element')

        input_dims = [e.input_dim for e in sub_elements]
        output_dims = [e.output_dim for e in sub_elements]
        super().__init__(
            input_dim=input_dims[-1] if all(d is not None and d == input_dims[-1] for d in input_dims) else None,
            output_dim=output_dims[0] if all(d is not None and d == output_dims[0] for d in output_dims) else None,
        )
        self.sub_elements = ModuleList(sub_elements)

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Composes the input space actions of the sub-elements right-to-left. Sub-elements that raise
        InputSpaceInvariantException are treated as identity for that step. If every sub-element raises
        InputSpaceInvariantException, the exception is propagated

        Parameters
        ----------
        x
            An input tensor in R^M

        Returns
        -------
        Tensor
            The composed action applied to x
        """
        any_acted = False
        for element in reversed(self.sub_elements):
            try:
                x = element.input_space_action(x)
                any_acted = True
            except InputSpaceInvariantException:
                continue
        if not any_acted:
            raise InputSpaceInvariantException()
        return x

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Composes the output space actions of the sub-elements right-to-left

        Parameters
        ----------
        x
            An input tensor of output values in R^N

        Returns
        -------
        Tensor
            The composed action applied to x
        """
        for element in reversed(self.sub_elements):
            x = element.output_space_action(x)
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
