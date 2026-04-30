from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleList

from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class ProductActionElement(GroupActionElement):
    """
    A wrapper element acting on disjoint dim ranges of the input and output spaces. Given sub-elements with input
    dimensions d1, ..., dN and output dimensions o1, ..., oN, the product acts on a feature vector of length d1+...+dN
    by applying the j'th sub-element to the slice [d1+...+d_{j-1} : d1+...+d_j) and concatenating the results, and
    likewise for the output space. Nested ProductActionElement instances are automatically un-curried when constructed
    using the bitwise and operator, '&', or ProductActionElement.merge. If every sub-element raises
    InputSpaceInvariantException for the input space action, the product element also raises
    InputSpaceInvariantException so callers can re-use model evaluations
    """
    def __init__(self, sub_elements: List[GroupActionElement]):
        """
        Parameters
        ----------
        sub_elements
            A list of GroupActionElements. Every sub-element must declare both input_dim and output_dim
        """
        if len(sub_elements) == 0:
            raise ValueError('ProductActionElement requires at least one sub-element')

        super().__init__(
            input_dim=sum(e.input_dim for e in sub_elements),
            output_dim=sum(e.output_dim for e in sub_elements),
        )
        self.sub_elements = ModuleList(sub_elements)
        self.register_buffer(
            'cum_input_dims',
            torch.tensor(np.cumsum([0] + [e.input_dim for e in sub_elements])).int(),
        )
        self.register_buffer(
            'cum_output_dims',
            torch.tensor(np.cumsum([0] + [e.output_dim for e in sub_elements])).int(),
        )

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Slices x along the last dim by the cumulative input dim edges, applies each sub-element's input space action to
        its slice, and concatenates the results. Sub-elements that raise InputSpaceInvariantException pass their slice
        through unchanged. If every sub-element raises InputSpaceInvariantException, the exception is propagated

        Parameters
        ----------
        x
            A tensor of shape (..., input_dim)

        Returns
        -------
        Tensor
            A tensor of shape (..., input_dim) with each slice transformed by the corresponding sub-element
        """
        if x.shape[-1] != self.input_dim:
            raise ValueError(f'Expected input shape (..., {self.input_dim}), got {tuple(x.shape)}')

        slices = []
        any_acted = False
        for element, low, high in zip(self.sub_elements, self.cum_input_dims[:-1], self.cum_input_dims[1:]):
            sub_slice = x[..., low:high]
            try:
                slices.append(element.input_space_action(sub_slice))
                any_acted = True
            except InputSpaceInvariantException:
                slices.append(sub_slice)
        if not any_acted:
            raise InputSpaceInvariantException()
        return torch.concatenate(slices, dim=-1)

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Slices x along the last dim by the cumulative output dim edges, applies each sub-element's output space action
        to its slice, and concatenates the results

        Parameters
        ----------
        x
            A tensor of shape (..., output_dim)

        Returns
        -------
        Tensor
            A tensor of shape (..., output_dim) with each slice transformed by the corresponding sub-element
        """
        if x.shape[-1] != self.output_dim:
            raise ValueError(f'Expected input shape (..., {self.output_dim}), got {tuple(x.shape)}')

        return torch.concatenate([
            element.output_space_action(x[..., low:high])
            for element, low, high in zip(self.sub_elements, self.cum_output_dims[:-1], self.cum_output_dims[1:])
        ], dim=-1)

    @classmethod
    def merge(cls, a: GroupActionElement, b: GroupActionElement) -> "ProductActionElement":
        """
        Constructs a ProductActionElement from a and b. If either is itself a ProductActionElement, its sub_elements
        are spliced in so that nested products are flattened into a single un-curried list

        Parameters
        ----------
        a
            A GroupActionElement
        b
            A GroupActionElement

        Returns
        -------
        ProductActionElement
            The flattened direct product a & b
        """
        a_elements = list(a.sub_elements) if isinstance(a, ProductActionElement) else [a]
        b_elements = list(b.sub_elements) if isinstance(b, ProductActionElement) else [b]
        return ProductActionElement(a_elements + b_elements)
