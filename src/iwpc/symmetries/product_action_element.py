from typing import List

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleList

from iwpc.symmetries.group_action_element import GroupActionElement


class ProductActionElement(GroupActionElement):
    """
    A wrapper element acting on disjoint dim ranges of R^dim. Given sub-elements with dims d1, ..., dN, the product
    acts on a feature vector of length d1+...+dN by applying the j'th sub-element to the slice
    [d1+...+d_{j-1} : d1+...+d_j) and concatenating the results. Nested ProductActionElement instances are
    automatically un-curried when constructed using the bitwise and operator ``&`` or ``merge``. The product
    element is the identity iff every sub-element is the identity
    """

    def __init__(self, sub_elements: List[GroupActionElement]):
        """
        Parameters
        ----------
        sub_elements
            A list of GroupActionElements
        """
        if len(sub_elements) == 0:
            raise ValueError('ProductActionElement requires at least one sub-element')

        super().__init__(dim=sum(e.dim for e in sub_elements))
        self.sub_elements = ModuleList(sub_elements)
        self.register_buffer(
            'cum_dims',
            torch.tensor(np.cumsum([0] + [e.dim for e in sub_elements])).int(),
        )
        self.is_identity = all(e.is_identity for e in sub_elements)

    def action(self, x: Tensor) -> Tensor:
        """
        Slices x along the last dim by the cumulative dim edges, applies each sub-element's action to its slice, and
        concatenates the results

        Parameters
        ----------
        x
            A tensor of shape (..., dim)

        Returns
        -------
        Tensor
            A tensor of shape (..., dim) with each slice transformed by the corresponding sub-element
        """
        if x.shape[-1] != self.dim:
            raise ValueError(f'Expected input shape (..., {self.dim}), got {tuple(x.shape)}')

        return torch.concatenate([
            element.action(x[..., low:high])
            for element, low, high in zip(self.sub_elements, self.cum_dims[:-1], self.cum_dims[1:])
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
