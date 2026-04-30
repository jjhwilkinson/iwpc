from typing import Optional

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class ProdAddAction(GroupActionElement):
    """
    Group action that acts by component-wise multiplying an element by a constant and then component-wise adding a
    constant for both the input and output space. Unspecified prod arrays default to ones and unspecified add arrays
    default to zeros so all four buffers are always materialised at full input/output dim. Overrides '*' and '&' so
    that compositions of two ProdAddActions are themselves ProdAddActions
    """

    def __init__(
        self,
        input_prod: Optional[ArrayLike] = None,
        input_add: Optional[ArrayLike] = None,
        output_prod: Optional[ArrayLike] = None,
        output_add: Optional[ArrayLike] = None,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        input_prod
            An array like with as many entries as the input space dimension. Used as the multiplier constant in the
            input space action. Defaults to ones if not provided
        input_add
            An array like with as many entries as the input space dimension. Used as the additive constant in the
            input space action. Defaults to zeros if not provided
        output_prod
            An array like with as many entries as the output space dimension. Used as the multiplier constant in the
            output space action. Defaults to ones if not provided
        output_add
            An array like with as many entries as the output space dimension. Used as the additive constant in the
            output space action. Defaults to zeros if not provided
        input_dim
            The dimensionality of the input space this element acts on. May be omitted if it can be inferred from the
            length of input_prod or input_add
        output_dim
            The dimensionality of the output space this element acts on. May be omitted if it can be inferred from the
            length of output_prod or output_add
        """
        if input_dim is None:
            input_dim = _infer_dim(input_prod, input_add)
        if output_dim is None:
            output_dim = _infer_dim(output_prod, output_add)
        if input_dim is None:
            raise ValueError('input_dim must be provided when neither input_prod nor input_add is supplied')
        if output_dim is None:
            raise ValueError('output_dim must be provided when neither output_prod nor output_add is supplied')
        super().__init__(input_dim=input_dim, output_dim=output_dim)

        self.register_buffer('input_prod', _materialise(input_prod, input_dim, fill=1.0))
        self.register_buffer('input_add', _materialise(input_add, input_dim, fill=0.0))
        self.register_buffer('output_prod', _materialise(output_prod, output_dim, fill=1.0))
        self.register_buffer('output_add', _materialise(output_add, output_dim, fill=0.0))

        affects = bool(((self.input_prod != 1).any() | (self.input_add != 0).any()).item())
        self.register_buffer('affects_input_space', torch.as_tensor(affects))

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Performs the specified action on the input space. Raises an InputSpaceInvariantException if the input
            space action is the identity
        """
        if not self.affects_input_space:
            raise InputSpaceInvariantException()
        return x * self.input_prod[None, :] + self.input_add[None, :]

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Performs the specified action on the output space
        """
        return x * self.output_prod[None, :] + self.output_add[None, :]

    def __mul__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised group multiplication. When both operands are ProdAddActions with matching dims, the composition
        can be expressed analytically as a single ProdAddAction. For (a * b)(x) = a(b(x)) with a and b acting as
        p*x + q on each space, the combined action is

            (a_p * b_p) * x + (a_p * b_a + a_a)

        Otherwise, falls back to the generic ComposedActionElement composition

        Parameters
        ----------
        other
            A GroupActionElement to compose with

        Returns
        -------
        GroupActionElement
            A ProdAddAction if other is a ProdAddAction with matching dims, otherwise a ComposedActionElement
        """
        if isinstance(other, ProdAddAction) and self.input_dim == other.input_dim and self.output_dim == other.output_dim:
            return ProdAddAction(
                input_prod=self.input_prod * other.input_prod,
                input_add=self.input_prod * other.input_add + self.input_add,
                output_prod=self.output_prod * other.output_prod,
                output_add=self.output_prod * other.output_add + self.output_add,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
        return super().__mul__(other)

    def __and__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised direct product on disjoint dim ranges. When both operands are ProdAddActions, the product is
        itself a ProdAddAction whose prod and add buffers are concatenations of the operands' buffers. Otherwise,
        falls back to the generic ProductActionElement product

        Parameters
        ----------
        other
            A GroupActionElement to take the direct product with

        Returns
        -------
        GroupActionElement
            A ProdAddAction if other is a ProdAddAction, otherwise a ProductActionElement
        """
        if isinstance(other, ProdAddAction):
            return ProdAddAction(
                input_prod=torch.cat([self.input_prod, other.input_prod]),
                input_add=torch.cat([self.input_add, other.input_add]),
                output_prod=torch.cat([self.output_prod, other.output_prod]),
                output_add=torch.cat([self.output_add, other.output_add]),
                input_dim=self.input_dim + other.input_dim,
                output_dim=self.output_dim + other.output_dim,
            )
        return super().__and__(other)


def _materialise(arr: Optional[ArrayLike], dim: int, fill: float) -> Tensor:
    """
    Returns a 1D buffer of length dim holding the contents of arr, or a constant-filled buffer when arr is None

    Parameters
    ----------
    arr
        An optional 1D array-like of length dim
    dim
        The expected length of the array
    fill
        The constant to use when arr is None

    Returns
    -------
    Tensor
        A 1D tensor of shape (dim,)
    """
    if arr is None:
        return torch.full((dim,), fill, dtype=torch.float)
    return torch.as_tensor(arr, dtype=torch.float)


def _infer_dim(prod: Optional[ArrayLike], add: Optional[ArrayLike]) -> Optional[int]:
    """
    Infers a dim size from the lengths of prod and add. Returns None if neither is provided

    Parameters
    ----------
    prod
        An optional array-like
    add
        An optional array-like

    Returns
    -------
    Optional[int]
        The inferred dim, or None if neither array-like is provided
    """
    for arr in (prod, add):
        if arr is None:
            continue
        return int(torch.as_tensor(arr).shape[-1])
    return None
