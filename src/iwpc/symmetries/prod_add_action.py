from typing import Optional

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class ProdAddAction(GroupActionElement):
    """
    Group action that acts by component-wise multiplying an element by a constant and then component-wise adding a
    constant for both the input and output space
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
            input space action
        input_add
            An array like with as many entries as the input space dimension. Used as the additive constant in the
            input space action
        output_prod
            An array like with as many entries as the output space dimension. Used as the multiplier constant in the
            output space action
        output_add
            An array like with as many entries as the output space dimension. Used as the additive constant in the
            output space action
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

        if input_prod is not None:
            self.register_buffer('input_prod', torch.as_tensor(input_prod, dtype=torch.float)[None, :])
        else:
            self.input_prod = None

        if input_add is not None:
            self.register_buffer('input_add', torch.as_tensor(input_add, dtype=torch.float)[None, :])
        else:
            self.input_add = None

        if output_prod is not None:
            self.register_buffer('output_prod', torch.as_tensor(output_prod, dtype=torch.float)[None, :])
        else:
            self.output_prod = None

        if output_add is not None:
            self.register_buffer('output_add', torch.as_tensor(output_add, dtype=torch.float)[None, :])
        else:
            self.output_add = None

        if (self.input_prod is None or (self.input_prod == 1).all()) and (self.input_add is None or (self.input_add == 0).all()):
            self.register_buffer('affects_input_space', torch.as_tensor(False))
        else:
            self.register_buffer('affects_input_space', torch.as_tensor(True))

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Performs the specified action on the input space
        """
        if not self.affects_input_space:
            raise InputSpaceInvariantException()

        if self.input_prod is not None:
            x = x * self.input_prod
        if self.input_add is not None:
            x = x + self.input_add

        return x

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Performs the specified action on the output space
        """
        if self.output_prod is not None:
            x = x * self.output_prod
        if self.output_add is not None:
            x = x + self.output_add

        return x

    def __mul__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised group multiplication. When both operands are ProdAddActions, the composition can be expressed
        analytically as a single ProdAddAction, avoiding a generic ComposedActionElement wrapper. For
        (a * b)(x) = a(b(x)) with a and b acting as p*x + q on each space, the combined action is

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
            input_prod, input_add = _compose_prod_add(self.input_prod, self.input_add, other.input_prod, other.input_add)
            output_prod, output_add = _compose_prod_add(self.output_prod, self.output_add, other.output_prod, other.output_add)
            return ProdAddAction(
                input_prod=input_prod,
                input_add=input_add,
                output_prod=output_prod,
                output_add=output_add,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
            )
        return super().__mul__(other)

    def __and__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised direct product on disjoint dim ranges. When both operands are ProdAddActions, the product is itself
        a ProdAddAction whose prod and add arrays are concatenations of the operands' arrays (filling missing arrays
        with the identity values 1 and 0 of the appropriate length). Otherwise, falls back to the generic
        ProductActionElement product

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
                input_prod=_concat_prod(self.input_prod, self.input_dim, other.input_prod, other.input_dim),
                input_add=_concat_add(self.input_add, self.input_dim, other.input_add, other.input_dim),
                output_prod=_concat_prod(self.output_prod, self.output_dim, other.output_prod, other.output_dim),
                output_add=_concat_add(self.output_add, self.output_dim, other.output_add, other.output_dim),
                input_dim=self.input_dim + other.input_dim,
                output_dim=self.output_dim + other.output_dim,
            )
        return super().__and__(other)


def _compose_prod_add(
    a_prod: Optional[Tensor],
    a_add: Optional[Tensor],
    b_prod: Optional[Tensor],
    b_add: Optional[Tensor],
) -> tuple:
    """
    Computes the analytical composition of two prod-and-add actions on a single space. Given a(x) = a_p * x + a_a and
    b(x) = b_p * x + b_a (with None standing for the identity multiplier 1 or additive 0), returns the new (prod, add)
    pair such that a(b(x)) = new_prod * x + new_add. None is preserved wherever the result is the identity multiplier
    or additive so the underlying optimisations in ProdAddAction continue to apply

    Parameters
    ----------
    a_prod
        A buffer of shape (1, dim) or None
    a_add
        A buffer of shape (1, dim) or None
    b_prod
        A buffer of shape (1, dim) or None
    b_add
        A buffer of shape (1, dim) or None

    Returns
    -------
    tuple
        A pair of 1D Tensors or None values suitable for passing to ProdAddAction.__init__
    """
    new_prod = _multiply(a_prod, b_prod)
    scaled_b_add = _scale(a_prod, b_add)
    new_add = _add(scaled_b_add, a_add)
    return _squeeze(new_prod), _squeeze(new_add)


def _multiply(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    """
    Returns the elementwise product a * b, treating None as the multiplicative identity 1
    """
    if a is None:
        return b
    if b is None:
        return a
    return a * b


def _scale(prod: Optional[Tensor], add: Optional[Tensor]) -> Optional[Tensor]:
    """
    Returns prod * add where prod is interpreted as a multiplier (None means the identity 1) and add is interpreted as
    an additive constant (None means the identity 0). Returns None when add is None since 0 scaled by anything is 0
    """
    if add is None:
        return None
    if prod is None:
        return add
    return prod * add


def _add(a: Optional[Tensor], b: Optional[Tensor]) -> Optional[Tensor]:
    """
    Returns the elementwise sum a + b, treating None as the additive identity 0
    """
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _squeeze(t: Optional[Tensor]) -> Optional[Tensor]:
    """
    Squeezes a (1, dim)-shaped buffer to its 1D form for re-passing to ProdAddAction.__init__. None passes through
    unchanged
    """
    return None if t is None else t[0]


def _concat_prod(
    a: Optional[Tensor],
    a_dim: int,
    b: Optional[Tensor],
    b_dim: int,
) -> Optional[Tensor]:
    """
    Concatenates two prod buffers along the last dim, filling missing operands with ones of the appropriate length.
    Returns None if both operands are None so the identity-multiplier optimisation is preserved
    """
    if a is None and b is None:
        return None
    return torch.cat([_or_ones(a, a_dim), _or_ones(b, b_dim)], dim=-1)[0]


def _concat_add(
    a: Optional[Tensor],
    a_dim: int,
    b: Optional[Tensor],
    b_dim: int,
) -> Optional[Tensor]:
    """
    Concatenates two add buffers along the last dim, filling missing operands with zeros of the appropriate length.
    Returns None if both operands are None so the identity-additive optimisation is preserved
    """
    if a is None and b is None:
        return None
    return torch.cat([_or_zeros(a, a_dim), _or_zeros(b, b_dim)], dim=-1)[0]


def _or_ones(t: Optional[Tensor], dim: int) -> Tensor:
    """
    Returns t if it is not None, otherwise a (1, dim) tensor of ones matching t's expected shape
    """
    if t is None:
        return torch.ones((1, dim))
    return t


def _or_zeros(t: Optional[Tensor], dim: int) -> Tensor:
    """
    Returns t if it is not None, otherwise a (1, dim) tensor of zeros matching t's expected shape
    """
    if t is None:
        return torch.zeros((1, dim))
    return t


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
