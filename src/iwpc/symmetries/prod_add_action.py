from typing import Optional

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.symmetries.group_action_element import GroupActionElement


class ProdAddAction(GroupActionElement):
    """
    Group action element acting on R^dim by component-wise multiplying by a constant ``p`` and adding a constant ``q``,
    i.e. ``x -> p * x + q``. Unspecified ``prod`` defaults to ones, unspecified ``add`` defaults to zeros, so both
    buffers are always materialised at full ``dim``. Overrides ``*`` and ``&`` so that compositions of two
    ProdAddActions are themselves ProdAddActions, keeping the analytic form across composition
    """

    def __init__(
        self,
        prod: Optional[ArrayLike] = None,
        add: Optional[ArrayLike] = None,
        dim: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        prod
            An array-like with as many entries as ``dim``. Used as the multiplier constant. Defaults to ones if not
            provided
        add
            An array-like with as many entries as ``dim``. Used as the additive constant. Defaults to zeros if not
            provided
        dim
            The dimensionality of the vector space this element acts on. May be omitted if it can be inferred from
            the length of ``prod`` or ``add``
        """
        if dim is None:
            dim = _infer_dim(prod, add)
        if dim is None:
            raise ValueError('dim must be provided when neither prod nor add is supplied')
        super().__init__(dim=dim)

        self.register_buffer('prod', _materialise(prod, dim, fill=1.0))
        self.register_buffer('add', _materialise(add, dim, fill=0.0))

        is_id = bool(((self.prod == 1).all() & (self.add == 0).all()).item())
        self.is_identity = is_id

    def action(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            An input tensor with last dimension self.dim

        Returns
        -------
        Tensor
            ``x * self.prod[None, :] + self.add[None, :]``
        """
        return x * self.prod[None, :] + self.add[None, :]

    def __mul__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised group multiplication. When both operands are ProdAddActions with matching dims, the composition can
        be expressed analytically as a single ProdAddAction. For (a * b)(x) = a(b(x)) with a and b acting as p*x + q,
        the combined action is

            (a_p * b_p) * x + (a_p * b_a + a_a)

        Otherwise falls back to the generic ComposedActionElement composition

        Parameters
        ----------
        other
            A GroupActionElement to compose with

        Returns
        -------
        GroupActionElement
            A ProdAddAction if other is a ProdAddAction with matching dim, otherwise a ComposedActionElement
        """
        if isinstance(other, ProdAddAction) and self.dim == other.dim:
            return ProdAddAction(
                prod=self.prod * other.prod,
                add=self.prod * other.add + self.add,
                dim=self.dim,
            )
        return super().__mul__(other)

    def __and__(self, other: GroupActionElement) -> GroupActionElement:
        """
        Specialised direct product on disjoint dim ranges. When both operands are ProdAddActions, the product is
        itself a ProdAddAction whose prod and add buffers are concatenations of the operands' buffers. Otherwise falls
        back to the generic ProductActionElement product

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
                prod=torch.cat([self.prod, other.prod]),
                add=torch.cat([self.add, other.add]),
                dim=self.dim + other.dim,
            )
        return super().__and__(other)


def _materialise(arr: Optional[ArrayLike], dim: int, fill: float) -> Tensor:
    """
    Materialises a 1D buffer of length ``dim``, either from a provided array-like or filled with a constant

    Parameters
    ----------
    arr
        An optional 1D array-like of length ``dim``
    dim
        The expected length of the resulting buffer
    fill
        The constant to use when ``arr`` is None

    Returns
    -------
    Tensor
        A 1D tensor of shape ``(dim,)`` and dtype ``float``
    """
    if arr is None:
        return torch.full((dim,), fill, dtype=torch.float)
    return torch.as_tensor(arr, dtype=torch.float)


def _infer_dim(prod: Optional[ArrayLike], add: Optional[ArrayLike]) -> Optional[int]:
    """
    Infers a dim size from the lengths of ``prod`` and ``add``

    Parameters
    ----------
    prod
        An optional 1D array-like
    add
        An optional 1D array-like

    Returns
    -------
    Optional[int]
        The inferred dim length, or None if neither array-like is provided
    """
    for arr in (prod, add):
        if arr is None:
            continue
        return int(torch.as_tensor(arr).shape[-1])
    return None
