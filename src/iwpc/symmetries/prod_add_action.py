from typing import Optional

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.symmetries.finite_group_action import FiniteGroupAction
from iwpc.symmetries.group_action_element import GroupActionElement, InputSpaceInvariantException


class ProdAddAction(GroupActionElement):
    """
    Group action that acts by component-wise multiplying an element by a constant and then component-wise adding a
    constant for both the input and output space
    """

    def __init__(
        self,
        input_prod: ArrayLike,
        input_add: ArrayLike,
        output_prod: ArrayLike,
        output_add: ArrayLike,
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
        """
        super().__init__()
        self.register_buffer('input_prod', torch.as_tensor(input_prod, dtype=torch.float)[None, :])
        self.register_buffer('input_add', torch.as_tensor(input_add, dtype=torch.float)[None, :])
        self.register_buffer('output_prod', torch.as_tensor(output_prod, dtype=torch.float)[None, :])
        self.register_buffer('output_add', torch.as_tensor(output_add, dtype=torch.float)[None, :])

        if (self.input_prod == 1).all() and (self.input_add == 0).all():
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
        return x * self.input_prod + self.input_add

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Performs the specified action on the output space
        """
        return x * self.output_prod + self.output_add


class ProdAddZ2GroupAction(FiniteGroupAction):
    """
    Group action containing a single nontrivial element that is an involutary ProdAddAction. Note that the user is
    responsible for ensuring that the provided constants do not amount to an identity operation and that they in fact
    provide an involution. Note the add values need not be zero if the corresponding quantities are periodic, like an
    angle
    """
    def __init__(
        self,
        input_prod: Optional[ArrayLike] = None,
        input_add: Optional[ArrayLike] = None,
        output_prod: Optional[ArrayLike] = None,
        output_add: Optional[ArrayLike] = None,
    ):
        """
        Note for both the input and output space, at least one add or prod array is required to infer dimension

        Parameters
        ----------
        input_prod
            An array like with as many entries as the input space dimension. Used as the multiplier constant in the
            input space action. Defaults to an array of 1s
        input_add
            An array like with as many entries as the input space dimension. Used as the additive constant in the
            input space action. Defaults to an array of 0s
        output_prod
            An array like with as many entries as the output space dimension. Used as the multiplier constant in the
            output space action. Defaults to an array of 1s
        output_add
            An array like with as many entries as the output space dimension. Used as the additive constant in the
            output space action. Defaults to an array of 0s
        """
        if input_prod is None and input_add is None:
            raise ValueError("ProdAddZ2GroupAction requires at least one input prod/add array")
        if output_prod is None and output_prod is None:
            raise ValueError("ProdAddZ2GroupAction requires at least one output prod/add array")

        input_dim = len(input_prod) if input_add is None else len(input_add)
        output_dim = len(output_prod) if output_add is None else len(output_add)

        super().__init__([ProdAddAction(
            input_prod if input_prod is not None else torch.ones(input_dim),
            input_add if input_add is not None else torch.zeros(input_dim),
            output_prod if output_prod is not None else torch.ones(output_dim),
            output_add if output_add is not None else torch.zeros(output_dim),
        )])
