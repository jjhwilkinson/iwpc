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
        input_prod: Optional[ArrayLike] = None,
        input_add: Optional[ArrayLike] = None,
        output_prod: Optional[ArrayLike] = None,
        output_add: Optional[ArrayLike] = None,
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
