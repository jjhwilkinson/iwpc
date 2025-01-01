from typing import Tuple

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class PeriodicEncoding(Encoding):
    """
    An encoding that provides a continuous parametrization of a 1D quantity such that all continuous functions on the
    encoded variables are automatically periodic on the given range, excluding the boundary where the function may be
    discontinuous. The periodicity is enforced by applying the appropriate modulus to the input
    """
    def __init__(self, range_: Tuple[float, float]):
        """
        Parameters
        ----------
        range_
            The range over which the periodic quantity is defined
        """
        super().__init__(1, 1)
        self.register_buffer('range', torch.tensor(range_))

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., 1)

        Returns
        -------
        Tensor
            of dimension (..., 1)
        """

        return (x - self.range[0]) % (self.range[1] - self.range[0]) + self.range[0]
