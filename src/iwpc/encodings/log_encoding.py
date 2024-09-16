from typing import Optional

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class LogEncoding(Encoding):
    """
    An encoding that applies a logarithm to its inputs
    """
    def __init__(self, dimension: int, base: Optional[float] = -1):
        """
        Parameters
        ----------
        dimension
            The number of features to expect
        base
            The base of the logarithm. Defaults to the natural logarithm.
        """
        super().__init__(dimension, dimension)
        self.register_buffer('base', base)

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., dimension)

        Returns
        -------
        Tensor
            torch.log(x)
        """
        log_base = 1 if self.base < 0 else torch.log(self.base)
        return torch.log(x) / log_base
