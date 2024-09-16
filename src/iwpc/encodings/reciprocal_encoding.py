import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class ReciprocalEncoding(Encoding):
    """
    Encoding that returns the reciprocal of its input tensor.
    """
    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension
            The number of features to expect
        """
        super().__init__(dimension, dimension)

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., dimension)

        Returns
        -------
        Tensor
            1 / x
        """
        return 1 / x
