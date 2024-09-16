import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class ExponentialEncoding(Encoding):
    """
    An encoding that exponentiates its inputs
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
            torch.exp(x)
        """
        return torch.exp(x)
