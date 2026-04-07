from torch import Tensor
import torch
from iwpc.encodings.encoding_base import Encoding
from typing import Optional


class WrappedAngleEncoding(Encoding):
    """
    An encoding that maps each feature to the range [-pi, pi] in a circular manner. 
    This is useful for encoding angles, where values of -pi and pi represent the same angle and we want to avoid discontinuities in the encoding space.
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
            A tensor of the same shape as x where each feature is mapped to the range [-pi, pi] in a circular manner.
        """
        return torch.atan2(torch.sin(x), torch.cos(x))
