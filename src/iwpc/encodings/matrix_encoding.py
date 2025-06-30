from typing import Optional

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class MatrixEncoding(Encoding):
    """
    An encoding that reshapes its input vector into a matrix of the specified shape.
    """
    def __init__(self, dimension: int, dimension2: Optional[int] = None):
        """
        Parameters
        ----------
        dimension
            The first dimension of the output matrix
        dimension2
            The second dimension of the output matrix. Defaults to dimension
        """
        dimension2 = dimension if dimension2 is None else dimension2
        super().__init__(dimension * dimension2, [dimension, dimension2])

        self.register_buffer('dimension', torch.as_tensor(dimension))
        self.register_buffer('dimension2', torch.as_tensor(dimension2))


    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., dimension * dimension2)

        Returns
        -------
        Tensor
            A tensor of shape (..., dimension, dimension2)
        """
        return x.reshape((*x.shape[:-1], self.dimension, self.dimension2))
