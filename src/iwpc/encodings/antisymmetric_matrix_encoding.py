from typing import Optional

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class AntiSymmetricMatrixEncoding(Encoding):
    """
    An encoding that reshapes its input vector into an antisymmetric square matrix of the specified shape.
    """
    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension
            The dimension of the output square matrix
        """
        super().__init__(dimension * dimension, [dimension, dimension])
        self.register_buffer('dimension', torch.as_tensor(dimension))

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A square tensor of dimension (..., dimension * dimension)

        Returns
        -------
        Tensor
            A antisymmetrised square tensor of shape (..., dimension, dimension)
        """
        mat = x.reshape((*x.shape[:-1], self.dimension, self.dimension))
        return 0.5 * (mat - mat.transpose(-1, -2))
