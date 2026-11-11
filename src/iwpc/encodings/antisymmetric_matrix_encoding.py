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
        # A Python int: reshaping with a tensor buffer reads it back from the device on every call
        self.dimension = int(dimension)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Loads as usual, dropping the `dimension` entry that state dicts saved when it was a buffer carry
        """
        state_dict.pop(f'{prefix}dimension', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

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
