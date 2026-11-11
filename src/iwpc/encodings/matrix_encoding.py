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

        # Python ints: reshaping with tensor buffers reads them back from the device on every call
        self.dimension = int(dimension)
        self.dimension2 = int(dimension2)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        """
        Loads as usual, dropping the `dimension` and `dimension2` entries that state dicts saved when they were buffers carry
        """
        for name in ('dimension', 'dimension2'):
            state_dict.pop(f'{prefix}{name}', None)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)


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
