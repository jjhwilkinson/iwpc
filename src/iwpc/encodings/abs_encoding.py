import torch

from iwpc.encodings.encoding_base import Encoding


class AbsEncoding(Encoding):
    """
    Encoding that returns the absolute value of a feature
    """
    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension
            The number of features to expect
        """
        super().__init__(dimension, dimension)

    def _encode(self, x):
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., dimension)

        Returns
        -------
        Tensor
            torch.abs(x)
        """
        return torch.abs(x)
