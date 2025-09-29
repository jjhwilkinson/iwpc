import torch

from iwpc.encodings.encoding_base import Encoding


class SignEncoding(Encoding):
    """
    Encoding that returns the sign of its input
    """
    def __init__(self, dimension: int):
        """
        Parameters
        ----------
        dimension
            The number of features to expect
        """
        super().__init__(dimension, dimension)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the sign of the input

        Parameters
        ----------
        x
            An input Tensor of shape (N, dimension)

        Returns
        -------
        Tensor
            The sign of each entry in the input tensor
        """
        return x.sign()
