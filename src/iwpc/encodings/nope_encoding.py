import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class NopeEncoding(Encoding):
    """
    An encoding that returns no output features. Useful for masking inputs
    """
    def __init__(self, dimension) -> None:
        super().__init__(dimension, 0)

    def _encode(self, x) -> Tensor:
        """
        Returns
        -------
        Tensor
            A Tensor with the same batch size as the input, but no features
        """
        return torch.zeros((x.shape[0], 0), device=x.device)
