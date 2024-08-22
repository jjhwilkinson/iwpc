from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class TrivialEncoding(Encoding):
    """
    An encoding which does nothing to its inputs. Useful when building a concatenated encoding layer
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
            The same tensor, x
        """
        return x
