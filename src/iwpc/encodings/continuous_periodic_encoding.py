from typing import Tuple

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class ContinuousPeriodicEncoding(Encoding):
    """
    An encoding that provides a continuous parametrization of a 1D quantity such that all continuous functions on the
    encoded variables are automatically continuous and periodic on the given range. For example,
    if one were to use a NN to learn a continuous function on an angle, then, without an encoding the output of the
    network would be discontinuous on the 0 / 2*pi boundary as it is free to output whatever value it pleases for 0 and for
    2*pi. While the training examples might cause the network to converge onto a function that is almost continuous
    across the boundary, there is nothing forcing this property and so there will always be small errors which will
    worsen when evaluated on unseen data. To resolve such issues, this encoding can be used to encode the angle using
    (cos(theta), sin(theta)). There is a unique 1-1 mapping between such pairs and the original angle, so no information
    is lost, but any continuous 2D function on these variables is automatically a continuous as a function of theta
    including at the boundary

    """
    def __init__(self, range_: Tuple[float, float] = (-torch.pi, torch.pi)):
        """
        Parameters
        ----------
        range_
            The range over which the periodic quantity is defined. Defaults to -pi to pi.
        """
        super().__init__(1, 2)
        self.register_buffer('period', torch.tensor(range_[1] - range_[0]))

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., 1)

        Returns
        -------
        Tensor
            of dimension (..., 2)
        """
        angular_x = 2 * torch.pi * x / self.period
        return torch.concatenate([torch.cos(angular_x), torch.sin(angular_x)], dim=-1)
