from typing import Tuple

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class SphericalUnitVectorEncoding(Encoding):
    """
    An encoding that expects as input the polar and azimuthal angle of a unit vector and returns the corresponding
    normalised 3D unit vector
    """

    def __init__(self):
        super().__init__(2, 3)

    def _encode(self, x: Tensor) -> Tensor:
        """
        Parameters
        ----------
        x
            A tensor of dimension (..., 2). x[..., 0] must be the polar angle, and x[..., 1] must be azimuthal angle

        Returns
        -------
        Tensor
            of dimension (..., 3) containing the 3D unit vectors corresponding to the polar angle and azimuthal angles
        """
        cos_theta = torch.cos(x[:, 0])
        sin_theta = torch.sin(x[:, 0])
        return torch.stack([sin_theta * torch.cos(x[:, 1]), sin_theta * torch.sin(x[:, 1]), cos_theta], dim=-1)
