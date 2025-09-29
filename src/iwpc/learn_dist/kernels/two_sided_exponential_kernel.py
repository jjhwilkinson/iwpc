from typing import Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Exponential
from torch.nn import Module

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class TwoSidedExponentialKernel(TrainableKernelBase):
    """
    A one dimensional two-sided Exponential kernel with a trainable scale parameter
    """
    def __init__(
        self,
        cond: Encoding | int,
        loc_model: Optional[Module] = None,
        scale_model: Optional[Module] = None,
    ):
        """
        Parameters
        ----------
        cond
            The conditioning space encoding or dimension
        loc_model
            Optional model that constructs the mean of the distribution for the given conditioning information
        scale_model
            Optional model that constructs the scale parameter of the distribution for the given conditioning information.
            Must provide positive values only, see ExponentialEncoding.
        """
        super().__init__(1, cond)
        self.loc_model = basic_model_factory(cond, 1) if loc_model is None else loc_model
        self.scale_model = basic_model_factory(cond, ExponentialEncoding(1)) if scale_model is None else scale_model
        self.register_buffer('log_2', torch.tensor(np.log(2), dtype=torch.float32))
        self.exponential_dist = Exponential(rate=1.)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Log probability of samples given the conditioning information
        """
        loc = self.loc_model(cond)
        scale = self.scale_model(cond)
        return (torch.log(scale) - scale * (samples - loc).abs() - self.log_2)[:, 0]

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            A sample from the two-sided exponential kernel for each row of conditioning information
        """
        loc = self.loc_model(cond)
        scale = self.scale_model(cond)

        samples = self.exponential_dist.sample(sample_shape=(cond.shape[0], 1)).to(cond.device)
        samples = samples * (2 * (torch.rand_like(samples) > 0.5) - 1)
        return samples / scale + loc

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        Tuple[Tensor, Tensor]
            A sample from the two-sided Exponential kernel for each row of conditioning information and its
            corresponding log probability efficiently implemented
        """
        loc = self.loc_model(cond)
        scale = self.scale_model(cond)

        with torch.no_grad():
            samples = self.exponential_dist.sample(sample_shape=(cond.shape[0], 1)).to(cond.device)
            samples = samples * (2 * (torch.rand_like(samples) > 0.5) - 1)
            samples = samples / scale + loc
        return samples, (torch.log(scale) - scale * (samples - loc).abs() - self.log_2)[:, 0]
