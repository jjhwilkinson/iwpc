from typing import Tuple, Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import Module

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class GaussianKernel(TrainableKernelBase):
    """
    A one dimensional Gaussian kernel with trainable mean and std deviations
    """
    def __init__(
        self,
        cond: Encoding | int,
        loc_model: Optional[Module] = None,
        scale_model: Optional[Module] = None,
        max_chi: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        cond
            The conditioning space encoding or dimension
        loc_model
            Optional model that constructs the mean of the distribution for the given conditioning information
        scale_model
            Optional model that constructs the std deviation of the distribution for the given conditioning information.
            Must provide positive values only, see ExponentialEncoding.
        max_chi
            Optional maximum chi value beyond which the log_prob is considered to be zero. Useful for suppressing the
            effects of outliers.
        """
        super().__init__(1, cond)
        self.register_buffer('log_two_pi', torch.tensor(0.5 * np.log(2 * np.pi), dtype=torch.float32))
        self.loc_model = basic_model_factory(cond, 1) if loc_model is None else loc_model
        self.scale_model = basic_model_factory(cond, ExponentialEncoding(1)) if scale_model is None else scale_model
        self.max_chi = max_chi

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Log probability of samples given the conditioning information
        """
        mean = self.loc_model(cond)
        sigma = self.scale_model(cond)
        chisq = ((samples - mean) / sigma) ** 2
        log_prob = - 0.5 * (self.log_two_pi + 2 * torch.log(sigma) + chisq)[:, 0]
        if self.max_chi is not None:
            log_prob[chisq[:, 0] > (self.max_chi ** 2)] = - torch.tensor(torch.inf)
        return log_prob

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            A sample from the gaussian kernel for each row of conditioning information
        """
        mean = self.loc_model(cond)
        sigma = self.scale_model(cond)
        return torch.normal(0, 1, size=(cond.shape[0], 1), dtype=torch.float32, device=cond.device) * sigma + mean

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        Tuple[Tensor, Tensor]
            A sample from the gaussian kernel for each row of conditioning information and its corresponding log
            probability efficiently implemented
        """
        mean = self.loc_model(cond)
        sigma = self.scale_model(cond)
        noise = torch.normal(0, 1, size=(cond.shape[0], 1), dtype=torch.float32, device=cond.device)
        with torch.no_grad():
            samples = noise * sigma + mean
        log_probs = -(0.5 * self.log_two_pi + 0.5 * ((samples - mean) / sigma)**2 + torch.log(sigma))[:, 0]
        return samples, log_probs
