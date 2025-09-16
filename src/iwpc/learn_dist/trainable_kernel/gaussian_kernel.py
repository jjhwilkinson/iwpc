from typing import Tuple

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.trainable_kernel.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class GaussianKernel(TrainableKernelBase):
    def __init__(
        self,
        cond_dimension,
        init_mean,
        init_std,
        mean_model = None,
        std_model = None,
    ):
        super().__init__(1, cond_dimension)
        self.register_buffer("init_mean", torch.tensor(init_mean, dtype=torch.float32))
        self.register_buffer("init_std", torch.tensor(init_std, dtype=torch.float32))
        self.register_buffer('log_two_pi', torch.tensor(np.log(2 * np.pi), dtype=torch.float32))
        self.mean_model = basic_model_factory(cond_dimension, 1) if mean_model is None else mean_model
        self.std_model = basic_model_factory(cond_dimension, ExponentialEncoding(1)) if std_model is None else std_model

    def construct_mean(self, cond: Tensor) -> Tensor:
        return self.init_mean + self.mean_model(cond) * self.init_std

    def construct_std(self, cond: Tensor) -> Tensor:
        return self.std_model(cond) * self.init_std

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        mean = self.construct_mean(cond)
        sigma = self.construct_std(cond)
        chisq = ((samples - mean) / sigma) ** 2
        return - 0.5 * (self.log_two_pi + 2 * torch.log(sigma) + chisq)[:, 0]

    def _draw(self, cond: Tensor) -> Tensor:
        mean = self.construct_mean(cond)
        sigma = self.construct_std(cond)
        return torch.normal(0, 1, size=(cond.shape[0], 1), dtype=torch.float32, device=cond.device) * sigma + mean

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        mean = self.construct_mean(cond)
        sigma = self.construct_std(cond)
        noise = torch.normal(0, 1, size=(cond.shape[0], 1), dtype=torch.float32, device=cond.device)
        with torch.no_grad():
            samples = noise * sigma + mean
        probs = - 0.5 * (self.log_two_pi + torch.log(sigma) + noise**2)[:, 0]
        return samples, probs
