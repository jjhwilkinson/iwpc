from typing import Tuple

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Exponential

from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.trainable_kernel.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class TwoSidedExponentialKernel(TrainableKernelBase):
    def __init__(
        self,
        cond_dimension,
        init_loc,
        init_scale,
        loc_model = None,
        scale_model = None,
    ):
        super().__init__(1, cond_dimension)
        self.register_buffer("init_loc", torch.tensor(init_loc, dtype=torch.float32))
        self.register_buffer("init_scale", torch.tensor(init_scale, dtype=torch.float32))
        self.loc_model = basic_model_factory(cond_dimension, 1) if loc_model is None else loc_model
        self.scale_model = basic_model_factory(cond_dimension, ExponentialEncoding(1)) if scale_model is None else scale_model
        self.register_buffer('log_2', torch.tensor(np.log(2), dtype=torch.float32))
        self.exponential_dist = Exponential(rate=1.)

    def construct_loc(self, cond: Tensor) -> Tensor:
        return self.init_loc + self.loc_model(cond) / self.init_scale

    def construct_scale(self, cond: Tensor) -> Tensor:
        return self.scale_model(cond) * self.init_scale

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        loc = self.construct_loc(cond)
        scale = self.construct_scale(cond)
        return (torch.log(scale) - scale * (samples - loc).abs() - self.log_2)[:, 0]

    def _draw(self, cond: Tensor) -> Tensor:
        loc = self.construct_loc(cond)
        scale = self.construct_scale(cond)

        samples = self.exponential_dist.sample(sample_shape=(cond.shape[0], 1)).to(cond.device)
        samples = samples * (2 * (torch.rand_like(samples) > 0.5) - 1)
        return samples / scale + loc
