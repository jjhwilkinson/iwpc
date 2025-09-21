import numpy as np
import torch
from torch import Tensor
from torch.distributions import Exponential

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class TwoSidedExponentialKernel(TrainableKernelBase):
    def __init__(
        self,
        cond: Encoding | int,
        loc_model = None,
        scale_model = None,
    ):
        super().__init__(1, cond)
        self.loc_model = basic_model_factory(cond, 1) if loc_model is None else loc_model
        self.scale_model = basic_model_factory(cond, ExponentialEncoding(1)) if scale_model is None else scale_model
        self.register_buffer('log_2', torch.tensor(np.log(2), dtype=torch.float32))
        self.exponential_dist = Exponential(rate=1.)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        loc = self.loc_model(cond)
        scale = self.scale_model(cond)
        return (torch.log(scale) - scale * (samples - loc).abs() - self.log_2)[:, 0]

    def _draw(self, cond: Tensor) -> Tensor:
        loc = self.loc_model(cond)
        scale = self.scale_model(cond)

        samples = self.exponential_dist.sample(sample_shape=(cond.shape[0], 1)).to(cond.device)
        samples = samples * (2 * (torch.rand_like(samples) > 0.5) - 1)
        return samples / scale + loc
