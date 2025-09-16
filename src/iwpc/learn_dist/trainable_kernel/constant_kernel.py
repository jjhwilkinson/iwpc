import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.learn_dist.trainable_kernel.trainable_kernel_base import TrainableKernelBase


class ConstantKernel(TrainableKernelBase):
    def __init__(
        self,
        constant_value: ArrayLike,
        cond_dimension,
    ):
        constant_value = torch.as_tensor(constant_value)
        super().__init__(constant_value.shape[0], cond_dimension)
        self.register_buffer("constant_value", constant_value)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        return torch.zeros(samples.shape[0], dtype=torch.float32, device=cond.device)

    def _draw(self, cond: Tensor) -> Tensor:
        return self.sample_dimension[None, :].repeat(cond.shape[0], 1)
