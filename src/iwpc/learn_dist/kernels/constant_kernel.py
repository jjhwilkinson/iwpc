import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class ConstantKernel(TrainableKernelBase):
    def __init__(
        self,
        cond: int,
        constant_value: ArrayLike,
    ):
        constant_value = torch.as_tensor(constant_value, dtype=torch.float32)
        if constant_value.ndim == 0:
            constant_value = constant_value[None]
        if constant_value.ndim == 1:
            constant_value = constant_value[None, :]
        super().__init__(constant_value.shape[0], cond)
        self.register_buffer("constant_value", constant_value)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        return torch.zeros(samples.shape[0], dtype=torch.float32, device=cond.device)

    def _draw(self, cond: Tensor) -> Tensor:
        return self.constant_value.repeat(cond.shape[0], 1)
