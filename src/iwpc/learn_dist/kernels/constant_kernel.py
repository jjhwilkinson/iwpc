import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class ConstantKernel(TrainableKernelBase):
    """
    A constant kernel that only returns a single value
    """
    def __init__(
        self,
        cond_dimension: int,
        constant_value: ArrayLike,
    ):
        """
        Parameters
        ----------
        cond_dimension
            The dimension of the conditioning space
        constant_value
            The constant value. Must be a scalar, or 1D ArrayLike
        """
        constant_value = torch.as_tensor(constant_value, dtype=torch.float32)
        if constant_value.ndim == 0:
            constant_value = constant_value[None]
        if constant_value.ndim == 1:
            constant_value = constant_value[None, :]
        else:
            raise ValueError("Constant value must be a scalar or 1D array")

        super().__init__(constant_value.shape[0], cond_dimension)
        self.register_buffer("constant_value", constant_value)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            The log probability of each sample. Note that since the kernel has a constant value, this function should
            return log(1) if sample==self.constant_value and log(0) otherwise. However, for speed reasons, the check is
            skipped and log(1) is always returned for every sample. This may change in future
        """
        return torch.zeros(samples.shape[0], dtype=torch.float32, device=cond.device)

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information

        Returns
        -------
        Tensor
            cond.shape[0] copies of self.constant_value
        """
        return self.constant_value.repeat(cond.shape[0], 1)
