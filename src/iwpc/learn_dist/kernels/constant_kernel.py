from typing import Iterator

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class ConstantKernel(FiniteKernelInterface, TrainableKernelBase):
    """
    A constant kernel that only returns a single value
    """

    def __init__(
        self,
        constant_value: ArrayLike,
        cond_dimension: int | Encoding,
    ):
        """
        Parameters
        ----------
        constant_value
            The constant value. Must be a scalar, or 1D ArrayLike
        cond_dimension
            The dimension of the conditioning space
        """
        constant_value = torch.as_tensor(constant_value, dtype=torch.float32)
        if constant_value.ndim == 0:
            constant_value = constant_value[None]
        if constant_value.ndim == 1:
            constant_value = constant_value[None, :]
        else:
            raise ValueError("Constant value must be a scalar or 1D array")

        FiniteKernelInterface.__init__(self, 1)
        super(FiniteKernelInterface, self).__init__(constant_value.shape[1], cond_dimension)
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

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, 1) of zeros
        """
        return torch.zeros((cond.shape[0], 1), dtype=torch.float32, device=cond.device)

    def outcomes_iter(self) -> Iterator[Tensor]:
        """
        Returns
        -------
        Iterator[Tensor]
            An iterator over the single outcome of this kernel
        """
        yield self.constant_value[0]

    def outcome_to_idx(self, samples: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of samples of shape (N, self.sample_dimension)

        Returns
        -------
        Tensor
            A tensor of shape (N, 1) of zeros. Strictly speaking this should return raise an error for samples that are
            not equal to self.constant_value, but this check is skipped for speed reasons. This may change in future
        """
        return torch.zeros(samples.shape[0], dtype=torch.int, device=samples.device)

    def idx_to_outcome(self, idxs: Tensor) -> Tensor:
        """
        Parameters
        ----------
        idxs
            A tensor of indices of shape (N, 1)

        Returns
        -------
        Tensor
            self.constant_value repeated N times
        """
        return self.constant_value.repeat((idxs.shape[0], 1))
