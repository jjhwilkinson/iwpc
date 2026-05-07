from typing import Iterable

import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class RestructuringKernel(TrainableKernelBase):
    """
    Utility kernel for reordering and restructuring the conditioning information defined by a list of 'indices'.
    Samples from the kernel are simply comprised of the conditioning information evaluated at the given indices
    """
    def __init__(
        self,
        base_kernel: TrainableKernelBase,
        indices: Iterable[int],
        cond_dimension: int | Encoding,
    ):
        """
        Parameters
        ----------
        indices
            An iterable of ints that are used to restructure the conditioning information
        cond_dimension
            The dimension of the conditioning information
        """
        self.indices = list(indices)
        super().__init__(sample_dimension=len(self.indices), cond_dimension=cond_dimension)
        self.base_kernel = base_kernel

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A Tensor containing the conditioning information

        Returns
        -------
        Tensor
            cond[:, self.reorder_indices]
        """
        return self.base_kernel._draw(cond[:, self.indices])

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Since this is a deterministic tensor, a Tensor of zeros is returned. Strictly speaking, we should check that
        self.draw(cond) == samples, but this is skipped to increase performance

        Parameters
        ----------
        samples
            A Tensor of samples
        cond
            A Tensor of conditioning information

        Returns
        -------
        Tensor
            A Tensor of zeros
        """
        return self.base_kernel.log_prob(samples, cond[:, self.indices])

    def draw_with_log_prob(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Draws a sample from the base kernel using the restructured conditioning information and returns it alongside
        its log probability under the base kernel

        Parameters
        ----------
        cond
            A Tensor of conditioning information of shape (N, self.cond_dimension)

        Returns
        -------
        tuple[Tensor, Tensor]
            A tensor of samples of shape (N, self.sample_dimension) and a tensor of log probabilities of shape (N,)
        """
        return self.base_kernel.draw_with_log_prob(cond[:, self.indices])
