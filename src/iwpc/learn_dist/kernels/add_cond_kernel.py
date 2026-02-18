from typing import Tuple

from torch import Tensor

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class AddCondKernel(TrainableKernelBase):
    """
    Wrapper for kernels that have the same conditioning and sample space. Samples are taken as the sum of the
    conditioning information and the samples from the base distribution. The base distribution can be thought of as
    providing a sample error or delta and the wrapper adds on the central value
    """
    def __init__(self, base_kernel: TrainableKernelBase):
        """
        Parameters
        ----------
        base_kernel
            A TrainableKernelBase that models the differences between the conditioning vectors and the target samples
        """
        if base_kernel.sample_dimension != base_kernel.cond_dimension:
            raise ValueError("AddCondKernel base kernel cond_dimension and sample_dimension must be the same")
        super().__init__(base_kernel.sample_dimension, base_kernel.cond_dimension)
        self.base_kernel = base_kernel

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            cond + a sample drawn from the base distribution
        """
        return cond + self.base_kernel.draw(cond)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A Tensor of samples
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            The log probability of each sample given the conditioning information
        """
        return self.base_kernel.log_prob(samples - cond, cond)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tuple[Tensor, Tensor]
            A sample for each row of conditioning information along with its associated log probability
        """
        samples, log_prob = self.base_kernel.draw_with_log_prob(cond)
        return cond + samples, log_prob
