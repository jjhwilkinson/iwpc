from typing import Tuple

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class ConditionedKernel(TrainableKernelBase):
    """
    Utility kernel that handles the merging of two kernels by concatenating their outputs, and wherein one kernel, the
    'sample_kernel' is conditioned on the output of the `conditioning_kernel` in addition to the shared conditioning
    information. Useful for implementing kernels of the form p(x, y | z) = p(y | x, z) p(x | z). The 'or' operation can
    be used as a shorthand: `sample_kernel | conditioning_kernel`
    """
    def __init__(self, sample_kernel: TrainableKernelBase, conditioning_kernel: TrainableKernelBase):
        """
        Parameters
        ----------
        sample_kernel
            A kernel that satisfies sample_kernel.cond_dimension == conditioning_kernel.cond_dimension + conditioning_kernel.sample_dimension
            the kernel should expect the first sample_kernel.cond_dimension components of its conditioning information
            to originate from the conditioning_kernel
        conditioning_kernel
            A kernel that produces samples upon which the sample kernel above is additionally conditioned
        """
        assert sample_kernel.cond_dimension == (conditioning_kernel.sample_dimension + conditioning_kernel.cond_dimension)
        super().__init__(
            sample_kernel.sample_dimension + conditioning_kernel.sample_dimension,
            conditioning_kernel.cond_dimension
        )

        self.sample_kernel = sample_kernel
        self.conditioning_kernel = conditioning_kernel

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        The log probability of the samples given the conditioning information. Calculated using p(x, y | z) = p(y | x, z) p(x | z)

        Parameters
        ----------
        samples
            The samples for which the log probability should be calculated. Should have shape (N, self.sample_dimension)
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            The log probability of each samples given the conditioning information with shape (N,)
        """
        sample_kernel_samples = samples[:, :self.sample_kernel.sample_dimension]
        conditioning_kernel_samples = samples[:, self.sample_kernel.sample_dimension:]

        sample_kernel_conditioning_info = torch.concat([
            conditioning_kernel_samples,
            cond,
        ], dim=-1)

        return (
            self.sample_kernel.log_prob(sample_kernel_samples, sample_kernel_conditioning_info)
            + self.conditioning_kernel.log_prob(conditioning_kernel_samples, cond)
        )

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Draws samples from sample_kernel and conditioning_kernel and concatenates their output

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A sample for each row of conditioning information with shape (N, self.sample_dimension). The first
            sample_kernel.sample_dimension components correspond to the samples produced by sample_kernel
        """
        conditioning_kernel_samples = self.conditioning_kernel.draw(cond)
        sample_kernel_conditioning_info = torch.concat([
            conditioning_kernel_samples,
            cond,
        ], dim=-1)

        return torch.cat([
            self.sample_kernel.draw(sample_kernel_conditioning_info),
            conditioning_kernel_samples,
        ], dim=-1)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Draws samples from sample_kernel and conditioning_kernel and concatenates their output. Also returns the log
        probability of said samples, see log_prob docstring

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension) and its
            corresponding log_probability. See log_prob and _draw docstrings
        """
        conditioning_kernel_samples, conditioning_kernel_log_prob = self.conditioning_kernel.draw_with_log_prob(cond)
        sample_kernel_conditioning_info = torch.concat([
            conditioning_kernel_samples,
            cond,
        ], dim=-1)
        sample_kernel_samples, sample_kernel_log_prob = self.sample_kernel.draw_with_log_prob(sample_kernel_conditioning_info)

        return torch.cat([
            sample_kernel_samples,
            conditioning_kernel_samples,
        ], dim=-1), sample_kernel_log_prob + conditioning_kernel_log_prob
