from typing import Tuple, List

import numpy as np
import torch
from torch import Tensor
from torch.nn import ModuleList

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class ConcatenatedKernel(TrainableKernelBase):
    """
    Utility kernel that merges any number of sub-kernels to produce samples that are concatenations of samples drawn
    from the sub-kernels. Since samples are drawn independently, the log probability of each sample can be calculated
    automatically as an independent sum
    """
    def __init__(self, sub_kernels: List[TrainableKernelBase], concatenate_cond=False):
        """
        Parameters
        ----------
        sub_kernels
            A list of TrainableKernelBase sub-kernels
        concatenate_cond
            Whether the conditioning information spaced should be concatenated, or are the same for all sub-kernels
        """
        assert concatenate_cond or all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        cond_dimension = sum(k.cond_dimension for k in sub_kernels) if concatenate_cond else sub_kernels[0].cond_dimension
        TrainableKernelBase.__init__(self, sum(k.sample_dimension for k in sub_kernels), cond_dimension)

        self.sub_kernels = ModuleList(sub_kernels)
        self.concatenate_cond = concatenate_cond
        cum_sample_sizes = np.cumsum([0] + [k.sample_dimension for k in sub_kernels])
        self.sample_edges = [slice(cum_sample_sizes[i], cum_sample_sizes[i+1]) for i in range(len(sub_kernels))]
        if self.concatenate_cond:
            cum_cond_sizes = np.cumsum([0] + [k.cond_dimension for k in sub_kernels])
            self.cond_edges = [slice(cum_cond_sizes[i], cum_cond_sizes[i+1]) for i in range(len(sub_kernels))]
        else:
            self.cond_edges = [slice(0, self.cond_dimension) for _ in range(len(sub_kernels))]

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        The log probability of the samples given the conditioning information. The log probability of each sub-sample
        corresponding to each sub-kernel is calculated and summed

        Parameters
        ----------
        samples
            The samples for which the log probability should be calculated. Should have shape (N, self.sample_dimension)
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            The log probability of each samples given the conditioning information with shape (N,)
        """
        log_prob = 0.
        for sample_edges, cond_edges, sub_kernel in zip(self.sample_edges, self.cond_edges, self.sub_kernels):
            log_prob = (
                log_prob
                + sub_kernel.log_prob(samples[:, sample_edges], cond[:, cond_edges])
            )
        return log_prob

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Draws samples from each sub-kernel and concatenates them

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension)
        """
        return torch.cat([
            k.draw(cond[:, cond_edges])
            for k, cond_edges in zip(self.sub_kernels, self.cond_edges)
        ], dim=-1)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Draws and concatenates samples from each sub-kernel and sums the log probability of each sub-sample using
        each sub-kernel's draw_with_log_prob function.

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension) and the log
            probability of each samples given the conditioning information with shape (N,)
        """
        samples, log_probs = zip(*[k.draw_with_log_prob(cond[:, c]) for k, c in zip(self.sub_kernels, self.cond_edges)])
        return (
            torch.cat(samples, dim=-1),
            sum(log_probs, torch.tensor(0.)),
        )

    def draw_with_separate_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tuple[Tensor, ...]]:
        """
        Utility method to draw samples but return the log-likelihoods of each independent sub-kernel's samples
        separately it is unlikely the end user ever needs to use this function, but its helpful for the implementation of
        UnlabelledMultiKernelTrainer.

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension) and a tuple of length
            len(self.sub_kernels) containing the log probability of each sample from each sub-kernel
        """
        samples, log_probs = zip(*[k.draw_with_log_prob(cond[:, c]) for k, c in zip(self.sub_kernels, self.cond_edges)])
        return (
            torch.cat(samples, dim=-1),
            log_probs,
        )

    @classmethod
    def merge(cls, a: TrainableKernelBase, b: TrainableKernelBase, concatenate_cond) -> 'ConcatenatedKernel':
        """
        Merges two trainable kernels into a single ConcatenatedKernel. If either sub-kernel is itself a
        ConcatenatedKernel with the same value of concatenate_cond, the sub-kernels are uncurried

        Parameters
        ----------
        a
            A TrainableKernelBase
        b
            A TrainableKernelBase
        concatenate_cond
            Whether the conditioning information for each sample-kernel should be concatenated or assume they're the
            same

        Returns
        -------
        ConcatenatedKernel
            Containing the sub-kernels
        """
        a_kernels = list(a.sub_kernels) if (isinstance(a, cls) and a.concatenate_cond == concatenate_cond) else [a]
        b_kernels = list(b.sub_kernels) if (isinstance(b, cls) and b.concatenate_cond == concatenate_cond) else [b]

        return cls(a_kernels + b_kernels, concatenate_cond=concatenate_cond)
