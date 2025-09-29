from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from sympy.printing.pytorch import torch
from torch import Tensor, nn

from iwpc.encodings.encoding_base import Encoding


class TrainableKernelBase(ABC, nn.Module):
    """
    Abstract base class for all trainable kernels. A kernel is defined as a conditional likelihood distribution that is
    convolved against some base distribution, like a detector response
    """
    def __init__(
        self,
        sample_dimension: int,
        cond_dimension: Encoding | int,
    ):
        """
        Parameters
        ----------
        sample_dimension
            The dimension of the sample space
        cond_dimension
            The dimension of the conditioning information
        """
        super().__init__()
        self.sample_dimension = sample_dimension
        self.cond_dimension = int(cond_dimension.input_shape) if isinstance(cond_dimension, Encoding) else cond_dimension

    @abstractmethod
    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        The log probability of the samples given the conditioning information. Must be differentiable

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

    @abstractmethod
    def _draw(self, cond: Tensor) -> Tensor:
        """
        Draw a sample from the conditional distribution for each row of conditioning information. Does not need to be
        differentiable

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension)
        """

    def draw(self, cond: Tensor) -> Tensor:
        """
        Draw a sample from the conditional distribution for each row of conditioning information. Ensures no gradient
        information is kept

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
            A sample for each row of conditioning information with shape (N, self.sample_dimension)
        """
        with torch.no_grad():
            return self._draw(cond)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Draw a sample from the conditional distribution for each row of conditioning information along with its
        corresponding log probability. Default implementation calls self.draw and self.log_prob, but sometimes these
        steps can be merged for additional performance

        Parameters
        ----------
        cond
            The conditioning information for each sample. Should have shape (N, self.cond_dimension)

        Returns
        -------
        Tuple[Tensor, Tensor]
            A sample for each row of conditioning information with shape (N, self.sample_dimension) and the log
            probability of each samples given the conditioning information with shape (N,)
        """
        samples = self.draw(cond)
        log_prob = self.log_prob(samples, cond)
        return samples, log_prob

    def __and__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
        """
        Syntactic sugar to merge two trainable kernels when they share the same conditional information. The sample
        dimensions are concatenated and samples are drawn independently

        Parameters
        ----------
        other
            Another instance of TrainableKernelBase that shares the same conditioning information space

        Returns
        -------
        ConcatenatedKernel
            A ConcatenatedKernel with sample dimension equal to self.sample_dimension + other.sample_dimension and
            condition dimension equal to self.cond_dimension
        """
        return ConcatenatedKernel.merge(self, other, False)

    def __or__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
        """
        Syntactic sugar to merge two trainable kernels when the conditional information spaces should be concatenated.
        The sample/conditioning dimensions are concatenated and samples are drawn independently

        Parameters
        ----------
        other
            Another instance of TrainableKernelBase

        Returns
        -------
        ConcatenatedKernel
            A ConcatenatedKernel with sample dimension equal to self.sample_dimension + other.sample_dimension and
            condition dimension equal to self.cond_dimension + other.cond_dimension
        """
        return ConcatenatedKernel.merge(self, other, True)


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
        super().__init__(sum(k.sample_dimension for k in sub_kernels), cond_dimension)

        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels
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
        Draws samples from each sub-kernel and concatenates the them

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
        a_kernels = a.sub_kernels if (isinstance(a, ConcatenatedKernel) and a.concatenate_cond==concatenate_cond) else [a]
        b_kernels = b.sub_kernels if (isinstance(b, ConcatenatedKernel) and b.concatenate_cond==concatenate_cond) else [b]

        return ConcatenatedKernel(a_kernels + b_kernels, concatenate_cond=concatenate_cond)
