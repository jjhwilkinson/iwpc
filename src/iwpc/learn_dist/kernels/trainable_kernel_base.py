from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from lightning import LightningModule
from sympy.printing.pytorch import torch
from torch import Tensor
from torch import optim
from torch.nn import ModuleList
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.encodings.encoding_base import Encoding


class TrainableKernelBase(LightningModule, ABC):
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
        self.cond_dimension = int(cond_dimension.input_shape[0]) if isinstance(cond_dimension, Encoding) else cond_dimension

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

    def __add__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
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

    def __or__(self, other: 'TrainableKernelBase') -> 'ConditionedKernel':
        """
        Syntactic sugar to merge two trainable kernels when the samples of 'other' should be prepended to the
        conditioning information of self. See ConditionedKernel

        Parameters
        ----------
        other
            Another instance of TrainableKernelBase

        Returns
        -------
        ConditionedKernel
            A ConditionedKernel with sample dimension equal to self.sample_dimension + other.sample_dimension and
            condition dimension equal to other.cond_dimension
        """
        return ConditionedKernel(self, other)

    def calculate_loss(self, batch: tuple) -> Tensor:
        """
        Calculate the loss of the given batch

        Parameters
        ----------
        batch : tuple
            Training batch

        Returns
        -------
        Tensor
            A tensor containing ``-mean(log_prob)`` over finite entries.
        """
        cond, targets, _ = batch
        log_prob = self.log_prob(targets, cond)
        return - log_prob[log_prob.isfinite()].mean()

    def training_step(self, batch: tuple) -> Tensor:
        """
        Lightning training step: compute and log the training loss.

        Parameters
        ----------
        batch : tuple
            Training batch

        Returns
        -------
        Tensor
            A tensor representing the training loss for this batch.
        """
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple) -> Tensor:
        """
        Lightning validation step: compute and log the validation loss.

        Parameters
        ----------
        batch : tuple
            Training batch

        Returns
        -------
        Tensor
            A scalar tensor representing the validation loss for this batch.
        """
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        """
        Configure the optimizer and learning-rate scheduler.

        Returns
        -------
        dict
            A Lightning optimizer/scheduler config dictionary
        """
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=10,
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }


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
        a_kernels = a.sub_kernels if (isinstance(a, cls) and a.concatenate_cond==concatenate_cond) else [a]
        b_kernels = b.sub_kernels if (isinstance(b, cls) and b.concatenate_cond==concatenate_cond) else [b]

        return cls(a_kernels + b_kernels, concatenate_cond=concatenate_cond)


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
        ], dim=-1), sample_kernel_log_prob + sample_kernel_log_prob
