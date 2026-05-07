from abc import ABC, abstractmethod
from typing import Tuple

from lightning import LightningModule
from sympy.printing.pytorch import torch
from torch import Tensor
from torch import optim
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
            if cond.shape[0] == 0:
                return torch.zeros((cond.shape[0], self.sample_dimension), dtype=cond.dtype, device=cond.device)
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
        from iwpc.learn_dist.kernels.concatenated_kernel import ConcatenatedKernel
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
        from iwpc.learn_dist.kernels.concatenated_kernel import ConcatenatedKernel
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
        from iwpc.learn_dist.kernels.conditioned_kernel import ConditionedKernel
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
            A tensor containing -mean(log_prob) over finite entries.
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
