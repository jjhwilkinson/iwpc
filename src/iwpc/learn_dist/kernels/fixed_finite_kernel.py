import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class FixedFiniteKernel(FiniteKernelInterface, TrainableKernelBase):
    """
    FiniteKernelInterface implementation over a finite sample space wherein the probability of each outcome is fixed
    and independent of the conditioning information
    """
    def __init__(
        self,
        sample_space: FiniteSampleSpace,
        probs: Tensor | list[float],
        cond: Encoding | int,
    ):
        """
        Parameters
        ----------
        sample_space
            A FiniteSampleSpace instance
        probs
            A list of probabilities satisfying len(probs) == sample_space.num_outcomes and sum(probs) == 1
        cond
            The encoding of the conditioning information
        """
        probs = torch.tensor(probs, dtype=torch.float)
        assert torch.isclose(probs.sum(), torch.tensor(1.))
        log_probs = probs.log()
        assert log_probs.shape[0] == sample_space.num_outcomes
        super().__init__(
            sample_space,
            sample_space.dimension,
            cond,
        )
        self.register_buffer('log_probs', log_probs)

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A Tensor containing the conditioning information

        Returns
        -------
        Tensor
            The log of the probabilities provided in the constructor, repeated for each sample
        """
        return self.log_probs.repeat(cond.shape[0], 1)
