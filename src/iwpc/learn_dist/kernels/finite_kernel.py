from itertools import chain
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ExplicitFiniteSampleSpace
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


class FiniteKernel(FiniteKernelInterface, TrainableKernelBase):
    """
    Kernel for discrete outcomes. Often discrete probability spaces are constructed as the cartesian product over
    variables. For example, consider the variables A, B, C that can either be true or false. There are 8 possible
    outcomes corresponding to {(not A and not B and not C), (not A and not B and C) , etc}. The sample space of this
    kernel is an integer vector of length equal to the number of distinct variables with each entry between 0 and the
    number of values said variables can take less one. In the above ABC example, samples are vectors of length three
    and entries equal to 0 or 1
    """
    def __init__(
        self,
        num_variable_outcomes: int | Iterable[int],
        cond: Encoding | int,
        logit_model: torch.nn.Module | None = None,
        init_prob: float | Iterable[float] | None = None,
    ):
        """
        Parameters
        ----------
        num_variable_outcomes
            A tuple of integers representing the number of possible values per variable. The product of the constituents
            gives the total number of possible outcomes. If an integer is given, it is interpreted as the tuple
            (num_outcomes,). In the ABC example, this would be (2, 2, 2)
        cond
            The conditioning space encoding or dimension
        logit_model
            Optional custom logit model. If None, a default model is constructed via basic_model_factory.
            When provided, the model must accept cond of shape (N, cond_dim) and return (N, M) where
            M = sample_space.num_outcomes
        init_prob
            Optional initial probability bias applied as a constant log-probability shift to the logits. A float p
            initialises a binary kernel with shift [log(1-p), log(p)] — raises ValueError if the kernel has more
            than 2 outcomes. An iterable of floats provides one probability per outcome for multi-outcome kernels.
            Ignored if logit_model is provided.
        """
        if isinstance(num_variable_outcomes, int):
            num_variable_outcomes = (num_variable_outcomes,)

        sample_space = ExplicitFiniteSampleSpace(torch.tensor([
            torch.unravel_index(outcome_idx, num_variable_outcomes)
            for outcome_idx in torch.arange(np.prod(num_variable_outcomes))
        ]), self.outcome_to_idx)

        super().__init__(sample_space, len(num_variable_outcomes), cond)
        self.num_variable_outcomes = num_variable_outcomes

        if logit_model is not None:
            self.logit_model = logit_model
        else:
            if init_prob is not None:
                if isinstance(init_prob, float):
                    if self.sample_space.num_outcomes != 2:
                        raise ValueError(f"A scalar init_prob can only be used with binary kernels (2 outcomes), got {self.sample_space.num_outcomes}")
                    probs = [1 - init_prob, init_prob]
                else:
                    probs = list(init_prob)
                final_layers = [ConstantScaleLayer(shift=[np.log(p) for p in probs])]
            else:
                final_layers = []
            self.logit_model = basic_model_factory(
                cond,
                TrivialEncoding(self.sample_space.num_outcomes),
                final_layers=final_layers,
            )
        self.register_buffer(
            'reversed_cumprod_num_variable_outcomes',
            torch.tensor(list(np.cumprod([num_variable_outcomes[::-1]])[::-1]) + [1])[1:],
        )

    def outcome_to_idx(self, samples: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.sample_dimension) of integers

        Returns
        -------
        Tensor
            An integer tensor of shape (N,) containing the indices for each sample
        """
        return (samples * self.reversed_cumprod_num_variable_outcomes[None, :]).sum(dim=-1).int()

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of size (N, self.num_outcomes) containing logits over the outcomes for each row of conditioning
            information
        """
        return self.logit_model(cond)

    def __ror__(self, other: list[TrainableKernelBase | list[TrainableKernelBase]]) -> "BranchingKernel":
        """
        Syntactic sugar to construct a BranchingKernel from a list of TrainableKernelBase instances. The branching
        kernel samples from each of its sub-kernels based upon the outcome index of this finite kernel

        Parameters
        ----------
        other
            Either a list with as many entries as self.num_outcomes, or a list of lists of TrainableKernelBase instances
            wherein len(other[i]) equals self.num_variable_outcomes[i]

        Returns
        -------
        BranchingKernel
            A branching kernel that samples from each of its sub-kernels based upon the outcome index of this finite
            kernel
        """
        if all(isinstance(e, list) for e in other):
            return super().__ror__(list(chain(*other)))
        return super().__ror__(other)
