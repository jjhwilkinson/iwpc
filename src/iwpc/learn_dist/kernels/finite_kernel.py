from itertools import chain
from typing import Callable, Iterable

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
    and entries equal to 0 or 1.

    Indexed mode
    ------------
    When ``index_cond_indices``, ``outcome_to_idx_fn``, and ``num_index_outcomes`` are all provided the kernel
    operates in *indexed mode*: it models p(A | B=b, x) by producing a single K×M logit table from one model(x)
    call and selecting the b-th row. This collapses the K separate forward passes that
    ``FiniteConditionedKernel`` would otherwise require into one.
    """
    def __init__(
        self,
        num_variable_outcomes: int | Iterable[int],
        cond: Encoding | int,
        logit_model: torch.nn.Module | None = None,
        init_prob: float | Iterable[float] | Iterable[Iterable[float]] | None = None,
        index_cond_indices: list[int] | int | None = None,
        outcome_to_idx_fn: Callable[[Tensor], Tensor] | None = None,
        num_index_outcomes: int = 0,
    ):
        """
        Parameters
        ----------
        num_variable_outcomes
            A tuple of integers representing the number of possible values per variable. The product of the constituents
            gives the total number of possible outcomes. If an integer is given, it is interpreted as the tuple
            (num_outcomes,). In the ABC example, this would be (2, 2, 2)
        cond
            The conditioning space encoding or dimension for the logit model (the standard conditioning x, not
            including the index part b in indexed mode)
        logit_model
            Optional custom logit model. If None, a default model is constructed via basic_model_factory
        init_prob
            Optional initial probability bias. A float p initialises a binary kernel with shift [log(1-p), log(p)].
            An iterable of floats provides one probability per outcome. In indexed mode, an iterable of iterables of
            shape K×M provides a distinct initial distribution per index value; a flat iterable of length M is
            broadcast uniformly across all K index values. Ignored if logit_model is provided.
        index_cond_indices
            Indexed mode only. Columns of the full cond tensor that carry the discrete index b. An int N is
            treated as list(range(N)).
        outcome_to_idx_fn
            Indexed mode only. Maps b tensor of shape (N, len(index_cond_indices)) to an integer index tensor
            of shape (N,) in [0, num_index_outcomes).
        num_index_outcomes
            Indexed mode only. Number of distinct values K that the index b can take.
        """
        if isinstance(num_variable_outcomes, int):
            num_variable_outcomes = (num_variable_outcomes,)

        sample_space = ExplicitFiniteSampleSpace(torch.tensor([
            torch.unravel_index(outcome_idx, num_variable_outcomes)
            for outcome_idx in torch.arange(np.prod(num_variable_outcomes))
        ]), self.outcome_to_idx)

        indexed = index_cond_indices is not None
        if indexed:
            if isinstance(index_cond_indices, int):
                index_cond_indices = list(range(index_cond_indices))
            standard_cond_dim = int(cond.input_shape[0]) if isinstance(cond, Encoding) else int(cond)
            total_cond_dim = len(index_cond_indices) + standard_cond_dim
        else:
            total_cond_dim = cond

        super().__init__(sample_space, len(num_variable_outcomes), total_cond_dim)
        self.num_variable_outcomes = num_variable_outcomes
        self.index_cond_indices = index_cond_indices
        self.outcome_to_idx_fn = outcome_to_idx_fn
        self.num_index_outcomes = num_index_outcomes
        if indexed:
            self.standard_cond_indices = [i for i in range(total_cond_dim) if i not in index_cond_indices]

        num_logit_outputs = (num_index_outcomes if indexed else 1) * sample_space.num_outcomes

        if logit_model is not None:
            self.logit_model = logit_model
        else:
            if init_prob is not None:
                if isinstance(init_prob, float):
                    if sample_space.num_outcomes != 2:
                        raise ValueError(f"A scalar init_prob can only be used with binary kernels (2 outcomes), got {sample_space.num_outcomes}")
                    probs_per_b = [[1 - init_prob, init_prob]] * (num_index_outcomes if indexed else 1)
                else:
                    probs_list = list(init_prob)
                    if indexed and isinstance(probs_list[0], (list, tuple)):
                        probs_per_b = [list(row) for row in probs_list]
                    else:
                        probs_per_b = [probs_list] * (num_index_outcomes if indexed else 1)
                final_layers = [ConstantScaleLayer(shift=[np.log(p) for row in probs_per_b for p in row])]
            else:
                final_layers = []
            self.logit_model = basic_model_factory(
                cond,
                TrivialEncoding(num_logit_outputs),
                final_layers=final_layers,
            )
        self.register_buffer(
            'reversed_cumprod_num_variable_outcomes',
            torch.tensor(list(np.cumprod([num_variable_outcomes[::-1]])[::-1]) + [1])[1:],
        )

    @classmethod
    def condition_on(
        cls,
        num_variable_outcomes: int | Iterable[int],
        conditioning_kernel: 'FiniteKernelInterface',
        standard_cond: Encoding | int,
        **kwargs,
    ) -> 'FiniteKernel':
        """
        Construct an indexed-mode FiniteKernel that pairs with the given conditioning kernel inside a
        FiniteConditionedKernel. The resulting kernel expects cond of shape (N, b_dim + x_dim), matching
        FiniteConditionedKernel's convention of prepending the b outcome to z.

        Parameters
        ----------
        num_variable_outcomes
            Number of outcomes per variable for the sample kernel (see __init__)
        conditioning_kernel
            The FiniteKernelInterface whose outcomes serve as the discrete index b
        standard_cond
            The encoding or dimension of the standard conditioning x passed to the logit model
        **kwargs
            Forwarded to __init__ (e.g. init_prob, logit_model)
        """
        return cls(
            num_variable_outcomes,
            standard_cond,
            index_cond_indices=list(range(conditioning_kernel.sample_dimension)),
            outcome_to_idx_fn=lambda x: conditioning_kernel.sample_space.outcome_to_idx(x),
            num_index_outcomes=conditioning_kernel.sample_space.num_outcomes,
            **kwargs,
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

    def construct_logit_table(self, standard_cond: Tensor) -> Tensor:
        """
        Indexed mode only. Returns the full K×M logit table for all index values.

        Parameters
        ----------
        standard_cond
            The standard conditioning x of shape (N, standard_cond_dim)

        Returns
        -------
        Tensor
            A tensor of shape (N, K, M) where K = num_index_outcomes and M = sample_space.num_outcomes
        """
        assert self.index_cond_indices is not None, "construct_logit_table is only valid in indexed mode"
        N = standard_cond.shape[0]
        return self.logit_model(standard_cond).reshape(N, self.num_index_outcomes, -1).transpose(1, 2)

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
        if self.index_cond_indices is None:
            return self.logit_model(cond)
        x = cond[:, self.standard_cond_indices]
        b = cond[:, self.index_cond_indices]
        table = self.construct_logit_table(x)                                              # (N, M, K)
        idxs = self.outcome_to_idx_fn(b).long()
        return table.gather(2, idxs[:, None, None].expand(-1, table.shape[1], 1)).squeeze(2)  # (N, M)

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
