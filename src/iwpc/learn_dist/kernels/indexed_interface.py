from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace


class IndexedInterface(ABC):
    """
    Interface for finite kernels that model p(A | B=b, x) by exposing a full logit table over
    all index values b in a single forward pass, rather than requiring a separate forward pass
    per index value.

    Implementors must set the following instance attributes in their __init__:
        index_sample_space: FiniteSampleSpace
            The discrete sample space of the index B, with K outcomes.
        index_cond_indices: list[int]
            Indices into the full cond tensor that carry the discrete index b

    construct_logit_table(x) takes the standard conditioning x — the full cond with the
    index_cond_indices columns removed — and returns a (N, M, K) tensor. Column k holds the M
    unnormalised logits for index value b_k.
    """
    def __init__(
        self,
        sample_space: FiniteSampleSpace,
        index_cond_indices: Iterable[int],
        *args,
        **kwargs,
    ):
        """
        Parameters
        ----------
        sample_space
            The discrete sample space of the index B, with K outcomes.
        index_cond_indices
            The indices into the conditioning information that correspond to the samples within sample_space.
        args
            Passed on to super constructor
        kwargs
            Passed on to super constructor
        """
        super().__init__(*args, **kwargs)
        self.index_sample_space = sample_space
        self.index_cond_indices = torch.tensor(index_cond_indices)
        self.standard_cond_indices = torch.tensor([i for i in range(self.cond_dimension) if i not in index_cond_indices])

    @abstractmethod
    def construct_logit_table(self, cond: Tensor) -> Tensor:
        """
        Returns the full logit table for all index values in a single forward pass.

        Parameters
        ----------
        cond
            Standard conditioning x of shape (N, cond_dim - len(index_cond_indices)), with the
            discrete index columns removed.

        Returns
        -------
        Tensor
            Shape (N, M, K) where M = sample_space.num_outcomes and
            K = index_sample_space.num_outcomes. Column k holds the M logits for index value k.
        """
        pass

    def __or__(self, other) -> 'IndexedFiniteConditionedKernel | FiniteConditionedKernel':
        """
        If other is an indexed kernel `compatible` with self as a sample kernel, returns an
        IndexedFiniteConditionedKernel modelling p(A, B2 | B1, z). Compatibility requires
        self.index_cond_indices == list(range(dim_B2)) + [dim_B2 + i for i in other.index_cond_indices],
        i.e. self must be indexed on the sample space of other, and the index space of other.
        Falls back to super().__or__ otherwise
        """
        from iwpc.learn_dist.kernels.indexed_finite_conditioned_kernel import IndexedFiniteConditionedKernel
        if isinstance(other, FiniteKernelInterface) and isinstance(other, IndexedInterface):
            if list(self.index_cond_indices) == IndexedFiniteConditionedKernel.expected_sample_index_cond_indices(other):
                return IndexedFiniteConditionedKernel(self, other)
        return super().__or__(other)
