from abc import ABC, abstractmethod
from typing import Iterable

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ExplicitFiniteSampleSpace, FiniteSampleSpace


def trivial_index_sample_space() -> ExplicitFiniteSampleSpace:
    """
    Constructs a 1-outcome 0-dimensional FiniteSampleSpace, used as a placeholder index space when an indexed kernel
    is not actually indexed (K = 1). Collapses the (N, M, K) log-prob table to (N, M, 1) so the indexed and
    non-indexed paths share a single implementation

    Returns
    -------
    ExplicitFiniteSampleSpace
        A FiniteSampleSpace with a single outcome of dimension 0
    """
    return ExplicitFiniteSampleSpace(
        torch.zeros(1, 0),
        lambda outcomes: torch.zeros(outcomes.shape[0], dtype=torch.long, device=outcomes.device),
    )


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

    construct_log_prob_table(x) takes the standard conditioning x — the full cond with the
    index_cond_indices columns removed — and returns a (N, M, K) tensor of normalised log-probabilities,
    where column k holds ``log p(A=m | B=k, x)``.

    Non-indexed kernels can satisfy this interface by passing ``index_cond_indices=[]`` and a 1-outcome
    placeholder index_sample_space (see ``trivial_index_sample_space``); ``construct_log_prob_table`` then
    returns a (N, M, 1) tensor whose only column is the standard (N, M) log-prob output. This unifies the
    indexed and non-indexed code paths
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
    def construct_log_prob_table(self, cond: Tensor) -> Tensor:
        """
        Returns the full log-probability table for all index values in a single forward pass, normalised over
        the sample axis (dim=1).

        Parameters
        ----------
        cond
            Standard conditioning x of shape (N, cond_dim - len(index_cond_indices)), with the discrete index
            columns removed.

        Returns
        -------
        Tensor
            Shape (N, M, K) where M = sample_space.num_outcomes and K = index_sample_space.num_outcomes.
            Column k holds ``log p(A=m | B=k, x)`` for m in [0, M).
        """
        pass

    @staticmethod
    def expected_sample_index_cond_indices(conditioning_kernel: 'FiniteKernelInterface') -> list[int]:
        """
        For a given conditioning kernel, returns the list of index_cond_indices that any compatible sample_kernel must
        satisfy for the fast aligned-table composition path

        Parameters
        ----------
        conditioning_kernel
            A FiniteKernelInterface, optionally also an IndexedInterface

        Returns
        -------
        list[int]
            The list of index_cond_indices that any compatible sample_kernel must have. When conditioning_kernel is a
            non-indexed FiniteKernelInterface (or an IndexedInterface with empty index_cond_indices), this is just
            list(range(conditioning_kernel.sample_dimension)) — the sample_kernel is indexed only on the prepended
            B2 outcome columns. When conditioning_kernel is itself indexed (B1), this extends with shifted copies of
            its index_cond_indices so the sample_kernel is indexed on both B2 and B1
        """
        dim_B2 = conditioning_kernel.sample_dimension
        if isinstance(conditioning_kernel, IndexedInterface):
            return list(range(dim_B2)) + [dim_B2 + int(i) for i in conditioning_kernel.index_cond_indices]
        return list(range(dim_B2))

    @staticmethod
    def is_aligned(sample_kernel, conditioning_kernel: 'FiniteKernelInterface') -> bool:
        """
        Whether the given sample_kernel is index-aligned with conditioning_kernel for the fast composition path

        Parameters
        ----------
        sample_kernel
            Candidate sample kernel
        conditioning_kernel
            Candidate conditioning kernel

        Returns
        -------
        bool
            True iff sample_kernel is an IndexedInterface and its index_cond_indices match
            expected_sample_index_cond_indices(conditioning_kernel)
        """
        if not isinstance(sample_kernel, IndexedInterface):
            return False
        return (
            list(int(i) for i in sample_kernel.index_cond_indices)
            == IndexedInterface.expected_sample_index_cond_indices(conditioning_kernel)
        )

