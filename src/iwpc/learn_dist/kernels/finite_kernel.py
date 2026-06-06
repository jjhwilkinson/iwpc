from itertools import chain
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import CartesianFiniteSampleSpace, FiniteSampleSpace
from iwpc.learn_dist.kernels.indexed_interface import IndexedInterface, trivial_index_sample_space
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


class FiniteKernel(IndexedInterface, FiniteKernelInterface, TrainableKernelBase):
    """
    Kernel for discrete outcomes. Often discrete probability spaces are constructed as the cartesian product over
    variables. For example, consider the variables A, B, C that can either be true or false. There are 8 possible
    outcomes corresponding to {(not A and not B and not C), (not A and not B and C) , etc}. The sample space of this
    kernel is an integer vector of length equal to the number of distinct variables with each entry between 0 and the
    number of values said variables can take less one. In the above ABC example, samples are vectors of length three
    and entries equal to 0 or 1.

    Offers an optional 'fast path' for modeling p(A | B=b, x) when B is a discrete variable by exposing a full M×K
    logit table over all K index outcomes b in a single forward pass. Use ``index_cond_indices`` and
    ``index_sample_space`` to enable.
    """
    def __init__(
        self,
        sample_space: int | Iterable[int] | FiniteSampleSpace,
        cond: Encoding | int,
        index_cond_indices: list[int] | int | None = None,
        index_sample_space: FiniteSampleSpace | None = None,
        logit_model: torch.nn.Module | None = None,
        init_log_probs: float | Iterable[float] | Iterable[Iterable[float]] | None = None,
    ):
        """
        Parameters
        ----------
        sample_space
            The FiniteSampleSpace this kernel models a distribution over. For Cartesian-product spaces, an
            ``int`` or iterable of ints can be passed as shorthand and is auto-wrapped in a
            ``CartesianFiniteSampleSpace``; the ABC example becomes ``(2, 2, 2)``
        cond
            The encoding or dimension of the standard (unindexed) conditioning information x passed to the logit model.
            When indexing is not used this is the full cond; when indexing is used, the full cond_dimension becomes
            len(index_cond_indices) + standard_cond_dim with the index columns at index_cond_indices
        index_cond_indices
            Optional columns of the full cond tensor that carry the discrete index b. An int N is treated as
            list(range(N)). None disables indexing (K=1)
        index_sample_space
            Optional FiniteSampleSpace of the discrete index b. Required when index_cond_indices is non-empty;
            ignored otherwise
        logit_model
            Optional custom logit model. Must accept x of shape (N, standard_cond_dim) and return (N, M*K) logits
            laid out in M-major order. If None, a default model is constructed via basic_model_factory
        init_log_probs
            Optional initial log-probability bias applied as a constant shift to the logits. A float lp initialises a
            binary kernel (M=2) with shift [log(1-exp(lp)), lp] broadcast across all K index values. A 1D iterable of
            length M provides one log-prob per outcome, broadcast across K. A 2D (M, K) iterable provides a distinct
            initial log-prob per outcome per index value. Ignored if logit_model is provided
        """
        if not isinstance(sample_space, FiniteSampleSpace):
            sample_space = CartesianFiniteSampleSpace(sample_space)
        if index_cond_indices is None:
            index_cond_indices = []
        elif isinstance(index_cond_indices, int):
            index_cond_indices = list(range(index_cond_indices))

        if len(index_cond_indices) == 0:
            if index_sample_space is None:
                index_sample_space = trivial_index_sample_space()
        else:
            if index_sample_space is None:
                raise ValueError("index_sample_space is required when index_cond_indices is non-empty")

        K = index_sample_space.num_outcomes
        M = sample_space.num_outcomes
        standard_cond_dim = int(cond.input_shape[0]) if isinstance(cond, Encoding) else int(cond)
        total_cond_dim = standard_cond_dim + len(index_cond_indices)

        super().__init__(
            index_sample_space,
            index_cond_indices,
            sample_space,
            sample_space.dimension,
            total_cond_dim,
        )

        if logit_model is not None:
            self.logit_model = logit_model
        else:
            final_layers = []
            if init_log_probs is not None:
                final_layers.append(ConstantScaleLayer(shift=self._build_init_shift(init_log_probs, M, K)))
            self.logit_model = basic_model_factory(
                cond,
                TrivialEncoding(M * K),
                final_layers=final_layers,
            )

    @staticmethod
    def _build_init_shift(init_log_probs, M: int, K: int) -> list[float]:
        """
        Normalise init_log_probs into a flat length-(M*K) shift to apply to the logit_model output prior to its
        reshape into (N, M, K). Layout matches the reshape order — M-major, K-minor — so the shift at flat index
        m*K + k corresponds to outcome m, index k

        Parameters
        ----------
        init_log_probs
            See FiniteKernel.__init__
        M
            Number of sample outcomes
        K
            Number of index outcomes

        Returns
        -------
        list[float]
            A list of length M*K
        """
        arr = np.asarray(init_log_probs, dtype=float)
        if arr.ndim == 0:
            if M != 2:
                raise ValueError(f"A scalar init_log_probs can only be used with binary kernels (2 outcomes), got {M}")
            per_outcome = np.array([np.log1p(-np.exp(arr)), float(arr)])
            return np.repeat(per_outcome, K).tolist()
        if arr.ndim == 1:
            if arr.shape[0] != M:
                raise ValueError(f"1D init_log_probs must have length M={M}, got {arr.shape[0]}")
            return np.repeat(arr, K).tolist()
        if arr.ndim == 2:
            if arr.shape != (M, K):
                raise ValueError(f"2D init_log_probs must have shape (M={M}, K={K}), got {tuple(arr.shape)}")
            return arr.reshape(-1).tolist()
        raise ValueError(f"init_log_probs must be 0D, 1D, or 2D, got {arr.ndim}D")

    @classmethod
    def condition_on(
        cls,
        sample_space: int | Iterable[int] | FiniteSampleSpace,
        conditioning_kernel: FiniteKernelInterface,
        standard_cond: Encoding | int,
        **kwargs,
    ) -> 'FiniteKernel':
        """
        Construct a FiniteKernel indexed on the outcomes of the given finite conditioning kernel. The resulting kernel
        expects cond of shape (N, conditioning_kernel.sample_dimension + standard_cond_dim), matching
        FiniteConditionedKernel's convention of prepending the b outcome to the conditioning information

        Parameters
        ----------
        sample_space
            Sample space of the resulting kernel (see __init__)
        conditioning_kernel
            The FiniteKernelInterface whose outcomes serve as the discrete index b
        standard_cond
            The encoding or dimension of the standard conditioning x passed to the logit model
        **kwargs
            Forwarded to __init__ (e.g. init_log_probs, logit_model)

        Returns
        -------
        FiniteKernel
            A FiniteKernel indexed on the conditioning kernel's sample space, with index_cond_indices set so that the
            first conditioning_kernel.sample_dimension columns of cond are interpreted as the index B
        """
        return cls(
            sample_space,
            standard_cond,
            list(range(conditioning_kernel.sample_dimension)),
            conditioning_kernel.sample_space,
            **kwargs,
        )

    def construct_log_prob_table(self, cond: Tensor) -> Tensor:
        """
        Returns the full log-probability table for all index values in a single forward pass.

        Parameters
        ----------
        cond
            The standard conditioning x of shape (N, standard_cond_dim) — i.e. with the index columns removed. When
            no indexing is used this is the full cond

        Returns
        -------
        Tensor
            A tensor of shape (N, M, K) where M = sample_space.num_outcomes and K = index_sample_space.num_outcomes.
            Column k holds ``log p(A=m | B=k, x)``. For non-indexed kernels K=1
        """
        N = cond.shape[0]
        return self.logit_model(cond).reshape(
            N,
            self.sample_space.num_outcomes,
            self.index_sample_space.num_outcomes,
        ).log_softmax(dim=1)

    def construct_log_probs(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of size (N, self.sample_space.num_outcomes) of log-probabilities over the outcomes for each row
            of conditioning information
        """
        if len(self.index_cond_indices) == 0:
            return self.construct_log_prob_table(cond).squeeze(-1)
        x = cond[:, self.standard_cond_indices]
        b = cond[:, self.index_cond_indices]
        table = self.construct_log_prob_table(x)
        idxs = self.index_sample_space.outcome_to_idx(b).long()
        return table.gather(2, idxs[:, None, None].expand(-1, table.shape[1], 1)).squeeze(2)

    def __ror__(self, other: list[TrainableKernelBase | list[TrainableKernelBase]]) -> "BranchingKernel":
        """
        Syntactic sugar to construct a BranchingKernel from a list of TrainableKernelBase instances. The branching
        kernel samples from each of its sub-kernels based upon the outcome index of this finite kernel

        Parameters
        ----------
        other
            Either a list with as many entries as self.num_outcomes, or, when the kernel's sample space is a
            ``CartesianFiniteSampleSpace``, a list of lists of TrainableKernelBase instances wherein
            ``len(other[i])`` equals ``self.sample_space.num_variable_outcomes[i]``

        Returns
        -------
        BranchingKernel
            A branching kernel that samples from each of its sub-kernels based upon the outcome index of this finite
            kernel
        """
        if all(isinstance(e, list) for e in other):
            return super().__ror__(list(chain(*other)))
        return super().__ror__(other)
