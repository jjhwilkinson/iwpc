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
    Kernel for discrete outcomes over an arbitrary ``FiniteSampleSpace``. The sample space defines both the
    enumeration of distinct outcomes and the mapping between an outcome tensor and its integer index.

    The common case where the space is a Cartesian product over a tuple of per-variable outcome counts is
    handled by ``CartesianFiniteSampleSpace`` — for convenience ``__init__`` also accepts an ``int`` or
    iterable of ints as shorthand and wraps it in that class. For example, the three binary variables A, B,
    C give a Cartesian sample space of size 8 with samples of dimension 3; pass ``sample_space=(2, 2, 2)``.
    Other shapes (e.g. a sample space restricted to a subset of a Cartesian product) can be supplied as an
    ``ExplicitFiniteSampleSpace`` directly.

    Offers an optional 'fast path' for modeling p(A | B=b, x) when B is a discrete variable by exposing a
    full M×K logit table over all K index outcomes b in a single forward pass. Use ``index_cond_indices``
    and ``index_sample_space`` to enable.
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
            list(range(N)). None disables indexing (number of index outcomes = 1)
        index_sample_space
            Optional FiniteSampleSpace of the discrete index b. Required when index_cond_indices is non-empty;
            ignored otherwise
        logit_model
            Optional custom logit model. Must accept x of shape (N, standard_cond_dim) and return
            (N, num_sample_outcomes * num_index_outcomes) logits laid out in sample-major order. If None, a
            default model is constructed via basic_model_factory
        init_log_probs
            Optional initial log-probability bias applied as a constant shift to the logits. A float lp
            initialises a binary kernel (num_sample_outcomes=2) with shift [log(1-exp(lp)), lp] broadcast
            across all index outcomes. A 1D iterable of length num_sample_outcomes provides one log-prob per
            sample outcome, broadcast across index outcomes. A 2D iterable of shape
            (num_sample_outcomes, num_index_outcomes) provides a distinct initial log-prob per sample
            outcome per index outcome. Ignored if logit_model is provided
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

        standard_cond_dim = int(cond.input_shape[0]) if isinstance(cond, Encoding) else int(cond)
        super().__init__(
            index_sample_space,
            index_cond_indices,
            sample_space,
            sample_space.dimension,
            standard_cond_dim + len(index_cond_indices),
        )

        if logit_model is not None:
            self.logit_model = logit_model
        else:
            final_layers = []
            if init_log_probs is not None:
                final_layers.append(ConstantScaleLayer(
                    shift=self._build_init_shift(
                        init_log_probs, sample_space.num_outcomes, index_sample_space.num_outcomes,
                    )
                ))
            self.logit_model = basic_model_factory(
                cond,
                TrivialEncoding(sample_space.num_outcomes * index_sample_space.num_outcomes),
                final_layers=final_layers,
            )

    @staticmethod
    def _build_init_shift(init_log_probs, num_sample_outcomes: int, num_index_outcomes: int) -> list[float]:
        """
        Normalise init_log_probs into a flat shift to apply to the logit_model output prior to its reshape
        into (N, num_sample_outcomes, num_index_outcomes). Layout matches the reshape order — sample-major,
        index-minor — so the shift at flat index ``m*num_index_outcomes + k`` corresponds to sample outcome
        m, index outcome k

        Parameters
        ----------
        init_log_probs
            See FiniteKernel.__init__
        num_sample_outcomes
            Number of sample outcomes
        num_index_outcomes
            Number of index outcomes

        Returns
        -------
        list[float]
            A list of length ``num_sample_outcomes * num_index_outcomes``
        """
        init_log_probs = np.asarray(init_log_probs, dtype=float)
        if init_log_probs.ndim == 0:
            if num_sample_outcomes != 2:
                raise ValueError(
                    f"A scalar init_log_probs can only be used with binary kernels (2 outcomes), got "
                    f"{num_sample_outcomes}"
                )
            per_outcome = np.array([np.log1p(-np.exp(init_log_probs)), float(init_log_probs)])
            return np.repeat(per_outcome, num_index_outcomes).tolist()
        if init_log_probs.ndim == 1:
            if init_log_probs.shape[0] != num_sample_outcomes:
                raise ValueError(
                    f"1D init_log_probs must have length num_sample_outcomes={num_sample_outcomes}, got "
                    f"{init_log_probs.shape[0]}"
                )
            return np.repeat(init_log_probs, num_index_outcomes).tolist()
        if init_log_probs.ndim == 2:
            expected_shape = (num_sample_outcomes, num_index_outcomes)
            if init_log_probs.shape != expected_shape:
                raise ValueError(
                    f"2D init_log_probs must have shape (num_sample_outcomes={num_sample_outcomes}, "
                    f"num_index_outcomes={num_index_outcomes}), got {tuple(init_log_probs.shape)}"
                )
            return init_log_probs.reshape(-1).tolist()
        raise ValueError(f"init_log_probs must be 0D, 1D, or 2D, got {init_log_probs.ndim}D")

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
            The standard conditioning x of shape (N, standard_cond_dim) — i.e. with the index columns
            removed. When no indexing is used this is the full cond

        Returns
        -------
        Tensor
            A tensor of shape (N, sample_space.num_outcomes, index_sample_space.num_outcomes). Slice
            ``[:, :, k]`` holds ``log p(A=m | B=k, x)`` for sample outcome m. For non-indexed kernels the
            trailing dimension is 1
        """
        return self.logit_model(cond).reshape(
            cond.shape[0],
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
            A tensor of size (N, self.sample_space.num_outcomes) of log-probabilities over the outcomes
            for each row of conditioning information
        """
        if len(self.index_cond_indices) == 0:
            return self.construct_log_prob_table(cond).squeeze(-1)
        standard_cond = cond[:, self.standard_cond_indices]
        index_cond = cond[:, self.index_cond_indices]
        log_prob_table = self.construct_log_prob_table(standard_cond)
        index_outcome_idxs = self.index_sample_space.outcome_to_idx(index_cond).long()
        return log_prob_table.gather(
            2, index_outcome_idxs[:, None, None].expand(-1, log_prob_table.shape[1], 1),
        ).squeeze(2)

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
