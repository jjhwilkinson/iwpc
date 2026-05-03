from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace
from iwpc.learn_dist.kernels.indexed_interface import IndexedInterface
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


class IndexedFiniteKernel(IndexedInterface, FiniteKernel):
    """
    A FiniteKernel that models p(A | B=b, x) where B takes on K discrete values and A takes on M discrete values.
    Log probabilities are produced by predicting a single K×M logit table from one logit_model(x)
    call and selecting the b-th column. This collapses the K separate forward passes that
    FiniteConditionedKernel would otherwise require into one.

    The conditioning is split into a discrete index part B given by cond[:, index_cond_indices], and the remaining
    components form the standard non-indexed conditioning information, x. The logit model takes only x and outputs
    (N, M*K) laid out in M-major order (M contiguous blocks of K logits), which is reshaped directly to (N, M, K) so
    that column k holds the M logits for index value k.
    """
    def __init__(
        self,
        num_variable_outcomes: int | Iterable[int],
        unindexed_cond: Encoding | int,
        index_cond_indices: list[int] | int,
        index_sample_space: FiniteSampleSpace,
        logit_model: torch.nn.Module | None = None,
        init_log_probs: float | Iterable[float] | Iterable[Iterable[float]] | None = None,
    ):
        """
        Parameters
        ----------
        num_variable_outcomes
            Number of possible values per sample variable (see FiniteKernel)
        unindexed_cond
            The encoding or dimension of the standard conditioning x passed to the logit model
        index_cond_indices
            Columns of the full cond tensor that carry the discrete index b. An int N is treated as
            list(range(N)).
        index_sample_space
            The FiniteSampleSpace of the discrete index b, providing outcome_to_idx and num_outcomes (K)
        logit_model
            Optional custom logit model. Must accept x of shape (N, standard_cond_dim) and return
            (N, M*K) logits. If None, a default model is constructed via basic_model_factory.
        init_log_probs
            Optional initial log-probability bias. A float init_log_probs initialises a binary kernel with the same
            [log(1-exp(init_log_probs)), init_log_probs] shift for all K index values. A flat iterable of length M is
            broadcast uniformly across all K index values. A M×K nested iterable provides a distinct initial
            log-prob per outcome per index value. Ignored if logit_model is provided.
        """
        if isinstance(num_variable_outcomes, int):
            num_variable_outcomes = (num_variable_outcomes,)
        if isinstance(index_cond_indices, int):
            index_cond_indices = list(range(index_cond_indices))

        K = index_sample_space.num_outcomes
        M = int(np.prod(num_variable_outcomes))
        standard_cond_dim = int(unindexed_cond.input_shape[0]) if isinstance(unindexed_cond, Encoding) else int(unindexed_cond)
        total_cond_dim = len(index_cond_indices) + standard_cond_dim

        if logit_model is None:
            if init_log_probs is not None:
                init_log_probs = np.asarray(init_log_probs)
                if init_log_probs.ndim == 0:
                    if M != 2:
                        raise ValueError(f"A scalar init_log_probs can only be used with binary kernels (2 outcomes), got {M}")
                    init_log_probs = np.asarray([np.log1p(-np.exp(init_log_probs)), init_log_probs])
                    init_log_probs = np.tile(init_log_probs[:, None], (1, K))
                else:
                    if init_log_probs.ndim == 1:
                        if M != 2:
                            raise ValueError(f"A vector init_log_probs can only be used with binary kernels (2 outcomes), got {M}")
                        init_log_probs = np.stack([np.log1p(-np.exp(init_log_probs)), init_log_probs], axis=0)
                final_layers = [ConstantScaleLayer(shift=init_log_probs)]
            else:
                final_layers = []
            logit_model = basic_model_factory(unindexed_cond, TrivialEncoding(M * K), final_layers=final_layers)

        super().__init__(
            index_sample_space,
            index_cond_indices,
            num_variable_outcomes,
            total_cond_dim,
            logit_model=logit_model,
        )

    @classmethod
    def condition_on(
        cls,
        num_variable_outcomes: int | Iterable[int],
        conditioning_kernel: FiniteKernelInterface,
        standard_cond: Encoding | int,
        **kwargs,
    ) -> 'IndexedFiniteKernel':
        """
        Construct an IndexedFiniteKernel over the specified num_variable_outcomes using the output of the given finite
        conditioning kernel as the indexed conditioning information, B. The resulting kernel expects cond of shape
        (N, conditioning_kernel.sample_dimension + conditioning_kernel.conditioning_dimension), matching
        FiniteConditionedKernel's convention of prepending the b outcome to the conditioning information of
        conditioning_kernel.

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
            list(range(conditioning_kernel.sample_dimension)),
            conditioning_kernel.sample_space,
            **kwargs,
        )

    def construct_log_prob_table(self, unindexed_cond: Tensor) -> Tensor:
        """
        Returns the full log-probability table for all index values in a single forward pass.

        Parameters
        ----------
        unindexed_cond
            The standard conditioning x of shape (N, standard_cond_dim)

        Returns
        -------
        Tensor
            A tensor of shape (N, M, K) where M = sample_space.num_outcomes and K = index_sample_space.num_outcomes.
            Column k holds ``log p(A=m | B=k, x)``.
        """
        N = unindexed_cond.shape[0]
        return self.logit_model(unindexed_cond).reshape(
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
            A tensor of size (N, M) of log-probabilities over the sample space for the given conditioning information.
        """
        x = cond[:, self.standard_cond_indices]
        b = cond[:, self.index_cond_indices]
        table = self.construct_log_prob_table(x)
        idxs = self.index_sample_space.outcome_to_idx(b).long()
        return table.gather(2, idxs.long()[:, None, None].expand(-1, table.shape[1], 1)).squeeze(2)
