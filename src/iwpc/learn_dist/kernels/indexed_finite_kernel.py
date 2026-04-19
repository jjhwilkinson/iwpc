from typing import Iterable

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


class IndexedFiniteKernel(FiniteKernel):
    """
    A FiniteKernel that models p(A | B=b, x) by producing a single K×M logit table from one logit_model(x)
    call and selecting the b-th column. This collapses the K separate forward passes that
    FiniteConditionedKernel would otherwise require into one.

    The conditioning is split into a discrete index part b (given by index_cond_indices) and a standard
    continuous part x. The logit model takes only x and outputs (N, K*M), which is reshaped to (N, K, M)
    and transposed to (N, M, K) so that column k holds the M logits for index value k.
    """
    def __init__(
        self,
        num_variable_outcomes: int | Iterable[int],
        standard_cond: Encoding | int,
        index_cond_indices: list[int] | int,
        index_sample_space: FiniteSampleSpace,
        logit_model: torch.nn.Module | None = None,
        init_prob: float | Iterable[float] | Iterable[Iterable[float]] | None = None,
    ):
        """
        Parameters
        ----------
        num_variable_outcomes
            Number of possible values per sample variable (see FiniteKernel)
        standard_cond
            The encoding or dimension of the standard conditioning x passed to the logit model
        index_cond_indices
            Columns of the full cond tensor that carry the discrete index b. An int N is treated as
            list(range(N)).
        index_sample_space
            The FiniteSampleSpace of the discrete index b, providing outcome_to_idx and num_outcomes (K)
        logit_model
            Optional custom logit model. Must accept x of shape (N, standard_cond_dim) and return
            (N, K*M) laid out as K contiguous blocks of M logits — reshape(N, K, M) gives block k the
            logits for index value k. If None, a default model is constructed via basic_model_factory.
        init_prob
            Optional initial probability bias. A float p initialises a binary kernel with the same
            [log(1-p), log(p)] shift for all K index values. A flat iterable of length M is broadcast
            uniformly across all K index values. A K×M nested iterable provides a distinct initial
            distribution per index value. Ignored if logit_model is provided.
        """
        if isinstance(num_variable_outcomes, int):
            num_variable_outcomes = (num_variable_outcomes,)
        if isinstance(index_cond_indices, int):
            index_cond_indices = list(range(index_cond_indices))

        K = index_sample_space.num_outcomes
        M = int(np.prod(num_variable_outcomes))
        standard_cond_dim = int(standard_cond.input_shape[0]) if isinstance(standard_cond, Encoding) else int(standard_cond)
        total_cond_dim = len(index_cond_indices) + standard_cond_dim

        if logit_model is None:
            if init_prob is not None:
                if isinstance(init_prob, float):
                    if M != 2:
                        raise ValueError(f"A scalar init_prob can only be used with binary kernels (2 outcomes), got {M}")
                    probs_per_b = [[1 - init_prob, init_prob]] * K
                else:
                    probs_list = list(init_prob)
                    if isinstance(probs_list[0], (list, tuple)):
                        probs_per_b = [list(row) for row in probs_list]
                    else:
                        probs_per_b = [probs_list] * K
                final_layers = [ConstantScaleLayer(shift=[np.log(p) for row in probs_per_b for p in row])]
            else:
                final_layers = []
            logit_model = basic_model_factory(standard_cond, TrivialEncoding(K * M), final_layers=final_layers)

        super().__init__(num_variable_outcomes, total_cond_dim, logit_model=logit_model)

        self.index_cond_indices = index_cond_indices
        self.index_sample_space = index_sample_space
        self.standard_cond_indices = [i for i in range(total_cond_dim) if i not in index_cond_indices]

    @classmethod
    def condition_on(
        cls,
        num_variable_outcomes: int | Iterable[int],
        conditioning_kernel: FiniteKernelInterface,
        standard_cond: Encoding | int,
        **kwargs,
    ) -> 'IndexedFiniteKernel':
        """
        Construct an IndexedFiniteKernel paired with the given conditioning kernel for use inside a
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
            list(range(conditioning_kernel.sample_dimension)),
            conditioning_kernel.sample_space,
            **kwargs,
        )

    def construct_logit_table(self, standard_cond: Tensor) -> Tensor:
        """
        Returns the full logit table for all index values in a single forward pass.

        Parameters
        ----------
        standard_cond
            The standard conditioning x of shape (N, standard_cond_dim)

        Returns
        -------
        Tensor
            A tensor of shape (N, M, K) where M = sample_space.num_outcomes and K = index_sample_space.num_outcomes.
            Column k holds the M logits for index value k.
        """
        N = standard_cond.shape[0]
        return self.logit_model(standard_cond).reshape(N, self.index_sample_space.num_outcomes, -1).transpose(1, 2)

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of size (N, M) containing logits for the b-th index value in each row
        """
        x = cond[:, self.standard_cond_indices]
        b = cond[:, self.index_cond_indices]
        table = self.construct_logit_table(x)                                              # (N, M, K)
        idxs = self.index_sample_space.outcome_to_idx(b).long()
        return table.gather(2, idxs[:, None, None].expand(-1, table.shape[1], 1)).squeeze(2)  # (N, M)