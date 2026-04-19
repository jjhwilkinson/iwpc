from typing import Callable, Iterator, Tuple

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.cut_kernel import CutKernelInterface
from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface, sample_idx_from_log_probs
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class FiniteCutKernel(CutKernelInterface, FiniteKernelInterface, TrainableKernelBase):
    """
    Finite kernel derived from an underlying finite 'base_kernel' wherein some outcomes are 'cut-out'. In effect
    providing a distribution over the remaining outcomes
    """
    def __init__(
        self,
        base_kernel: FiniteKernelInterface,
        cut_fn: Callable[[Tensor], bool],
    ):
        """
        Parameters
        ----------
        base_kernel
            A base finite kernel
        cut_fn
            A function that accepts an outcome from the base kernel and returns a boolean on whether the outcome passes
            the cut (True) or is cut-out (False)
        """
        self.allowed_indices = [idx for idx, outcome in enumerate(base_kernel.sample_space.outcomes_iter()) if cut_fn(outcome)]
        self.disallowed_indices = [idx for idx, outcome in enumerate(base_kernel.sample_space.outcomes_iter()) if not cut_fn(outcome)]

        super().__init__(
            base_kernel.sample_space.cut(cut_fn),
            base_kernel.sample_dimension,
            base_kernel.cond_dimension,
        )
        self.cut_fn = cut_fn
        self.base_kernel = base_kernel

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Constructs the logits over the possible outcomes given the conditioning information. Since logits are not
        normalized, we are free to use the same logits as the base kernel.

        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes)
        """
        return self.base_kernel.construct_logits(cond)[:, self.allowed_indices]

    def cut_pass_log_prob(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            The log-probability that a sample from the base kernel is one of the allowed outcomes
        """
        return self.base_kernel.construct_logits(cond).log_softmax(dim=-1)[:, self.allowed_indices].logsumexp(dim=-1)

    def cut_fail_log_prob(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            The log-probability that a sample from the base kernel is one of the dis-allowed outcomes
        """
        return self.base_kernel.construct_logits(cond).log_softmax(dim=-1)[:, self.disallowed_indices].logsumexp(dim=-1)

    def draw_with_log_prob_and_cut_pass_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            1. A sample drawn from the cut-kernel for each row of conditioning information
            2. The log-probability of observing the above sample for the given conditioning information
            3. The log-probability that a sample from the base kernel passes the cut for each row of conditioning information
        """
        base_log_probs = self.base_kernel.construct_logits(cond).log_softmax(dim=-1)
        cut_pass_log_probs = base_log_probs[:, self.allowed_indices].logsumexp(dim=-1)
        log_probs = base_log_probs[:, self.allowed_indices] - cut_pass_log_probs
        sample_idxs = sample_idx_from_log_probs(log_probs)
        return self.outcomes[sample_idxs], log_probs, cut_pass_log_probs

    def pass_log_prob_and_outcomes_with_log_prob_iter(self, cond: Tensor) -> tuple[Tensor, Iterator[tuple[Tensor, Tensor]]]:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        1. The probability that a sample from the base kernel passes the cut for each row of conditioning information
        2. An iterator over the allowed outcomes of the cut-kernel for each row of conditioning information and the
            log-probability of observing said outcome
        """
        base_log_probs = self.base_kernel.construct_logits(cond).log_softmax(dim=-1)
        pass_log_probs = base_log_probs[:, self.allowed_indices].logsumexp(dim=-1)
        log_probs = base_log_probs[:, self.allowed_indices] - pass_log_probs

        def outcomes_with_log_prob_iter() -> Iterator[tuple[Tensor, Tensor]]:
            """
            Wrapper that iterates over the allowed outcomes of the cut-kernel for each row of conditioning information

            Returns
            -------
            Iterator[tuple[Tensor, Tensor]]
                An iterator over the allowed outcomes of the cut-kernel for each row of conditioning information and the
                log-probability of observing said outcome
            """
            for outcome, log_prob in zip(self.outcomes, log_probs.T):
                yield outcome, log_prob
        return pass_log_probs, outcomes_with_log_prob_iter()

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.sample_dimension) of integers
        cond
            The conditioning information

        Returns
        -------
        Tensor
            The probability of observing each sample given the sample passes the cut for each row of conditioning information
        """
        base_log_probs = self.base_kernel.construct_logits(cond).log_softmax(dim=-1)
        cut_pass_log_probs = base_log_probs[:, self.allowed_indices].logsumexp(dim=-1)
        return base_log_probs[
            range(base_log_probs.shape[0]),
            self.base_kernel.sample_space.outcome_to_idx(samples)
        ] - cut_pass_log_probs

    def cut(self, cut_fn: Callable[[Tensor], bool]) -> "FiniteCutKernel":
        """
        Utility method to construct a FiniteCutKernel from this finite kernel. Un-curries the nested cuts
        into a single FiniteCutKernel instance with cut given by combining the results of self.cut_fn and the provided
        cut_fn

        Parameters
        ----------
        cut_fn
            A function that accepts outcomes from this kernel and returns a boolean of whether the outcome passes the
            cut (True) or not (False).

        Returns
        -------
        FiniteCutKernel
            A FiniteCutKernel over the outcomes that pass the cut
        """
        return FiniteCutKernel(
            self.base_kernel,
            lambda x: cut_fn(x) & self.cut_fn(x),
        )