from abc import ABC, abstractmethod
from typing import Iterator, Callable

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.learn_dist.kernels.concatenated_kernel import ConcatenatedKernel
from iwpc.learn_dist.kernels.conditioned_kernel import ConditionedKernel


def sample_idx_from_log_probs(log_probs: Tensor) -> Tensor:
    """
    Given a Tensor containing N rows of K log-probabilities, randomly samples an integer between 0 and K-1 from the
    implied distribution

    Parameters
    ----------
    log_probs
        A 2D tensor of shape (N, K) of log-probabilities

    Returns
    -------
    Tensor
        A float tensor containing the sampled integers of shape (N,)
    """
    with torch.no_grad():
        probs = log_probs.exp()
        cum_probs = torch.cumsum(probs, dim=-1)
        rand = torch.rand(size=(log_probs.shape[0], 1), device=log_probs.device)
        samples = ((cum_probs - rand) > 0).float().argmax(dim=-1)
    return samples


class FiniteKernelInterface(ABC):
    """
    Abstract interface for all finite discrete kernels. Since there are a finite number of possible outcomes, a mapping
    can be constructed between the sample space of size K and the integers [0, K-1]. Provides many operations and
    utilities common to finite kernels.

    Warning
    -------
    Avoid overriding ``log_prob`` or ``draw_with_log_prob`` unless you know what you are doing. These methods are
    implemented here via a single ``construct_logits`` call followed by one ``log_softmax``, which is the numerically
    optimal path for composite finite kernels. Overriding them to chain sub-kernel ``log_prob`` calls introduces
    cascading ``log_softmax`` operations whose float32 rounding errors accumulate in deep kernel compositions. Instead,
    override ``construct_logits`` and let the interface handle normalisation.
    """

    def __init__(self, sample_space: FiniteSampleSpace, *args, **kwargs):
        """
        Parameters
        ----------
        sample_space
            A FiniteSampleSpace instance describing the set of possible outcomes
        args
            Passed to super constructor
        kwargs
            Passed to super constructor
        """
        super().__init__(*args, **kwargs)
        self.sample_space = sample_space

    @abstractmethod
    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Returns a set of logits over the self.num_outcomes possible outcomes for the given condition information

        Parameters
        ----------
        cond
            The condition information of shape (N, K)

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes)
        """
        pass

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            The condition information

        Returns
        -------
        Tensor
            A sample drawn from the distribution over self.outcomes
        """
        return self.sample_space.idx_to_outcome(sample_idx_from_log_probs(self.construct_logits(cond).log_softmax(dim=-1)))

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Calculates the log probability of the given samples for the given conditioning information

        Parameters
        ----------
        samples
            A tensor of size (N, self.sample_dimension) of outcomes
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            A tensor of shape (N,)
        """
        idxs = self.sample_space.outcome_to_idx(samples)
        return self.construct_logits(cond).log_softmax(dim=-1).gather(1, idxs.unsqueeze(1)).squeeze(1)

    def draw_with_log_prob(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Draw a sample and return its log-probability in a single construct_logits call.

        Parameters
        ----------
        cond
            The condition information of shape (N, K)

        Returns
        -------
        tuple[Tensor, Tensor]
            Samples of shape (N, sample_dimension) and log-probabilities of shape (N,)
        """
        logits = self.construct_logits(cond)
        log_probs = logits.log_softmax(dim=-1)
        sample_idxs = sample_idx_from_log_probs(log_probs)
        return self.sample_space.idx_to_outcome(sample_idxs), log_probs[range(cond.shape[0]), sample_idxs]

    def outcomes_with_log_prob_iter(self, cond: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
        """
        Iterate over the set of possible outcomes alongside the associated log-probability for the given conditioning
        information

        Parameters
        ----------
        cond
            The condition information of shape (N, K)

        Returns
        -------
        Iterator[tuple[Tensor, Tensor]]
            Iterator over the set of possible outcomes alongside the associated log-probability
        """
        logits = self.construct_logits(cond)
        log_probs = logits.log_softmax(dim=-1)
        for outcome, logit in zip(self.sample_space.outcomes_iter(), log_probs.T):
            yield outcome, logit

    def __and__(self, other: TrainableKernelBase) -> 'FiniteConcatenatedKernel | ConcatenatedKernel':
        """
        Overrides the default '&'-operator behavior when both kernels are finite to return an instance of
        FiniteConcatenatedKernel. If the other kernel is not finite, then the standard ConcatenatedKernel is returned

        Parameters
        ----------
        other
            TrainableKernelBase

        Returns
        -------
        FiniteConcatenatedKernel | ConcatenatedKernel
            A FiniteConcatenatedKernel if 'other' is an instance of FiniteKernelInterface
        """
        if isinstance(other, FiniteKernelInterface):
            from iwpc.learn_dist.kernels.finite_concatenated_kernel import FiniteConcatenatedKernel
            return FiniteConcatenatedKernel.merge(self, other, False)
        return super().__and__(other)

    def __add__(self, other: TrainableKernelBase) -> 'FiniteConcatenatedKernel | ConcatenatedKernel':
        """
        Overrides the default '+'-operator behavior when both kernels are finite to return an instance of
        FiniteConcatenatedKernel. If the other kernel is not finite, then the standard ConcatenatedKernel is returned

        Parameters
        ----------
        other
            TrainableKernelBase

        Returns
        -------
        FiniteConcatenatedKernel | ConcatenatedKernel
            A FiniteConcatenatedKernel if 'other' is an instance of FiniteKernelInterface
        """
        if isinstance(other, FiniteKernelInterface):
            from iwpc.learn_dist.kernels.finite_concatenated_kernel import FiniteConcatenatedKernel
            return FiniteConcatenatedKernel.merge(self, other, True)
        return super().__add__(other)

    def __or__(self, other) -> ConditionedKernel:
        """
        Overrides the default '|'-operator behavior when both kernels are finite to return an instance of
        FiniteConcatenatedKernel. If the other kernel is not finite, then the standard ConcatenatedKernel is returned

        Parameters
        ----------
        other
            TrainableKernelBase

        Returns
        -------
        FiniteConcatenatedKernel | ConcatenatedKernel
            A FiniteConcatenatedKernel if 'other' is an instance of FiniteKernelInterface
        """
        if isinstance(other, FiniteKernelInterface):
            from iwpc.learn_dist.kernels.finite_conditioned_kernel import FiniteConditionedKernel
            return FiniteConditionedKernel(self, other)
        return super().__or__(other)

    def cut(self, cut_fn: Callable[[Tensor], bool]) -> "FiniteCutKernel":
        """
        Utility method to construct a FiniteCutKernel from this finite kernel

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
        from iwpc.learn_dist.kernels.finite_cut_kernel import FiniteCutKernel
        return FiniteCutKernel(self, cut_fn)

    def __ror__(self, other: list[TrainableKernelBase]) -> ConditionedKernel:
        """
        Syntactic sugar to construct a ConditionedKernel from a list of TrainableKernelBase instances. The conditioned
        kernel samples from a branching kernel that branches based upon the outcome index of this finite kernel

        Parameters
        ----------
        other
            A list of TrainableKernelBase instances of length self.num_outcomes

        Returns
        -------
        ConditionedKernel
            A ConditionedKernel that samples from a BranchingKernel that branches based upon the outcome index of this
            finite kernel
        """
        from iwpc.learn_dist.kernels.branching_kernel import BranchingKernel, FiniteBranchingKernel

        if not isinstance(other, list):
            return other.__or__(self)
        if all(isinstance(k, FiniteKernelInterface) for k in other):
            branching_kernel = FiniteBranchingKernel.condition_on(other, self)
        else:
            branching_kernel = BranchingKernel.condition_on(other, self)
        return branching_kernel | self
