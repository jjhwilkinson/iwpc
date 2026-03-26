from abc import ABC, abstractmethod
from itertools import product, chain
from typing import Iterable, Tuple, Iterator, List, Callable

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding
from iwpc.learn_dist.kernels.cut_kernel import CutKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import FiniteSampleSpace, ExplicitFiniteSampleSpace, \
    ConcatenatedFiniteSampleSpace
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase, ConcatenatedKernel, ConditionedKernel
from iwpc.models.utils import basic_model_factory


def sample_idx_from_logits(logits: Tensor) -> Tensor:
    """
    Given a Tensor containing N rows of K logits, randomly samples an integer between 0 and K-1 from the implied
    distribution

    Parameters
    ----------
    logits
        A 2D tensor of shape (N, K)

    Returns
    -------
    Tensor
        A float tensor containing the sampled integers of shape (N,)
    """
    with torch.no_grad():
        probs = logits.softmax(dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        rand = torch.rand(size=(logits.shape[0], 1), device=logits.device)
        samples = ((cum_probs - rand) > 0).float().argmax(dim=-1)
    return samples


class FiniteKernelInterface(ABC):
    """
    Abstract interface for all finite discrete kernels. Since there are a finite number of possible outcomes, a mapping
    can be constructed between the sample space of size K and the integers [0, K-1]. Provides many operations and
    utilities common to finite kernels
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
        return self.sample_space.idx_to_outcome(sample_idx_from_logits(self.construct_logits(cond)))

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
        return self.construct_logits(cond).log_softmax(dim=-1)[range(cond.shape[0]), idxs]

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
            return FiniteConcatenatedKernel.merge(self, other, True)
        return super().__add__(other)

    def __or__(self, other):
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
        return FiniteCutKernel(
            self,
            cut_fn,
        )

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
        """
        if isinstance(num_variable_outcomes, int):
            num_variable_outcomes = (num_variable_outcomes,)

        sample_space = ExplicitFiniteSampleSpace(torch.tensor([
            torch.unravel_index(outcome_idx, num_variable_outcomes)
            for outcome_idx in torch.arange(np.prod(num_variable_outcomes))
        ]), self.outcome_to_idx)

        super().__init__(sample_space, len(num_variable_outcomes), cond)
        self.num_variable_outcomes = num_variable_outcomes
        self.logit_model = basic_model_factory(
            cond,
            LogSoftmaxEncoding(self.sample_space.num_outcomes)
        ) if logit_model is None else logit_model
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

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tensor of shape (cond.shape[0], self.sample_dimension) of outcomes, and a Tensor of shape (N,) containing
            the corresponding log probabilities
        """
        log_probs = self.construct_logits(cond).log_softmax(dim=-1)
        sample_idxs = sample_idx_from_logits(log_probs)
        samples = self.sample_space.idx_to_outcome(sample_idxs)
        sample_log_probs = log_probs[range(cond.shape[0]), sample_idxs]
        return samples, sample_log_probs

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


class FiniteConcatenatedKernel(FiniteKernelInterface, ConcatenatedKernel):
    """
    Utility kernel that merges any number of sub-kernels to produce samples that are concatenations of samples drawn
    from its sub-kernels. ConcatenatedKernel implementation extended to comply with the FiniteKernelInterface
    """
    def __init__(self, sub_kernels: List[FiniteKernel], concatenate_cond=False):
        """
        Parameters
        ----------
        sub_kernels
            A list of TrainableKernelBase sub-kernels
        concatenate_cond
            Whether the conditioning information spaced should be concatenated, or are the same for all sub-kernels
        """
        super().__init__(
            ConcatenatedFiniteSampleSpace([k.sample_space for k in sub_kernels]),
            sub_kernels,
            concatenate_cond,
        )

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Constructs the logits over the possible outcomes given the conditioning information using p(A and B) = p(A) p(B)
        if A and B are independent

        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes)
        """
        sub_logits = []
        for i, (cond_edges, sub_kernel) in enumerate(zip(self.cond_edges, self.sub_kernels)):
            sub_logit = sub_kernel.construct_logits(cond[:, cond_edges])
            sub_logit = sub_logit.reshape((cond.shape[0],) + (1,) * i + (-1,) + (1,) * (len(self.sub_kernels) - i - 1))
            sub_logits.append(sub_logit)
        return sum(sub_logits).reshape((cond.shape[0], self.sample_space.num_outcomes))

    def outcomes_with_log_prob_iter(self, cond: Tensor) -> Iterator[tuple[Tensor, Tensor]]:
        """
        Iterate over the set of possible outcomes alongside the associated log-probability for the given conditioning
        information

        Parameters
        ----------
        cond
            The condition information

        Returns
        -------
        Iterator[tuple[Tensor, Tensor]]
            Iterator over the set of possible outcomes alongside the associated log-probability
        """
        outcomes_iters = [
            k.outcomes_with_log_prob_iter(cond[:, cond_edges])
            for cond_edges, k in zip(self.cond_edges, self.sub_kernels)
        ]
        for outcome_idx, sub_outcomes in enumerate(product(*outcomes_iters)):
            sub_samples, log_probs = zip(*sub_outcomes)
            yield (
                torch.concat(sub_samples, dim=0),
                sum(log_probs),
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
            An integer tensor of shape (N,)
        """
        idxs = 0.
        cum_prod = 1.
        for sample_edges, sub_kernel in zip(self.sample_edges[::-1], self.sub_kernels[::-1]):
            idxs += sub_kernel.outcome_to_idx(samples[:, sample_edges]) * cum_prod
            cum_prod *= sub_kernel.num_outcomes
        return idxs.int()


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
        sample_idxs = sample_idx_from_logits(log_probs)
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


class FiniteConditionedKernel(FiniteKernelInterface, ConditionedKernel):
    """
    ConditionedKernel implementation that also satisfies the FiniteKernelInterface
    """
    def __init__(
        self,
        sample_kernel: FiniteKernelInterface,
        conditioning_kernel: FiniteKernelInterface,
    ):
        """
        Parameters
        ----------
        sample_kernel
            A finite kernel that satisfies sample_kernel.cond_dimension == conditioning_kernel.cond_dimension + conditioning_kernel.sample_dimension
            the kernel should expect the first sample_kernel.cond_dimension components of its conditioning information
            to originate from the conditioning_kernel
        conditioning_kernel
            A finite kernel that produces samples upon which the sample kernel above is additionally conditioned
        """
        assert sample_kernel.cond_dimension == (conditioning_kernel.sample_dimension + conditioning_kernel.cond_dimension)
        super().__init__(
            ConcatenatedFiniteSampleSpace([sample_kernel.sample_space, conditioning_kernel.sample_space]),
            sample_kernel,
            conditioning_kernel,
        )

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Constructs the logits over the possible outcomes given the conditioning information using p(A and B) = p(A|B) p(B)

        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes)
        """
        outputs = []

        conditioning_logits = self.conditioning_kernel.construct_logits(cond)
        for idx, outcome in enumerate(self.conditioning_kernel.sample_space.outcomes_iter()):
            full_cond = torch.concat([outcome.repeat(cond.shape[0], 1), cond], dim=1)
            sample_kernel_logits = self.sample_kernel.construct_logits(full_cond)
            outputs.append(sample_kernel_logits + conditioning_logits[:, idx:idx+1])

        return torch.concat(outputs, dim=1)

    def outcome_to_idx(self, samples: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of size (N, self.sample_dimension) of integers

        Returns
        -------
        Tensor
            An integer tensor of shape (N,)
        """
        samples_kernel_idxs = self.conditioning_kernel.sample_space.outcome_to_idx(samples[:, :self.sample_kernel.sample_dimension])
        cond_kernel_idxs = self.conditioning_kernel.sample_space.outcome_to_idx(samples[:, self.sample_kernel.sample_dimension:])
        return samples_kernel_idxs + cond_kernel_idxs * self.sample_kernel.sample_space.num_outcomes
