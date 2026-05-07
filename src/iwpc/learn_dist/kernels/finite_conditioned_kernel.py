import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ConcatenatedFiniteSampleSpace
from iwpc.learn_dist.kernels.conditioned_kernel import ConditionedKernel
from iwpc.learn_dist.kernels.indexed_interface import IndexedInterface


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

    def construct_log_probs(self, cond: Tensor) -> Tensor:
        """
        Computes the joint log probability table log p(A, B | cond) = log p(A | B, cond) + log p(B | cond) directly in
        log-prob space. When the sample kernel is an IndexedInterface, the joint table is assembled from the children's
        pre-normalised log-prob tables via a single broadcasted addition; otherwise the conditioning kernel's outcomes
        are enumerated one at a time and each branch is summed independently

        Parameters
        ----------
        cond
            A tensor of conditioning information of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of shape (N, self.sample_space.num_outcomes) where entry (n, k) is the log probability of joint
            outcome k given cond[n]. Outcome ordering follows ConcatenatedFiniteSampleSpace's convention
            (sample-kernel index slowest, conditioning-kernel index fastest)
        """
        if cond.shape[0] == 0:
            return torch.zeros((0, self.sample_space.num_outcomes), device=cond.device, dtype=cond.dtype)

        cond_log_probs = self.conditioning_kernel.construct_log_probs(cond)
        if isinstance(self.sample_kernel, IndexedInterface):
            z = cond[:, self.sample_kernel.standard_cond_indices - self.conditioning_kernel.sample_dimension]
            sample_log_probs = self.sample_kernel.construct_log_prob_table(z)        # (N, M, K)
            joint = sample_log_probs + cond_log_probs.unsqueeze(1)                   # (N, M, K)
            return joint.reshape(cond.shape[0], -1)

        outputs = []
        for b_idx, outcome in enumerate(self.conditioning_kernel.sample_space.outcomes_iter()):
            full_cond = torch.concat([outcome.repeat((cond.shape[0], 1)), cond], dim=1)
            sample_log_probs = self.sample_kernel.construct_log_probs(full_cond)
            outputs.append(sample_log_probs + cond_log_probs[:, b_idx:b_idx + 1])

        return torch.stack(outputs, dim=2).reshape((cond.shape[0], -1))

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Delegates to ConditionedKernel._draw, which first draws an outcome from the conditioning kernel and then draws
        from the sample kernel using that outcome appended to cond. Defined explicitly here so the FiniteKernelInterface
        MRO does not shadow the ConditionedKernel implementation

        Parameters
        ----------
        cond
            Conditioning tensor of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            Samples drawn from the joint distribution, shape (N, self.sample_dimension)
        """
        return ConditionedKernel._draw(self, cond)

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
        return samples_kernel_idxs * self.conditioning_kernel.sample_space.num_outcomes + cond_kernel_idxs
