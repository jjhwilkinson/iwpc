from typing import List, Iterator

import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ConcatenatedFiniteSampleSpace
from iwpc.learn_dist.kernels.concatenated_kernel import ConcatenatedKernel


class FiniteConcatenatedKernel(FiniteKernelInterface, ConcatenatedKernel):
    """
    Utility kernel that merges any number of sub-kernels to produce samples that are concatenations of samples drawn
    from its sub-kernels. ConcatenatedKernel implementation extended to comply with the FiniteKernelInterface
    """
    def __init__(self, sub_kernels: List[FiniteKernelInterface], concatenate_cond=False):
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

    def construct_log_probs(self, cond: Tensor) -> Tensor:
        """
        Computes log p(A_1, ..., A_n) = sum_i log p(A_i) using the independence assumption.

        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes) of joint log-probabilities.
        """
        sub_log_probs = []
        for i, (cond_edges, sub_kernel) in enumerate(zip(self.cond_edges, self.sub_kernels)):
            sub_log_prob = sub_kernel.construct_log_probs(cond[:, cond_edges])
            sub_log_prob = sub_log_prob.reshape((cond.shape[0],) + (1,) * i + (-1,) + (1,) * (len(self.sub_kernels) - i - 1))
            sub_log_probs.append(sub_log_prob)
        return sum(sub_log_probs).reshape((cond.shape[0], self.sample_space.num_outcomes))

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
        idxs = 0
        cum_prod = 1
        for sample_edges, sub_kernel in zip(self.sample_edges[::-1], self.sub_kernels[::-1]):
            idxs += sub_kernel.outcome_to_idx(samples[:, sample_edges]) * cum_prod
            cum_prod *= sub_kernel.num_outcomes
        return idxs.int()

    def _draw(self, cond: Tensor) -> Tensor:
        return ConcatenatedKernel._draw(self, cond)


if __name__ == "__main__":
    import torch
    from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel

    torch.manual_seed(0)

    # joint: p(A, B | z) = p(A | z) p(B | z), independent
    kernel_a = FiniteKernel(3, 4)
    kernel_b = FiniteKernel(2, 4)
    joint = kernel_a & kernel_b
    joint.eval()

    N = 1
    cond = torch.randn(N, joint.cond_dimension)

    with torch.no_grad():
        log_probs_from_kernel = joint.construct_log_probs(cond)

        for idx in range(joint.sample_space.num_outcomes):
            outcome = joint.sample_space.idx_to_outcome(torch.full((N,), idx, dtype=torch.long))
            lp_kernel = log_probs_from_kernel[:, idx]
            lp_log_prob = joint.log_prob(outcome, cond)
            max_diff = (lp_kernel - lp_log_prob).abs().max().item()
            assert max_diff < 1e-5, f"Mismatch at outcome {idx}: max diff {max_diff}"

    print("All outcomes match.")