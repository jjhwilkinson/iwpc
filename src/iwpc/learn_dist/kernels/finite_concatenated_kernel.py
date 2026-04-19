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
        logits = joint.construct_logits(cond)
        log_probs_from_logits = logits.log_softmax(dim=-1)

        for idx in range(joint.sample_space.num_outcomes):
            outcome = joint.sample_space.idx_to_outcome(torch.full((N,), idx, dtype=torch.long))
            lp_logits = log_probs_from_logits[:, idx]
            lp_log_prob = joint.log_prob(outcome, cond)
            max_diff = (lp_logits - lp_log_prob).abs().max().item()
            assert max_diff < 1e-5, f"Mismatch at outcome {idx}: max diff {max_diff}"

    print("All outcomes match.")