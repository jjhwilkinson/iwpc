import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ConcatenatedFiniteSampleSpace
from iwpc.learn_dist.kernels.conditioned_kernel import ConditionedKernel


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
        if cond.shape[0] == 0:
            return torch.zeros((0, self.sample_space.num_outcomes), device=cond.device, dtype=cond.dtype)

        conditioning_logits = self.conditioning_kernel.construct_logits(cond)

        from iwpc.learn_dist.kernels.indexed_finite_kernel import IndexedFiniteKernel
        if isinstance(self.sample_kernel, IndexedFiniteKernel):
            logit_table = self.sample_kernel.construct_logit_table(cond)  # (N, M, K)
            a_log_probs = logit_table.log_softmax(dim=1)                  # normalise over M per b
            joint = a_log_probs + conditioning_logits.unsqueeze(1)        # (N, M, K) + (N, 1, K)
            return joint.reshape(cond.shape[0], -1)

        outputs = []
        for b_idx, outcome in enumerate(self.conditioning_kernel.sample_space.outcomes_iter()):
            full_cond = torch.concat([outcome.repeat((cond.shape[0], 1)), cond], dim=1)
            sample_log_probs = self.sample_kernel.construct_logits(full_cond).log_softmax(dim=-1)
            outputs.append(sample_log_probs + conditioning_logits[:, b_idx:b_idx + 1])

        return torch.stack(outputs, dim=2).reshape((cond.shape[0], -1))

    def _draw(self, cond: Tensor) -> Tensor:
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


if __name__ == "__main__":
    from iwpc.learn_dist.kernels.finite_conditioned_kernel import FiniteConditionedKernel
    from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
    from iwpc.learn_dist.kernels.indexed_finite_kernel import IndexedFiniteKernel

    torch.manual_seed(0)

    def check_joint(joint, cond, label):
        joint.eval()
        with torch.no_grad():
            logits = joint.construct_logits(cond)
            log_probs_from_logits = logits.log_softmax(dim=-1)
            for idx in range(joint.sample_space.num_outcomes):
                outcome = joint.sample_space.idx_to_outcome(torch.full((N,), idx, dtype=torch.long))
                lp_logits = log_probs_from_logits[:, idx]
                lp_log_prob = joint.log_prob(outcome, cond)
                max_diff = (lp_logits - lp_log_prob).abs().max().item()
                assert max_diff < 1e-5, f"[{label}] Mismatch at outcome {idx}: max diff {max_diff}"
        print(f"[{label}] All outcomes match.")

    N = 4
    cond = torch.randn(N, 4)

    # slow path: plain FiniteKernel sample kernel
    cond_kernel = FiniteKernel(3, 4)
    sample_kernel = FiniteKernel(2, 5)
    joint_slow = sample_kernel | cond_kernel
    check_joint(joint_slow, cond, "slow path")

    # fast path: IndexedFiniteKernel sample kernel
    cond_kernel2 = FiniteKernel(3, 4)
    sample_kernel_indexed = IndexedFiniteKernel.condition_on(2, cond_kernel2, 4)
    joint_fast = sample_kernel_indexed | cond_kernel2
    check_joint(joint_fast, cond, "fast path")
