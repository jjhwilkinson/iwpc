import torch
from torch import Tensor

from iwpc.learn_dist.kernels.finite_kernel_interface import FiniteKernelInterface
from iwpc.learn_dist.kernels.finite_sample_space import ConcatenatedFiniteSampleSpace
from iwpc.learn_dist.kernels.conditioned_kernel import ConditionedKernel
from iwpc.learn_dist.kernels.indexed_interface import IndexedInterface, trivial_index_sample_space


class FiniteConditionedKernel(IndexedInterface, FiniteKernelInterface, ConditionedKernel):
    """
    ConditionedKernel implementation that also satisfies the FiniteKernelInterface.

    Models p(A, B2 | z, [B1]) = p(A | B2, [B1], z) p(B2 | [B1], z). When the conditioning_kernel itself satisfies
    IndexedInterface, the joint kernel inherits its index space (B1) and exposes ``construct_log_prob_table`` so it
    can in turn be composed as the conditioning kernel of a deeper chain. Otherwise B1 is absent and the joint
    kernel still implements IndexedInterface trivially with K_B1 = 1
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

        if isinstance(conditioning_kernel, IndexedInterface):
            index_sample_space = conditioning_kernel.index_sample_space
            index_cond_indices = list(int(i) for i in conditioning_kernel.index_cond_indices)
        else:
            index_sample_space = trivial_index_sample_space()
            index_cond_indices = []

        super().__init__(
            index_sample_space,
            index_cond_indices,
            ConcatenatedFiniteSampleSpace([sample_kernel.sample_space, conditioning_kernel.sample_space]),
            sample_kernel,
            conditioning_kernel,
        )

    def construct_log_probs(self, cond: Tensor) -> Tensor:
        """
        Computes the joint log probability table log p(A, B2 | cond) directly in log-prob space.

        When the joint kernel is itself indexed (i.e. conditioning_kernel is indexed on B1), routes via
        ``construct_log_prob_table`` + gather on the B1 index columns. When the joint kernel is not indexed
        (K_B1 = 1), uses the fast aligned-table path on the sample kernel if applicable, otherwise enumerates
        the conditioning kernel's outcomes one at a time

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
        N = cond.shape[0]
        if N == 0:
            return torch.zeros((0, self.sample_space.num_outcomes), device=cond.device, dtype=cond.dtype)

        if len(self.index_cond_indices) > 0:
            z = cond[:, self.standard_cond_indices]
            b1 = cond[:, self.index_cond_indices]
            table = self.construct_log_prob_table(z)
            idxs = self.index_sample_space.outcome_to_idx(b1).long()
            return table.gather(2, idxs[:, None, None].expand(-1, table.shape[1], 1)).squeeze(2)

        return self._joint_log_probs_for_outer_cond(cond)

    def _joint_log_probs_for_outer_cond(self, outer_cond: Tensor) -> Tensor:
        """
        Compute log p(A, B2 | outer_cond) for a single (or absent) B1, returning (N, M_A * M_B2). Uses the fast
        aligned-table path on the sample kernel when applicable; otherwise enumerates B2 outcomes one at a time.
        Used both as the K_B1 = 1 path of ``construct_log_probs`` and as the inner kernel of the slow B1
        enumeration in ``construct_log_prob_table``

        Parameters
        ----------
        outer_cond
            Full outer conditioning of shape (N, conditioning_kernel.cond_dimension), with the B1 columns (if any)
            already populated

        Returns
        -------
        Tensor
            Shape (N, M_A * M_B2)
        """
        N = outer_cond.shape[0]
        cond_log_probs = self.conditioning_kernel.construct_log_probs(outer_cond)
        if IndexedInterface.is_aligned(self.sample_kernel, self.conditioning_kernel):
            sample_standard_cond = outer_cond[:, self.sample_kernel.standard_cond_indices - self.conditioning_kernel.sample_dimension]
            sample_log_probs = self.sample_kernel.construct_log_prob_table(sample_standard_cond)
            if isinstance(self.conditioning_kernel, IndexedInterface) and len(self.conditioning_kernel.index_cond_indices) > 0:
                K_B2 = self.conditioning_kernel.sample_space.num_outcomes
                K_B1 = self.conditioning_kernel.index_sample_space.num_outcomes
                b1 = outer_cond[:, self.conditioning_kernel.index_cond_indices]
                b1_idxs = self.conditioning_kernel.index_sample_space.outcome_to_idx(b1).long()
                sample_log_probs = sample_log_probs.reshape(N, -1, K_B2, K_B1).gather(
                    3, b1_idxs[:, None, None, None].expand(-1, sample_log_probs.shape[1], K_B2, 1)
                ).squeeze(3)
            joint = sample_log_probs + cond_log_probs.unsqueeze(1)
            return joint.reshape(N, -1)

        outputs = []
        for b_idx, outcome in enumerate(self.conditioning_kernel.sample_space.outcomes_iter()):
            full_cond = torch.concat([outcome.repeat((N, 1)), outer_cond], dim=1)
            sample_log_probs = self.sample_kernel.construct_log_probs(full_cond)
            outputs.append(sample_log_probs + cond_log_probs[:, b_idx:b_idx + 1])
        return torch.stack(outputs, dim=2).reshape(N, -1)

    def construct_log_prob_table(self, cond: Tensor) -> Tensor:
        """
        Computes log p(A, B2 | [B1,] z) for all index values B1 in a single call. Three paths:

        1. K_B1 = 1: degenerates to ``self._joint_log_probs_for_outer_cond(cond)`` with a trailing singleton axis
        2. Conditioning kernel indexed and sample kernel aligned: each child returns a pre-normalised table; the
           joint table is built by a single broadcasted addition (no re-softmax)
        3. Conditioning kernel indexed but sample kernel non-indexed or misaligned: enumerate over B1 outcomes and
           call the single-B1 inner path once per outcome

        Parameters
        ----------
        cond
            Standard conditioning of shape (N, cond_dimension - len(index_cond_indices)) — i.e. the outer cond with
            the B1 columns stripped (or the full outer cond when conditioning_kernel is non-indexed)

        Returns
        -------
        Tensor
            Shape (N, M_A * M_B2, K_B1)
        """
        N = cond.shape[0]
        if N == 0:
            return torch.zeros((0, self.sample_space.num_outcomes, self.index_sample_space.num_outcomes), device=cond.device, dtype=cond.dtype)

        if not isinstance(self.conditioning_kernel, IndexedInterface) or len(self.index_cond_indices) == 0:
            return self._joint_log_probs_for_outer_cond(cond).unsqueeze(-1)

        if IndexedInterface.is_aligned(self.sample_kernel, self.conditioning_kernel):
            sample_log_probs = self.sample_kernel.construct_log_prob_table(cond)
            cond_log_probs = self.conditioning_kernel.construct_log_prob_table(cond)
            K_B1 = self.conditioning_kernel.index_sample_space.num_outcomes
            K_B2 = self.conditioning_kernel.sample_space.num_outcomes
            return (
                sample_log_probs.reshape(N, -1, K_B2, K_B1) + cond_log_probs[:, None]
            ).reshape(N, -1, K_B1)

        outputs = []
        for b1 in self.conditioning_kernel.index_sample_space.outcomes_iter():
            outer_cond = self._inject_index(cond, b1)
            outputs.append(self._joint_log_probs_for_outer_cond(outer_cond))
        return torch.stack(outputs, dim=-1)

    def _inject_index(self, standard_cond: Tensor, index_outcome: Tensor) -> Tensor:
        """
        Re-assemble the full outer cond by scattering ``standard_cond`` into ``self.standard_cond_indices`` and
        ``index_outcome`` into ``self.index_cond_indices``. Used by the slow B1-enumeration path

        Parameters
        ----------
        standard_cond
            Shape (N, len(self.standard_cond_indices))
        index_outcome
            Shape (self.index_sample_space.dimension,) — a single B1 outcome to broadcast across all rows

        Returns
        -------
        Tensor
            Shape (N, self.cond_dimension)
        """
        N = standard_cond.shape[0]
        out = torch.empty(N, self.cond_dimension, dtype=standard_cond.dtype, device=standard_cond.device)
        out[:, self.standard_cond_indices] = standard_cond
        out[:, self.index_cond_indices] = index_outcome.to(out.dtype).to(out.device)[None, :].expand(N, -1)
        return out

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
