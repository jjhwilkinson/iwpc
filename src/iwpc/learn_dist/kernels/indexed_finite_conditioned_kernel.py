from torch import Tensor

from iwpc.learn_dist.kernels.finite_conditioned_kernel import FiniteConditionedKernel
from iwpc.learn_dist.kernels.indexed_finite_kernel import IndexedFiniteKernel
from iwpc.learn_dist.kernels.indexed_interface import IndexedInterface


class IndexedFiniteConditionedKernel(IndexedInterface, FiniteConditionedKernel):
    """
    A FiniteConditionedKernel over the joint space of sample_kernel and conditioning_kernel that satisfies
    IndexedInterface over the indexed space, B1, of the conditioning_kernel

    Models p(A, B2 | z, B1) = p(A | B2, B1, z) p(B2 | B1, z) where:
      - conditioning_kernel is an IndexedFiniteKernel modelling p(B2 | B1, z) indexed over B1
      - sample_kernel is an IndexedInterface modelling p(A | (B2, B1), z) with
        index_cond_indices = list(range(dim_B2)) + [dim_B2 + i for i in conditioning_kernel.index_cond_indices]
        and index_sample_space covering the joint (B2, B1) outcome space

    construct_logit_table(z) takes standard conditioning z (outer cond with B1 stripped) and
    returns (N, M_A * K_B2, K_B1) — joint (A, B2) logits for each B1 value.
    """

    def __init__(
        self,
        sample_kernel: IndexedInterface,
        conditioning_kernel: IndexedInterface,
    ):
        """
        Parameters
        ----------
        sample_kernel
            An IndexedFiniteKernel modelling p(A | B2, B1, z). Its index_cond_indices must equal
            list(range(dim_B2)) + [dim_B2 + i for i in conditioning_kernel.index_cond_indices].
        conditioning_kernel
            An IndexedFiniteKernel modelling p(B2 | B1, z), indexed over B1.
        """
        assert list(sample_kernel.index_cond_indices) == IndexedFiniteConditionedKernel.expected_sample_index_cond_indices(conditioning_kernel), (
            f"sample_kernel.index_cond_indices {list(sample_kernel.index_cond_indices)} does not match "
            f"expected {IndexedFiniteConditionedKernel.expected_sample_index_cond_indices(conditioning_kernel)}"
        )
        super().__init__(
            conditioning_kernel.index_sample_space,
            conditioning_kernel.index_cond_indices,
            sample_kernel,
            conditioning_kernel
        )

    @staticmethod
    def expected_sample_index_cond_indices(conditioning_kernel: IndexedInterface) -> list[int]:
        """
        For a given conditioning kernel, returns the list of index_cond_indices that any compatible sample_kernel must
        satisfy

        Parameters
        ----------
        conditioning_kernel
            An IndexedInterface

        Returns
        -------
        list[int]
            The list of index_cond_indices that any compatible sample_kernel must have
        """
        dim_B2 = conditioning_kernel.sample_dimension
        return list(range(dim_B2)) + [dim_B2 + i for i in conditioning_kernel.index_cond_indices]

    def construct_logit_table(self, unindexed_cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        unindexed_cond
            Standard conditioning of shape (N, standard_cond_dim), i.e. the cond_kernel's conditioning information
            with B1 columns (self.index_cond_indices) stripped.

        Returns
        -------
        Tensor
            Shape (N, M_A * K_B2, K_B1). Column k=b1 holds unnormalised logits for the joint
            (A, B2) outcome proportional to p(A, B2 | B1=b1, z) = p(A | B2, B1, z) * p(B2 | B1, z).
        """
        num_conditioning_kernel_index_outcomes = self.conditioning_kernel.sample_space.num_outcomes
        num_conditioning_kernel_outcomes = self.conditioning_kernel.index_sample_space.num_outcomes

        cond_table = self.conditioning_kernel.construct_logit_table(unindexed_cond)
        sample_table = self.sample_kernel.construct_logit_table(unindexed_cond).log_softmax(dim=1)

        return (
            sample_table.reshape(unindexed_cond.shape[0], -1, num_conditioning_kernel_index_outcomes, num_conditioning_kernel_outcomes)
            + cond_table[:, None]).reshape(unindexed_cond.shape[0], -1, num_conditioning_kernel_outcomes
        )


if __name__ == "__main__":
    import torch
    from iwpc.learn_dist.kernels.finite_sample_space import ExplicitFiniteSampleSpace

    torch.manual_seed(0)
    N = 4
    z = torch.randn(N, 3)  # standard cond (B1 stripped)

    b1_space = ExplicitFiniteSampleSpace(
        torch.tensor([[0], [1]]),
        lambda s: s[:, 0].int(),
    )
    b2_space = ExplicitFiniteSampleSpace(
        torch.tensor([[0], [1], [2]]),
        lambda s: s[:, 0].int(),
    )
    joint_b2_b1_space = b2_space & b1_space

    # conditioning_kernel: p(B2 | B1, z) — IndexedFiniteKernel over B1
    # outer cond = [B1, z_rest], index_cond_indices=[0], standard_cond_dim=3
    cond_kernel = IndexedFiniteKernel(
        num_variable_outcomes=3,
        unindexed_cond=3,
        index_cond_indices=[0],
        index_sample_space=b1_space,
    )

    # sample_kernel: p(A | B2, B1, z) — IndexedFiniteKernel over A conditioned on (B2, B1)
    # full cond = [B2, B1, z_rest], index_cond_indices=[0, 1], standard_cond_dim=3
    sample_kernel = IndexedFiniteKernel(
        num_variable_outcomes=2,
        unindexed_cond=3,
        index_cond_indices=[0, 1],
        index_sample_space=joint_b2_b1_space,
    )

    ifck = IndexedFiniteConditionedKernel(sample_kernel, cond_kernel)
    ifck.eval()

    # construct_logit_table: shape (N, M_A*K_B2, K_B1), indexed over B1 only
    table = ifck.construct_logit_table(z)
    K_B1, K_B2, M_A = 2, 3, 2
    assert table.shape == (N, M_A * K_B2, K_B1), f"Expected ({N}, {M_A * K_B2}, {K_B1}), got {table.shape}"

    # For each B1 value, verify joint (A, B2) log-probs match direct computation.
    # Joint outcome row index in column b1: a_idx * K_B2 + b2_idx (M_A slowest, K_B2 fastest).
    for b1_idx in range(K_B1):
        cond = torch.cat([torch.full((z.shape[0], 1), b1_idx), z], dim=1)
        b2_log_probs = cond_kernel.construct_logits(cond).log_softmax(dim=-1)  # (N, K_B2)
        for b2_idx in range(K_B2):
            sample_cond = torch.cat([torch.full((z.shape[0], 1), b2_idx), cond], dim=1)
            a_log_probs_direct = sample_kernel.construct_logits(sample_cond).log_softmax(dim=-1)
            col = table.reshape(N, sample_kernel.sample_space.num_outcomes, b2_space.num_outcomes, b1_space.num_outcomes)[:, :, b2_idx, b1_idx]                       # (N, M_A*K_B2)
            assert (a_log_probs_direct - col.log_softmax(dim=-1)).abs().max().item() < 1e-5, f"log_softmax mismatch at (b1={b1_idx}, b2={b2_idx})"

    print("All checks passed.")
