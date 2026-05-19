from typing import Tuple

from torch import Tensor

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


def invert_permutation(permutation: list[int]) -> list[int]:
    """
    Inverts a permutation

    Parameters
    ----------
    permutation
        A list of the integers from 0 to len(permutation)-1

    Returns
    -------
        A list of the integers from 0 to len(permutation)-1 that inverts the given permutation
    """
    reverse_permutation = [0] * len(permutation)
    for idx, oidx in enumerate(permutation):
        reverse_permutation[oidx] = idx
    return reverse_permutation


class PermutationKernel(TrainableKernelBase):
    """
    Utility Kernel that allows the user to re-order the sample and/or conditioning components of the base kernel
    """
    def __init__(
        self,
        base_kernel: TrainableKernelBase,
        sample_permutation: list[int] | None = None,
        cond_permutation: list[int] | None = None,
    ):
        """
        Parameters
        ----------
        base_kernel
            The base kernel
        sample_permutation
            A list of the integers from 0 to sample_dimension-1 such that sample_permutation[i] is the index in the
            base kernel's output that should be placed at index i in the permuted output. None means no permutation.
        cond_permutation
            A list of the integers from 0 to cond_dimension-1 such that cond_permutation[i] is the column index in
            the external cond that the base kernel expects at position i. None means no permutation.
        """
        if sample_permutation is not None:
            assert len(sample_permutation) == base_kernel.sample_dimension
            assert set(sample_permutation) == set(range(base_kernel.sample_dimension))
        if cond_permutation is not None:
            assert len(cond_permutation) == base_kernel.cond_dimension
            assert set(cond_permutation) == set(range(base_kernel.cond_dimension))
        super().__init__(
            base_kernel.sample_dimension,
            base_kernel.cond_dimension,
        )
        self.base_kernel = base_kernel
        self.sample_permutation = sample_permutation
        self.inverse_sample_permutation = invert_permutation(sample_permutation) if sample_permutation is not None else None
        self.cond_permutation = cond_permutation

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        return self.base_kernel.log_prob(
            samples[:, self.inverse_sample_permutation] if self.inverse_sample_permutation is not None else samples,
            cond[:, self.cond_permutation] if self.cond_permutation is not None else cond,
        )

    def _draw(self, cond: Tensor) -> Tensor:
        out = self.base_kernel.draw(cond[:, self.cond_permutation] if self.cond_permutation is not None else cond)
        return out[:, self.sample_permutation] if self.sample_permutation is not None else out

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        samples, log_probs = self.base_kernel.draw_with_log_prob(
            cond[:, self.cond_permutation] if self.cond_permutation is not None else cond
        )
        return (samples[:, self.sample_permutation] if self.sample_permutation is not None else samples), log_probs
