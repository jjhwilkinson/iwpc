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
    Utility Kernel that allows the user to re-order the components of the base kernel
    """
    def __init__(self, base_kernel: TrainableKernelBase, permutation: list[int]):
        """
        Parameters
        ----------
        base_kernel
            The base kernel
        permutation
            A list of the integers from 0 to len(permutation)-1 such that permutation[i] is the index in the sample of
            the base kernel that should be placed at index i in the permuted output
        """
        assert len(permutation) == base_kernel.sample_dimension
        assert set(permutation) == set(range(base_kernel.sample_dimension))
        super().__init__(
            base_kernel.sample_dimension,
            base_kernel.cond_dimension,
        )
        self.base_kernel = base_kernel
        self.permutation = permutation
        self.inverse_permutation = invert_permutation(permutation)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
            A tensor of samples of shape (N, self.sample_dimension)
        cond
            A tensor of conditions of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of log probabilities of shape (N,) calculated by evaluate the log_prob of the base kernel on the
            inverse-permuted samples
        """
        return self.base_kernel.log_prob(samples[:, self.inverse_permutation], cond)

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditions of shape (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of shape (N, self.sample_dimension) containing samples from the base kernel with components
            permutated by the specified permutation
        """
        return self.base_kernel.draw(cond)[:, self.permutation]

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            A tensor of conditions of shape (N, self.cond_dimension)

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tensor of samples and their corresponding log probabilities drawn using the base kernel and then permuted
        """
        samples, log_probs = self.base_kernel.draw_with_log_prob(cond)
        return samples[:, self.permutation], log_probs
