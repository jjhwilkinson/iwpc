from typing import Callable, Iterable

import torch
from torch import Tensor
from torch.nn import ModuleList

from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


def map_indexing(
    tensor_or_tensors: Tensor | list[Tensor],
    slices_tensor: Tensor
) -> Tensor | tuple[Tensor, ...]:
    """
    Given a pytorch-indexing  compatible slice object, slices all tensors in tensor_or_tensors with the given slice
    along the first dimension

    Parameters
    ----------
    tensor_or_tensors
        A tensor or list of tensors
    slices_tensor
        A pytorch-indexing compatible slice

    Returns
    -------
    Tensor | tuple[Tensor, ...]
        The tensor, or list of tensors sliced along the first dimension
    """
    if isinstance(tensor_or_tensors, Tensor):
        return tensor_or_tensors[slices_tensor]
    return tuple(map_indexing(t, slices_tensor) for t in tensor_or_tensors)


def branched_evaluation(
    function_indices: Tensor,
    functions: list[Callable[[Tensor, ...], Tensor | Iterable[Tensor]]],
    *inputs: Tensor,
) -> tuple[Tensor, ...]:
    """
    Given a sequence of functions that operate on the same input and output space, and an integer tensor of indices
    within said sequence, the output tensor satisfies output[i] = functions[function_indices[i]](*inputs[i]). The
    functions may however return an interable of tensors in which case output[k][i] = functions[function_indices[i]](*inputs[i])[k]

    Parameters
    ----------
    function_indices
        An integer tensor indicating which function should operate on the i'th input row
    functions
        A list of functions that operate on the same input and output space. Functions may return a single tensor or an
        iterable of tensors
    inputs
        A sequence of input tensors

    Returns
    -------
    tuple[Tensor, ...]
        See description above
    """
    outputs_by_label = []
    output_indices = torch.zeros(inputs[0].shape[0], dtype=torch.int, device=inputs[0].device)

    cum_cnt = 0
    for idx in range(len(functions)):
        mask = function_indices == idx
        branch_size = mask.sum()
        outputs_by_label.append(functions[idx](*map_indexing(inputs, mask)))
        output_indices[mask] = torch.arange(cum_cnt, cum_cnt + branch_size, dtype=torch.int, device=output_indices.device)
        cum_cnt += branch_size

    if isinstance(outputs_by_label[0], Tensor):
        outputs_by_label = [[output] for output in outputs_by_label]
    outputs_by_label = [torch.concat(ts, dim=0) for ts in zip(*outputs_by_label)]

    return map_indexing(outputs_by_label, output_indices)


class BranchingKernel(TrainableKernelBase):
    """
    Kernel that samples one of any number of sub-kernels based on the value of the conditioning information. The
    components of the conditioning information specified by branch_sample_indices define the branching sample which is
    converted to an integer index using outcome_to_idx_fn. This integer is in turn used to pick out the entry in
    sub_kernels to sample
    """
    def __init__(
        self,
        sub_kernels: Iterable[TrainableKernelBase],
        branch_sample_indices: list[int] | int,
        outcome_to_idx_fn: Callable[[Tensor], Tensor],
    ):
        """
        Parameters
        ----------
        sub_kernels
            An iterable of TrainableKernelBase sub-kernels
        branch_sample_indices
            The indices of the conditioning information that making up the 'branching sample'
        outcome_to_idx_fn
            Function that accepts a tensor of shape (N, len(branch_sample_indices)) and returns an integer tensor of
            shape (N,)
        """
        sub_kernels = list(sub_kernels)
        assert all(k.sample_dimension == sub_kernels[0].sample_dimension for k in sub_kernels)
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        super().__init__(
            sub_kernels[0].sample_dimension,
            len(branch_sample_indices) + sub_kernels[0].cond_dimension,
        )
        
        self.branching_indices = [branch_sample_indices,] if isinstance(branch_sample_indices, int) else branch_sample_indices
        self.sub_kernel_conditioning_indices = [i for i in range(self.cond_dimension) if i not in self.branching_indices]
        self.sub_kernels = ModuleList(sub_kernels)
        self.outcome_to_idx_fn = outcome_to_idx_fn

    @classmethod
    def condition_on(
        cls,
        sub_kernels: Iterable[TrainableKernelBase],
        finite_kernel: "FiniteKernelInterface",
    ) -> "BranchingKernel":
        """
        Utility method to construct a BranchingKernel that branches on the output of a finite kernel

        Parameters
        ----------
        sub_kernels
            An iterable of TrainableKernelBase sub-kernels
        finite_kernel
            A finite kernel

        Returns
        -------
        BranchingKernel
            A BranchingKernel that branches on the output of a finite kernel
        """
        sub_kernels = list(sub_kernels)
        assert len(sub_kernels) == finite_kernel.num_outcomes
        return cls(
            sub_kernels,
            list(range(finite_kernel.sample_dimension)),
            finite_kernel.outcome_to_idx
        )

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Calculates the log probability of the given samples for the given conditioning information using
        p(A and B) = p(A|B) p(B)

        Parameters
        ----------
        samples
            A tensor of size (N, self.sample_dimension)
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            A tensor of shape (N,)
        """
        branching_cond = cond[:, self.branching_indices]
        branching_idxs = self.outcome_to_idx_fn(branching_cond)
        sub_kernel_cond = cond[:, self.sub_kernel_conditioning_indices]

        return branched_evaluation(
            branching_idxs,
            [k.log_prob for k in self.sub_kernels],
            samples,
            sub_kernel_cond,
        )[0]

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of size (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of shape (N, base_kernels[0].sample_dimension)
        """
        branching_cond = cond[:, self.branching_indices]
        branching_idxs = self.outcome_to_idx_fn(branching_cond)

        return branched_evaluation(
            branching_idxs,
            [k.draw for k in self.sub_kernels],
            cond[:, self.sub_kernel_conditioning_indices],
        )[0]

    def draw_with_log_prob(self, cond: Tensor) -> tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            A tensor of size (N, self.cond_dimension)

        Returns
        -------
        Tensor
            A tensor of shape (N, base_kernels[0].sample_dimension) and corresponding probability of each sample in a
            tensor of shape (N,)
        """
        branching_cond = cond[:, self.branching_indices]
        branching_idxs = self.outcome_to_idx_fn(branching_cond)
        sub_kernel_cond = cond[:, self.sub_kernel_conditioning_indices]

        return branched_evaluation(
            branching_idxs,
            [k.draw_with_log_prob for k in self.sub_kernels],
            sub_kernel_cond,
        )


class FiniteBranchingKernel(FiniteKernelInterface, BranchingKernel):
    """
    BranchingKernel implementation that also satisfies the FiniteKernelInterface
    """
    def __init__(
        self,
        sub_kernels: Iterable[TrainableKernelBase],
        branch_sample_indices: list[int] | int,
        outcome_to_idx_fn: Callable[[Tensor], Tensor],
    ):
        """
        Parameters
        ----------
        sub_kernels
            An iterable of FiniteKernelInterface sub-kernels
        branch_sample_indices
            The indices of the conditioning information that make up the 'branching sample'
        outcome_to_idx_fn
            Function that accepts a tensor of shape (N, len(branch_sample_indices)) and returns an integer tensor of
            shape (N,)
        """
        assert all(s.sample_space == sub_kernels[0].sample_space for s in sub_kernels)
        super().__init__(
            sub_kernels[0].sample_space,
            sub_kernels,
            branch_sample_indices,
            outcome_to_idx_fn
        )

    def construct_logits(self, cond: Tensor) -> Tensor:
        """
        Constructs the logits over the possible outcomes given the conditioning information

        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            A tensor of shape (N, self.num_outcomes)
        """
        branching_cond = cond[:, self.branching_indices]
        branching_idxs = self.outcome_to_idx_fn(branching_cond)
        sub_kernel_cond = cond[:, self.sub_kernel_conditioning_indices]

        return branched_evaluation(
            branching_idxs,
            [k.construct_logits for k in self.sub_kernels],
            sub_kernel_cond,
        )[0]
