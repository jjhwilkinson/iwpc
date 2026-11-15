"""
Tests for the exact-kernel conditioning and sample-width guards of
`FDivergenceMinimizingKernelTrainer`.
"""
import pytest
import torch
from torch import Tensor

from iwpc.divergences import KLDivergence
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer import (
    FDivergenceMinimizingKernelTrainer,
)
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory

BASE_DIMENSION = 2


def build_exact_kernel() -> FiniteKernel:
    """
    Returns
    -------
    FiniteKernel
        A binary finite kernel conditioned on the base samples alone, so that `sample_dimension == 1` and
        `cond_dimension == BASE_DIMENSION`
    """
    return FiniteKernel(2, cond=BASE_DIMENSION)


def build_sampled_kernel(cond_dimension: int) -> TrainableKernelBase:
    """
    Builds a two-dimensional sampled kernel over a conditioning space of the requested width. A pair of Gaussian
    kernels merged with `&` is used because the resulting `ConcatenatedKernel` slices its conditioning information
    by the widths it was built for, which is what lets a too-narrow kernel silently consume a widened conditioning
    tensor instead of failing

    Parameters
    ----------
    cond_dimension
        The width of the conditioning space the kernel is built for

    Returns
    -------
    TrainableKernelBase
        A kernel with `sample_dimension == 2` and the requested `cond_dimension`
    """
    return GaussianKernel(cond=cond_dimension) & GaussianKernel(cond=cond_dimension)


def build_trainer(
    sampled_kernel: TrainableKernelBase,
    exact_kernel: FiniteKernel | None = None,
) -> FDivergenceMinimizingKernelTrainer:
    """
    Parameters
    ----------
    sampled_kernel
        The sampled kernel to hand to the trainer
    exact_kernel
        The optional exact kernel to hand to the trainer

    Returns
    -------
    FDivergenceMinimizingKernelTrainer
        A trainer over the given kernels, with a throwaway log-density-ratio model
    """
    return FDivergenceMinimizingKernelTrainer(
        sampled_kernel=sampled_kernel,
        log_p_over_q_model=basic_model_factory(input=3, output=1),
        divergence=KLDivergence(),
        exact_kernel=exact_kernel,
    )


def build_q_init_samples(num_rows: int, sample_dimension: int) -> Tensor:
    """
    Parameters
    ----------
    num_rows
        The number of rows to build
    sample_dimension
        The width of the sample vectors

    Returns
    -------
    Tensor
        A tensor of zeros of shape (num_rows, sample_dimension)
    """
    return torch.zeros(num_rows, sample_dimension)


def test_exact_kernel_with_too_narrow_sampled_kernel_raises() -> None:
    """
    A sampled kernel conditioned on the base samples alone does not account for the prepended discrete outcome and
    must be rejected at construction, naming both conditioning dimensions
    """
    exact_kernel = build_exact_kernel()

    with pytest.raises(ValueError) as excinfo:
        build_trainer(build_sampled_kernel(BASE_DIMENSION), exact_kernel)

    message = str(excinfo.value)
    assert "sampled_kernel.cond_dimension" in message
    assert "exact_kernel.sample_dimension" in message
    assert f"{exact_kernel.cond_dimension} + {exact_kernel.sample_dimension}" in message
    assert str(exact_kernel.cond_dimension + exact_kernel.sample_dimension) in message
    assert f"is {BASE_DIMENSION}" in message


def test_exact_kernel_with_too_wide_sampled_kernel_raises() -> None:
    """
    Over-wide conditioning spaces are rejected for the same reason as too-narrow ones: the trainer prepends exactly
    `exact_kernel.sample_dimension` columns
    """
    exact_kernel = build_exact_kernel()

    with pytest.raises(ValueError, match="sampled_kernel.cond_dimension"):
        build_trainer(build_sampled_kernel(BASE_DIMENSION + exact_kernel.sample_dimension + 1), exact_kernel)


def test_matching_dimensions_construct_and_draw() -> None:
    """
    A sampled kernel built with the exact kernel's `sample_dimension` extra leading conditioning columns constructs
    without complaint, and yields one full q sample per discrete outcome
    """
    torch.manual_seed(0)
    exact_kernel = build_exact_kernel()
    sampled_kernel = build_sampled_kernel(BASE_DIMENSION + exact_kernel.sample_dimension)
    trainer = build_trainer(sampled_kernel, exact_kernel)

    num_rows = 8
    full_sample_dimension = sampled_kernel.sample_dimension + exact_kernel.sample_dimension
    full_sample_iter, cut_pass_log_prob = trainer.full_sample_iter_and_cut_pass_log_prob(
        torch.randn(num_rows, BASE_DIMENSION),
        build_q_init_samples(num_rows, full_sample_dimension),
    )

    assert cut_pass_log_prob.shape == (num_rows,)
    yielded_outcomes = []
    for q, sample_log_prob, exact_outcome, exact_log_prob in full_sample_iter:
        assert q.shape == (num_rows, full_sample_dimension)
        assert sample_log_prob.shape == (num_rows,)
        assert exact_log_prob.shape == (num_rows,)
        assert exact_outcome.shape == (exact_kernel.sample_dimension,)
        yielded_outcomes.append(exact_outcome)

    assert len(yielded_outcomes) == exact_kernel.sample_space.num_outcomes


def test_wrong_q_init_sample_width_raises() -> None:
    """
    The initial q samples must be as wide as the sampled kernel's draw plus the exact kernel's outcome. A mismatch
    used to surface as a bare broadcast error from the addition
    """
    torch.manual_seed(0)
    exact_kernel = build_exact_kernel()
    sampled_kernel = build_sampled_kernel(BASE_DIMENSION + exact_kernel.sample_dimension)
    trainer = build_trainer(sampled_kernel, exact_kernel)

    num_rows = 8
    with pytest.raises(ValueError, match="exact_kernel.sample_dimension"):
        trainer.full_sample_iter_and_cut_pass_log_prob(
            torch.randn(num_rows, BASE_DIMENSION),
            build_q_init_samples(num_rows, sampled_kernel.sample_dimension),
        )


def test_no_exact_kernel_is_unconstrained() -> None:
    """
    Without an exact kernel there is nothing to prepend, so a kernel conditioned on the base samples alone is
    correct and must still construct and draw
    """
    torch.manual_seed(0)
    sampled_kernel = build_sampled_kernel(BASE_DIMENSION)
    trainer = build_trainer(sampled_kernel)

    num_rows = 8
    full_sample_iter, cut_pass_log_prob = trainer.full_sample_iter_and_cut_pass_log_prob(
        torch.randn(num_rows, BASE_DIMENSION),
        build_q_init_samples(num_rows, sampled_kernel.sample_dimension),
    )

    assert cut_pass_log_prob.shape == (num_rows,)
    (q, sample_log_prob, exact_outcome, exact_log_prob), = list(full_sample_iter)
    assert q.shape == (num_rows, sampled_kernel.sample_dimension)
    assert sample_log_prob.shape == (num_rows,)
    assert exact_outcome.shape == (0,)
    assert exact_log_prob.shape == (num_rows,)


def test_finite_cut_exact_kernel_satisfies_the_invariant() -> None:
    """
    A `FiniteCutKernel` inherits its base kernel's sample and conditioning dimensions, so the same invariant applies
    unchanged: a sampled kernel built with the extra leading columns is accepted and the cut-pass log-probability is
    non-trivial
    """
    torch.manual_seed(0)
    exact_kernel = build_exact_kernel().cut(lambda outcome: bool(outcome[0] == 1))
    sampled_kernel = build_sampled_kernel(BASE_DIMENSION + exact_kernel.sample_dimension)
    trainer = build_trainer(sampled_kernel, exact_kernel)

    num_rows = 8
    full_sample_dimension = sampled_kernel.sample_dimension + exact_kernel.sample_dimension
    full_sample_iter, cut_pass_log_prob = trainer.full_sample_iter_and_cut_pass_log_prob(
        torch.randn(num_rows, BASE_DIMENSION),
        build_q_init_samples(num_rows, full_sample_dimension),
    )

    assert cut_pass_log_prob.shape == (num_rows,)
    assert torch.all(cut_pass_log_prob < 0)
    yielded_sample_shapes = [q.shape for q, _, _, _ in full_sample_iter]
    assert yielded_sample_shapes == [(num_rows, full_sample_dimension)]
