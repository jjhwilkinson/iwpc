# `iwpc.learn_dist.kernels`

## Purpose

Trainable parametric **conditional distributions** `k(y | x)`. A "kernel"
in this package is a Lightning `nn.Module` that exposes two methods:

- `log_prob(samples, cond)` — differentiable log-density of `y` given `x`.
- `draw(cond)` (and the abstract `_draw`) — sample `y ~ k(. | x)` for each
  row of `x`. `draw_with_log_prob(cond)` is a fused variant that several
  kernels override for efficiency.

These are the trainable factors that the rest of `learn_dist` convolves
against a fixed `base_distributions` measure to model a data
distribution. They can be fit standalone via the maximum-likelihood
`training_step` on `TrainableKernelBase`, or in the unsupervised
"two-sample" setting via the trainer LightningModules in this package
(see `unlabelled_kernel_trainer.py`). The same kernels are reused by
`fdivergence_minimization/` and `classifier_reweighting`.

A small algebra of operators makes complex joint distributions concise:

- `a & b` — independent product (`ConcatenatedKernel`, sample axes
  concatenate, cond shared).
- `a + b` — independent product with cond also concatenated.
- `a | b` — `ConditionedKernel`, i.e. `p(a, b | z) = p(a | b, z) p(b | z)`.

## Layout

The package is large; group by family.

### Core contract

- `trainable_kernel_base.py` — `TrainableKernelBase` (ABC,
  `LightningModule`). Subclasses implement `log_prob` and `_draw`.
  Defines the `&`/`+`/`|` operators and a default Adam +
  `ReduceLROnPlateau(monitor="val_loss")` MLE training loop.

### Continuous-output leaf kernels

- `gaussian_kernel.py` — `GaussianKernel` (1D, trainable mean and scale
  networks, optional `max_chi` outlier mask).
- `multivariate_gaussian_kernel.py` — `MultivariateGaussianKernel` and
  the `MultivariateGaussianParameters` dataclass. Parameterises the
  covariance as `std . correlation . std` with an eigendecomposed
  unnormalised correlation, so the mean, log-std, log-eigvals, and
  log-rotation are each driven by their own subnetwork. Includes
  `initialise`/`initialise_cov` factories that seed the parameter
  networks from the empirical mean and covariance of a dataset.
- `two_sided_exponential_kernel.py` — `TwoSidedExponentialKernel` (1D
  Laplace-like, trainable location and scale).

### Structural combinators (continuous and finite)

- `concatenated_kernel.py` — `ConcatenatedKernel`. Independent product
  of sub-kernels (`&` and `+`); merges nested ConcatenatedKernels.
- `conditioned_kernel.py` — `ConditionedKernel`. Chains `p(y | x, z)`
  with `p(x | z)` to build `p(x, y | z)` (`|` operator).
- `mixture_kernel.py` — `MixtureKernel`. Weighted mixture with
  cond-dependent log-weights via a `LogSoftmaxEncoding` head.
- `branching_kernel.py` — `BranchingKernel` (and `FiniteBranchingKernel`).
  Routes each row to one of several sub-kernels based on a discrete
  feature in `cond`; `branched_evaluation` does the gather/scatter.
- `restructuring_kernel.py` — reorders / re-indexes the cond before
  delegating to a base kernel.
- `permutation_kernel.py` — reorders the **sample** axis of a base
  kernel.
- `add_cond_kernel.py` — `AddCondKernel`. Treats the base kernel as a
  residual: `y = cond + delta`, with an optional `custom_encoding` on
  the difference for circular features.
- `pass_through_kernel.py` — deterministic passthrough of the first N
  components of `cond` (zero log-prob).
- `dirac_kernel.py` — `DiracKernel`. Identity: `y = cond` (zero
  log-prob, no equality check).
- `cut_kernel.py` — `CutKernelInterface` (ABC), the contract for
  kernels that restrict a base kernel to a subset of its sample space
  while exposing the cut-pass log-probability.

### Finite-support kernels

A kernel over a discrete sample space — one whose outcomes are
enumerable — implements `FiniteKernelInterface` (overrides
`construct_log_probs(cond) -> (N, num_outcomes)` instead of
implementing `log_prob` / `_draw` directly).

- `finite_sample_space.py` — `FiniteSampleSpace` (ABC) plus
  `ConcatenatedFiniteSampleSpace`, `CutFiniteSampleSpace`,
  `ExplicitFiniteSampleSpace`. Provides the index <-> outcome bijection
  used everywhere.
- `finite_kernel_interface.py` — `FiniteKernelInterface` and
  `sample_idx_from_log_probs` (categorical inverse-CDF sampler).
  Overrides `&`/`+`/`|` to produce the `Finite…` variants when both
  operands are finite, plus a `.cut(fn)` shortcut.
- `finite_kernel.py` — `FiniteKernel` over a cartesian-product sample
  space, with one trainable `logit_model`.
- `finite_concatenated_kernel.py`, `finite_conditioned_kernel.py`,
  `finite_cut_kernel.py` — finite-aware versions of the corresponding
  combinators that compose log-probs directly rather than re-softmaxing.
- `fixed_finite_kernel.py` — non-trainable finite kernel with fixed
  probabilities.
- `constant_kernel.py` — finite kernel that always returns one value.

### Indexed finite kernels (fast `p(A | B = b, x)`)

- `indexed_interface.py` — `IndexedInterface`. When `B` is itself a
  finite variable read off `cond`, the kernel can emit the full
  `(N, M, K)` log-prob table for every `b` value from a single forward
  pass.
- `indexed_finite_kernel.py` — `IndexedFiniteKernel` (a `FiniteKernel`
  whose logit model fans out all `K` columns at once).
- `indexed_finite_conditioned_kernel.py` —
  `IndexedFiniteConditionedKernel`. Auto-selected by
  `IndexedInterface.__or__` when both children are indexed and
  compatible.

### Trainers

LightningModules that train kernels in the **unlabelled** two-sample
regime: only samples from `p` (real data) and from the base
distribution feeding `q` are required. A co-trained `log_p_over_q_model`
learns the log density ratio via a cross-entropy loss; the kernel's
loss then consumes that (detached) `log(p/q)` estimate.

- `unlabelled_kernel_trainer.py` — `UnLabelledKernelTrainer`,
  `KernelKLDivergenceGradientLoss`, and `KernelLRAdjustor` (the
  fluctuation-based LR scheduler). The standard choice.
- `partially_exact_unlabelled_kernel_trainer.py` —
  `PartiallyExactUnLabelledKernelTrainer`. Splits the model into a
  finite `exact_kernel` (whose outcomes are enumerated exactly) and a
  sampled remainder; reduces variance when one factor is discrete.
- `unlabelled_multi_kernel_trainer.py` — `UnlabelledMultiKernelTrainer`.
  Trains a `ConcatenatedKernel` alongside one `log_p_over_q_model`
  per independent factor.

## Usage

### A simple 1D Gaussian kernel

```python
import torch
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel

k = GaussianKernel(cond=2)                              # cond x in R^2
cond = torch.randn(128, 2)
y, logp = k.draw_with_log_prob(cond)                    # y: (128, 1)
```

### Build a joint `p(x, y | z)` by composition

```python
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.kernels.multivariate_gaussian_kernel import MultivariateGaussianKernel

x_given_z   = GaussianKernel(cond=1)                    # p(x | z)
y_given_xz  = MultivariateGaussianKernel(cond=2, sample_dim=2)  # p(y | x, z)
joint = y_given_xz | x_given_z                          # p(x, y | z), 3D sample
```

### Train against unlabelled data with a co-trained `log_p_over_q_model`

```python
import lightning as L
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import UnLabelledKernelTrainer

trainer_module = UnLabelledKernelTrainer(
    kernel=k,
    log_p_over_q_model=ratio_estimator,
    min_train_divergence=1.0,
    kernel_lr=1e-4,
)
L.Trainer(max_epochs=200).fit(trainer_module, datamodule=dm)
```

Use `PartiallyExactUnLabelledKernelTrainer` when part of the model is a
finite-support kernel whose outcomes can be enumerated, and
`UnlabelledMultiKernelTrainer` when several independent factors of a
`ConcatenatedKernel` should each have their own `log_p_over_q_model`.
