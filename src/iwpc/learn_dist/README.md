# `iwpc.learn_dist`

The distribution-learning side of `iwpc`. Where the main package estimates lower
bounds on f-divergences between two empirical distributions, `learn_dist` reuses
the same machinery to **learn a distribution** (or a conditional distribution
`k(y|x)`) from samples — either by classifier reweighting, by training a kernel
to minimise an f-divergence against a target, or by training a kernel against
unlabelled data via a discriminator.

It is **independent** of `calculate_divergence` and `run_reweight_loop`, but
shares:

- `iwpc.divergences.DifferentiableFDivergence` — both the surrogate kernel loss
  and the discriminator's BCE loss in `fdivergence_minimization` are derived
  from a chosen `DifferentiableFDivergence`.
- The PyTorch Lightning trainer scaffolding — every trainer in `learn_dist` is
  a `LightningModule`, so existing `Trainer`/`ModelCheckpoint`/`EarlyStopping`
  patterns transfer over.
- `iwpc.models.utils.basic_model_factory` and `iwpc.encodings.Encoding` for the
  underlying neural networks and feature transforms.

It does **not** use the `(features, labels, weights)` `labels==0/1` batch
convention; trainers here define their own data flow (samples + conditioning
information, or weighted samples from a target distribution).

## Layout

### `classifier_reweighting.py` — `DistributionApproximator`

A `LightningModule` that learns an unconditional density `p(x)` from samples by
training a binary classifier between the data samples and samples drawn from a
known `SamplableBaseModel`. The classifier output is interpreted as
`log p(x)/q(x)`; combined with the base distribution's analytic
`log_prob`, this yields a tractable `learned_log_prob`, plus a `draw` method
that returns weighted samples from the learned distribution. Optimiser is Adam
with `ReduceLROnPlateau` on `val_loss`.

### `base_distributions/`

Sampleable, analytic base distributions used as the "q" side of classifier
reweighting and as the input distribution that kernels convolve against. All
inherit from `SamplableBaseModel` (`draw`, `log_prob`, `dimension`,
`total_volume`). Includes uniform, Cauchy, exponential, multivariate normal,
and a histogram-based distribution. See `base_distributions/README.md`.

### `kernels/`

Trainable conditional distributions `k(y|x)` — the core abstraction of
`learn_dist`. `TrainableKernelBase` defines the `log_prob(samples, cond)` /
`_draw(cond)` interface. Concrete kernels include Gaussian, multivariate
Gaussian, two-sided exponential, mixture, finite (discrete sample space),
indexed-finite, plus structural kernels for composing simpler kernels
(`ConcatenatedKernel`, `ConditionedKernel`, `BranchingKernel`,
`PermutationKernel`, `RestructuringKernel`, `CutKernel`, …). Also contains
the unlabelled trainers (`UnlabelledKernelTrainer`,
`UnlabelledMultiKernelTrainer`,
`PartiallyExactUnlabelledKernelTrainer`) that fit a kernel to unlabelled data
by alternating a kernel optimiser with a discriminator. See
`kernels/README.md`.

### `fdivergence_minimization/`

`FDivergenceMinimizingKernelTrainer` — Lightning trainer that optimises a
`TrainableKernelBase` to minimise a chosen `DifferentiableFDivergence` between
a target distribution `p` (provided through samples) and `q` (the kernel
convolved against an input/base sample). A discriminator network estimates
`log p/q` and is trained as a BCE classifier; the kernel is trained against a
surrogate gradient loss derived from the divergence's `f_dash_given_log`. An
optional `FiniteKernelInterface` lets a discrete component of `q` be summed
exactly rather than sampled. See `fdivergence_minimization/README.md`.

## Usage sketches

### Learn an unconditional density via classifier reweighting

```python
from lightning import Trainer
from iwpc.learn_dist.classifier_reweighting import DistributionApproximator
from iwpc.learn_dist.base_distributions.multivaraite_normal_base_model import (
    MultivariateNormalBaseModel,
)

base = MultivariateNormalBaseModel(mean=..., cov=...)
model = DistributionApproximator(base_distribution=base)
Trainer(max_epochs=200).fit(model, datamodule=my_data_module)

log_p = model.learned_log_prob(x)
samples, log_w = model.draw(10_000)
```

### Fit a conditional kernel by f-divergence minimisation

```python
from lightning import Trainer
from iwpc.divergences import KLDivergence
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.fdivergence_minimization.fdivergence_minimizing_kernel_trainer \
    import FDivergenceMinimizingKernelTrainer
from iwpc.models.utils import basic_model_factory

kernel = GaussianKernel(cond=C)              # 1D sample, C-D conditioning
disc = basic_model_factory(input=1, output=1)
trainer_mod = FDivergenceMinimizingKernelTrainer(
    sampled_kernel=kernel,
    log_p_over_q_model=disc,
    divergence=KLDivergence(),
)
Trainer(max_epochs=500).fit(trainer_mod, datamodule=my_paired_data_module)
```

See the sub-package READMEs for details on individual kernels, base
distributions, and trainer options.
