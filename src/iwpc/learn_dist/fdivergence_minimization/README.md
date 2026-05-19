# `iwpc.learn_dist.fdivergence_minimization`

## Purpose

Train a kernel (any `TrainableKernelBase` from `learn_dist/kernels/`) so that the
distribution `q` it induces minimises a chosen `DifferentiableFDivergence`
`Df(p || q)` against a target distribution `p` represented by samples. This is
the companion of the f-divergence-*estimation* flow in the main package: there
the divergence is the quantity of interest, here it is the loss that drives the
generative model.

The trainer follows an adversarial schedule. A discriminator
`log_p_over_q_model` is trained as a binary classifier between `p` and `q`
samples (label `0 = p`, `1 = q`) and produces a detached estimate of
`log(p/q)`. The kernel is then updated against a score-function surrogate whose
gradient matches the gradient of `Df(p || q)` with respect to the kernel
parameters.

## Layout

- `fdivergence_minimizing_kernel_trainer.py` -
  `FDivergenceMinimizingKernelTrainer`, a `LightningModule` with manual
  optimisation that alternates a discriminator step (BCE) and a kernel step
  (gradient surrogate). Logs `train_divergence` / `val_divergence`
  (`1 - BCE / log 2`, a JS-style lower bound) and `train_kernel_loss`.
- `fdivergence_gradient_surrogate_loss.py` -
  `FDivergenceGradientSurrogateLoss`, the standalone per-sample surrogate
  `w * f'(p/q) * log q(x|base)`, evaluated through
  `DifferentiableFDivergence._f_dash_given_log(log_p_over_q)` for numerical
  stability. (The trainer currently inlines the same expression in
  `calculate_kernel_loss`; the class documents the identity.)

## Usage

Batches must be 4-tuples `(base_samples, samples, labels, weights)` where
`labels == 0` marks `p` samples and `labels == 1` marks `q` rows whose
`base_samples` are drawn from the kernel's base distribution.

```python
import lightning as L
from iwpc.divergences import KLDivergence
from iwpc.learn_dist.kernels.gaussian_kernel import GaussianKernel
from iwpc.learn_dist.fdivergence_minimization import (
    FDivergenceMinimizingKernelTrainer,
)
from iwpc.models.utils import basic_model_factory

kernel = GaussianKernel(cond=2)                             # trainable q (1D sample, 2D conditioning)
discriminator = basic_model_factory(input=2, output=1)      # log(p/q) net

module = FDivergenceMinimizingKernelTrainer(
    sampled_kernel=kernel,
    log_p_over_q_model=discriminator,
    divergence=KLDivergence(),
    discriminator_opt_lr=1e-3,
    kernel_opt_lr=1e-4,
    start_kernel_train_epoch=1,    # warm up the discriminator first
    kernel_resample_rate=4,        # variance reduction on the kernel step
)

trainer = L.Trainer(max_epochs=50)
trainer.fit(module, datamodule=my_data_module)
samples = module.sampled_kernel.draw(base_samples)
```

If part of `q` is discrete and enumerable, pass it as `exact_kernel` (a
`FiniteKernelInterface`); expectations over its outcomes are summed exactly
instead of sampled and the surrogate is reweighted by `exp(log_prob)`:

```python
module = FDivergenceMinimizingKernelTrainer(
    sampled_kernel=continuous_part,
    exact_kernel=discrete_part,
    log_p_over_q_model=discriminator,
    divergence=JensenShannonDivergence(),
    zero_out_init_q_samples=True,  # treat kernel draw as the full q sample
)
```

Use `accumulate_kernel_batches > 0` to accumulate kernel gradients across
several batches before stepping (the discriminator still steps every batch).
