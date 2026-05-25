# AGENTS — `iwpc.learn_dist`

Distribution-learning sub-package. Independent of `calculate_divergence` and
`run_reweight_loop`; shares `iwpc.divergences.DifferentiableFDivergence`,
`iwpc.models.utils.basic_model_factory`, `iwpc.encodings.Encoding`, and the
PyTorch Lightning trainer pattern. Does NOT use the
`(features, labels, weights)` / `0=p, 1=q` batch convention from the main
divergence flow — trainers here define their own batch layout.

## How the pieces compose

- `base_distributions/` provides analytic, sampleable distributions
  (`SamplableBaseModel`: `draw`, `log_prob`, `dimension`). They serve as the
  "q" in classifier reweighting and as the input/noise distribution that a
  kernel convolves against in kernel training.
- `kernels/` provides trainable conditional distributions
  (`TrainableKernelBase`: `log_prob(samples, cond)`, `_draw(cond)`).
  Structural kernels (`Concatenated`, `Branching`, `Conditioned`,
  `Restructuring`, `Cut`, `Mixture`, `Permutation`, `AddCond`, …) compose
  simpler kernels into richer conditional models. Finite variants
  (`FiniteKernelInterface`, `IndexedFiniteKernel`, …) model discrete sample
  spaces and allow exact enumeration of outcomes.
- `kernels/` also contains the **unlabelled trainers**
  (`UnlabelledKernelTrainer`, `UnlabelledMultiKernelTrainer`,
  `PartiallyExactUnlabelledKernelTrainer`) — Lightning modules that train a
  kernel against unlabelled samples by alternating a kernel update with a
  step of a co-trained `log_p_over_q_model` that learns the log density
  ratio via a cross-entropy loss. The kernel's loss consumes the detached
  `log(p/q)` estimate. A custom `KernelLRAdjustor` LR scheduler drives the
  kernel learning rate.
- `fdivergence_minimization/FDivergenceMinimizingKernelTrainer` is the
  divergence-aware counterpart: given paired `(p_samples, q_input_samples)` it
  trains a kernel to minimise a chosen `DifferentiableFDivergence` between `p`
  and the kernel-convolved `q`, alongside a co-trained `log_p_over_q_model`
  that estimates the density ratio. The kernel gradient comes from
  `fdivergence_gradient_surrogate_loss` (uses
  `DifferentiableFDivergence.f_dash_given_log` for numerical stability) and
  matches the gradient of `Df(p || q)` w.r.t. the kernel parameters.

## `classifier_reweighting.py` — `DistributionApproximator`

```python
DistributionApproximator(
    base_distribution: SamplableBaseModel,
    base_distribution_sample_rate: int = 1,
    log_p_over_q_model: Optional[Module] = None,
)
```

`LightningModule` that learns an *unconditional* `p(x)` from a data module
emitting `(samples, _, weights)` batches. Trains
`log_p_over_q_model(x) ≈ log p(x)/q(x)` via BCE against base-distribution
samples; exposes `learned_log_prob(x)` and `draw(n) -> (samples, log_weights)`.
Adam + `ReduceLROnPlateau(monitor="val_loss")`. This is the simplest entry
point into `learn_dist` and is not conditional — for conditional density
estimation, build a kernel and use one of the kernel trainers above.

## Typical plumbing

1. Pick (or compose) a `TrainableKernelBase` describing the conditional model.
2. Pick a `SamplableBaseModel` (or other input source) that the kernel
   transforms.
3. Pick a trainer: `FDivergenceMinimizingKernelTrainer` when you have paired
   `p`/`q-input` samples and want explicit divergence control; an unlabelled
   kernel trainer when you have only unlabelled data; `DistributionApproximator`
   for the unconditional, no-kernel case.
4. Fit with a standard Lightning `Trainer` — no `calculate_divergence` wrapper
   is involved.
