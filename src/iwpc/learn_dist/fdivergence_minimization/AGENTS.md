# Agent notes: `fdivergence_minimization`

## `FDivergenceMinimizingKernelTrainer.__init__`

```
sampled_kernel: TrainableKernelBase,
log_p_over_q_model: Module,
divergence: DifferentiableFDivergence,
exact_kernel: FiniteKernelInterface | None = None,
discriminator_opt_lr: float = 1e-3,
kernel_opt_lr: float = 1e-4,
start_kernel_train_epoch: int = 1,
start_discriminator_train_epoch: int = 0,
kernel_resample_rate: int = 1,
zero_out_init_q_samples: bool = False,
accumulate_kernel_batches: int = -1,
```

`automatic_optimization = False`; `configure_optimizers` returns two Adam
optimisers (discriminator, then kernel + optional `exact_kernel`). Batch
contract: `(base_samples, samples, labels, weights)`, label `0 = p`, `1 = q`.
`zero_out_init_q_samples` overrides the `q_init_samples = samples[q_mask]`
residual term (kernel output becomes the full sample). The "divergence"
logged (`val_divergence = 1 - BCE / log 2`) is the JS lower bound from the
discriminator, regardless of `self.divergence`.

## Surrogate gradient identity

Score-function / REINFORCE identity for `Df(p || q) = E_q[ f(p/q) ]`:

```
d/dθ Df(p || q) = E_q[ f'(p/q) * d/dθ log q_θ(x) ]
```

so the trainer minimises the surrogate

```
L(θ) = E_q[ w * f'(p/q) * (log P(exact_outcome) + log q_θ(x | base, outcome)) ]
```

Implemented in `calculate_kernel_loss` as
`q_weights * exp(exact_outcome_log_prob).detach() * divergence.f_dash_given_log(-log_p_over_q) * (exact_outcome_log_prob + log_prob)`,
where `log_prob` comes from `sampled_kernel.draw_with_log_prob` (the only
tensor with grad through θ), `log_p_over_q` is detached from the
discriminator, and `f_dash_given_log` evaluates `f'(p/q)` stably from
`log(q/p) = -log_p_over_q`. The discriminator side uses standard BCE
(`logsigmoid` on `+log_p_over_q` for p, on `-log_p_over_q` for q),
sample-weighted by `exp(exact_outcome_log_prob)` per discrete branch.

## Cross-package deps

- `iwpc.divergences.DifferentiableFDivergence` - supplies `f_dash_given_log`
  (public) / `_f_dash_given_log` (used by the standalone surrogate class). Any
  new divergence must implement the torch path.
- `iwpc.learn_dist.kernels.trainable_kernel_base.TrainableKernelBase` - must
  expose `draw(cond)` and `draw_with_log_prob(cond) -> (sample, log_prob)`.
- `iwpc.learn_dist.kernels.finite_kernel.FiniteKernelInterface` - optional
  exact branch; needs `outcomes_with_log_prob_iter(base_samples)` and
  `parameters()`.
- `iwpc.learn_dist.base_distributions.*` is reached transitively through the
  kernel (kernels consume a `SampleableBaseModel`); this module never imports
  it directly.

## Adding a new surrogate

`FDivergenceGradientSurrogateLoss` is the canonical per-sample form; the
trainer currently inlines the same arithmetic. To swap surrogates: subclass
`FDivergenceMinimizingKernelTrainer` and override `calculate_kernel_loss`,
keeping the `(q_weights, exact_outcome_log_prob, log_prob, log_p_over_q)`
inputs and the score-function structure. The new surrogate's gradient w.r.t.
θ must equal the gradient of the target functional; verify by finite
differences on a small kernel before plugging in.
