# Agent notes: `fdivergence_minimization`

## `FDivergenceMinimizingKernelTrainer.__init__`

```
sampled_kernel: TrainableKernelBase,
log_p_over_q_model: Module,
divergence: DifferentiableFDivergence,
exact_kernel: FiniteKernelInterface | None = None,
log_p_over_q_model_opt_lr: float = 1e-3,
kernel_opt_lr: float = 1e-4,
start_kernel_train_epoch: int = 1,
start_log_p_over_q_model_train_epoch: int = 0,
zero_out_init_q_samples: bool = False,
accumulate_kernel_batches: int = -1,
target_cut_pass_prob: float | None = None,
```

`automatic_optimization = False`; `configure_optimizers` returns a two-element
`list[dict[str, Optimizer]]` of Lightning optimiser-spec dicts
(`log_p_over_q_model`, then kernel + optional `exact_kernel`). The
`log_p_over_q_model` is trained to learn the log density ratio using a
cross-entropy loss; the kernel consumes its detached `log(p/q)` estimate.
Batch contract:
`(base_samples, samples, labels, weights)`, label `0 = p`, `1 = q`.
`zero_out_init_q_samples` overrides the `q_init_samples = samples[q_mask]`
residual term (kernel output becomes the full sample). The "divergence"
logged (`val_divergence = 1 - BCE / log 2`) is the JS lower bound recovered
from the `log_p_over_q_model`, regardless of `self.divergence`.
`target_cut_pass_prob` is only active when `exact_kernel` is a
`FiniteCutKernel` — see the Poisson penalty section below.

## Surrogate gradient identity

Score-function / REINFORCE identity for `Df(p || q) = E_q[ f(p/q) ]`:

```
d/dθ Df(p || q) = E_q[ f'(p/q) * d/dθ log q_θ(x) ]
```

When `exact_kernel` is a `FiniteCutKernel`, q-side samples actually come from
the un-cut base distribution and are reweighted by the per-row cut-pass
probability `r_i = exp(cut_pass_log_prob_i)`; the trainer normalises by the
batch-level (q-weighted) average `r̄ = (1/N) Σ w_i r_i`, computed in log space
via a signed logsumexp that tolerates negative `w_i`.

`calculate_kernel_loss` builds, for each allowed exact outcome,

```
total_q_weight = w * exp(exact_log_prob + cut_pass_log_prob - log r̄).detach()
loss += mean( total_q_weight
              * divergence.f_dash_given_log(-log_p_over_q)
              * (sample_log_prob + cut_pass_log_prob - log r̄) )
```

where `sample_log_prob = sampled_log_prob + exact_log_prob` comes from
`full_sample_iter_and_cut_pass_log_prob` (the only term carrying grad through
θ — `log_p_over_q` and the `r̄` correction are detached). `f_dash_given_log`
evaluates `f'(p/q)` stably from `log(q/p) = -log_p_over_q`.

When `exact_kernel` is `None` or a non-cut `FiniteKernelInterface`,
`cut_pass_log_prob` is identically zero (every sample passes), `r̄` reduces to
`mean(w)`, and the expression collapses to the plain reweighted surrogate.

The ratio-estimator side uses standard BCE (`logsigmoid` on `+log_p_over_q`
for p, on `-log_p_over_q` for q), sample-weighted by `exp(exact_log_prob)`
per discrete branch, with the q-side ratio rescaled by `cut_pass_log_prob` so
that the q expectation is over the cut distribution.

### Poisson penalty (cut kernels only)

When `target_cut_pass_prob = π` is set together with a `FiniteCutKernel`
exact kernel, an additional term is added to the loss:

```
- ( π * log r̄ - r̄ - (π * log π - π) )
```

This is the (negated, normalised) Poisson log-likelihood evaluated at mean
`r̄` with target rate `π`. Minimised at `r̄ = π`, gradients pull the realised
average cut-pass probability toward `π`. The constant `π log π - π` zeroes the
term at the minimum. Disabled when `target_cut_pass_prob is None` or
`exact_kernel` is not a `FiniteCutKernel`.

## Cross-package deps

- `iwpc.divergences.DifferentiableFDivergence` - supplies `f_dash_given_log`
  (public) / `_f_dash_given_log` (used by the standalone surrogate class). Any
  new divergence must implement the torch path.
- `iwpc.learn_dist.kernels.trainable_kernel_base.TrainableKernelBase` - must
  expose `draw(cond)` and `draw_with_log_prob(cond) -> (sample, log_prob)`.
- `iwpc.learn_dist.kernels.finite_kernel.FiniteKernelInterface` - optional
  exact branch; needs `outcomes_with_log_prob_iter(base_samples)` and
  `parameters()`. If it is also a `FiniteCutKernel`, the trainer additionally
  calls `outcome_with_log_prob_iter_and_cut_pass_log_prob(base_samples)` to
  recover the per-row cut-pass log-probability.
- `iwpc.learn_dist.base_distributions.*` is reached transitively through the
  kernel (kernels consume a `SampleableBaseModel`); this module never imports
  it directly.

## Adding a new surrogate

`FDivergenceGradientSurrogateLoss` is the canonical per-sample form; the
trainer currently inlines a cut-aware variant of the same arithmetic. To swap
surrogates: subclass `FDivergenceMinimizingKernelTrainer` and override
`calculate_kernel_loss`, consuming the iterator returned by
`full_sample_iter_and_cut_pass_log_prob(q_base_samples, q_init_samples)` which
yields `(q, sample_log_prob, exact_outcome, exact_log_prob)` per exact-kernel
outcome along with a batch-level `cut_pass_log_prob`. Keep the score-function
structure and the `cut_pass_log_prob - log r̄` reweighting (or drop it
deliberately for non-cut kernels). The new surrogate's gradient w.r.t. θ must
equal the gradient of the target functional; verify by finite differences on
a small kernel before plugging in.
