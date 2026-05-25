# `iwpc.learn_dist.kernels` — agent notes

## Core contract: `TrainableKernelBase`

Every kernel is a `LightningModule` exposing:

- `sample_dimension: int` and `cond_dimension: int` (derived from an
  `Encoding.input_shape[0]` if one is passed in).
- `log_prob(samples, cond) -> (N,)` — differentiable.
- `_draw(cond) -> (N, sample_dimension)` — abstract; **do not call
  directly**, use `draw(cond)` (wraps in `no_grad` and handles the
  empty-batch case).
- `draw_with_log_prob(cond) -> (samples, log_prob)` — defaults to
  `draw` + `log_prob` but should be overridden when the two can share
  work (e.g. Gaussian).
- Operators: `&` -> `ConcatenatedKernel(.., concatenate_cond=False)`,
  `+` -> `ConcatenatedKernel(.., concatenate_cond=True)`, `|` ->
  `ConditionedKernel(self, other)`. `FiniteKernelInterface` overrides
  these to produce `Finite*` variants when both operands are finite.
- Default training loop: `calculate_loss = -mean(finite(log_prob))`,
  Adam @ 1e-3, `ReduceLROnPlateau(monitor="val_loss", mode="min")`.
  Batches are `(cond, targets, weights)` (weights currently unused).

## Structural combinators

What composes with what:

- `ConcatenatedKernel(sub_kernels, concatenate_cond)` — sample dim is
  the sum of sub sample dims; cond is either shared (False) or sliced
  per child by cumulative `cond_dimension` (True). `merge` flattens
  nested same-mode concatenations. `draw_with_separate_log_prob` is the
  hook used by `UnlabelledMultiKernelTrainer`.
- `ConditionedKernel(sample_kernel, conditioning_kernel)` — requires
  `sample_kernel.cond_dimension == conditioning_kernel.sample_dimension
  + conditioning_kernel.cond_dimension`. Sample layout: first
  `sample_kernel.sample_dimension` columns come from `sample_kernel`,
  rest from `conditioning_kernel`.
- `MixtureKernel` — sub-kernels must share both sample and cond
  dimensions; weights come from a `LogSoftmaxEncoding` head.
- `BranchingKernel` — sub-kernels share sample and cond dims; a slice
  of `cond` (`branch_sample_indices`) is mapped to an integer via
  `outcome_to_idx_fn` and routed to one of `sub_kernels`. Use the
  `condition_on(sub_kernels, finite_kernel)` classmethod plus the
  `[k1, k2, ...] | finite_kernel` (`__ror__`) shorthand to dispatch on
  a `FiniteKernelInterface`. `FiniteBranchingKernel` is the
  finite-aware variant.
- `RestructuringKernel` — reindexes `cond` before a base kernel.
- `PermutationKernel` — reorders the sample axis of a base kernel.
- `AddCondKernel` — `y = cond + base.sample`; requires
  `base.sample_dimension == base.cond_dimension`. Honour
  `custom_encoding` for circular features.
- `CutKernelInterface` — defines `cut_pass_log_prob` and
  `cut_fail_log_prob`. `FiniteCutKernel` is the concrete finite
  implementation (built via `finite.cut(fn)` or `sample_space.cut(fn)`)
  and additionally exposes
  `outcome_with_log_prob_iter_and_cut_pass_log_prob(cond)`, which shares
  the base kernel's log-probs between the per-outcome iterator and the
  cut-pass log-probability so gradients flow through both.

## Finite kernels

`FiniteKernelInterface` exists so any kernel over an enumerable sample
space can compose log-prob *tables* rather than chaining
`exp -> sample -> log_prob`. Implementors override
`construct_log_probs(cond) -> (N, num_outcomes)` (must be normalised:
`logsumexp(dim=-1) == 0`); the interface provides `log_prob`, `_draw`
(via `sample_idx_from_log_probs`), `draw_with_log_prob`, and
`outcomes_with_log_prob_iter`.

`FiniteSampleSpace` is an `nn.Module` providing
`outcome_to_idx` / `idx_to_outcome` / `outcomes_iter`. Composition:

- `ConcatenatedFiniteSampleSpace` — cartesian product, built via `&`.
- `CutFiniteSampleSpace` — subset of a base space, built via
  `.cut(fn)`.
- `ExplicitFiniteSampleSpace` — list of outcomes plus a user-supplied
  `outcome_to_idx_fn`.

Composite finite kernels are designed to keep log-prob composition flat
(one softmax per leaf). `FiniteConcatenatedKernel` sums child
log-probs; `FiniteConditionedKernel` enumerates the conditioning
kernel's outcomes (or, if both children are `IndexedInterface`,
broadcasts directly).

## Indexed kernels

`IndexedInterface` is an opt-in extension for finite kernels modelling
`p(A | B = b, x)` where `B` is discrete and lives at fixed positions
in `cond` (`index_cond_indices`). The implementor provides
`construct_log_prob_table(x) -> (N, M, K)` so the cost of evaluating
all `K` index values is **one** forward pass through a single network
with a wide head. `IndexedInterface.__or__` auto-promotes a
`FiniteConditionedKernel` to `IndexedFiniteConditionedKernel` when both
children are indexed and the index-cond-indices line up. Use only when
the index variable has few outcomes and is read directly from `cond`.

## Trainers (which to pick)

All three are `LightningModule`s that take an outer Lightning `Trainer`
and a co-trained `log_p_over_q_model`; batches are
`(base_samples, data_samples, labels, weights)` with `label==0` for real
data (`p`) and `label==1` for kernel samples (`q`). The
`log_p_over_q_model` is trained to learn the log density ratio using a
cross-entropy loss; the kernel's loss consumes the detached `log(p/q)`
estimate.

- `UnLabelledKernelTrainer` — default. Alternates a `log_p_over_q_model` step
  with a kernel step via `KernelKLDivergenceGradientLoss`
  (`-mean(weights * log_prob * stop_grad(p/q))` from kernel samples).
  Gates kernel updates on `train_divergence > min_train_divergence`;
  decays `min_train_divergence` if the `log_p_over_q_model` saturates; drops
  `kernel_lr` via `KernelLRAdjustor` (fits a line to recent
  divergences, drops LR if the residual std rivals the intercept).
- `PartiallyExactUnLabelledKernelTrainer` — for joint kernels of the
  form `(finite exact_kernel) | (sampled_kernel)`. Enumerates outcomes
  of the finite part exactly and only Monte-Carlos the continuous part
  (uses a `JensenShannonDivergence._f_dash_given_log_torch` weighting).
  Lower-variance when applicable.
- `UnlabelledMultiKernelTrainer` — for a `ConcatenatedKernel` whose
  independent factors each have their own `log_p_over_q_model`
  (`log_p_over_q_models: list`). Uses
  `MultiKernelKLDivergenceGradientLoss` which calls
  `combined_kernel.draw_with_separate_log_prob`.

`KernelKLDivergenceGradientLoss` is the load-bearing object: its
gradient w.r.t. kernel parameters equals the gradient of the KL
divergence of the data under the model, while only requiring samples
from the base distribution.

## Cross-package dependencies

- `iwpc.encodings.*` — every cond/sample dim can be passed as either an
  `int` or an `Encoding`. `basic_model_factory(input, output, ...)`
  builds the default subnetworks; `LogSoftmaxEncoding`,
  `ExponentialEncoding`, `AntiSymmetricMatrixEncoding`, and
  `TrivialEncoding` are reused widely.
- `iwpc.models.utils.basic_model_factory` and
  `iwpc.models.layers.ConstantScaleLayer` — used by every kernel that
  builds default subnetworks and by `MultivariateGaussianKernel.initialise_cov`
  for warm-starting.
- `iwpc.divergences.JensenShannonDivergence` — used inside
  `PartiallyExactKernelKLDivergenceGradientLoss`.
- `iwpc.learn_dist.base_distributions` — the typical `q` is the
  convolution of one of those measures with a kernel from this package.
- `iwpc.learn_dist.fdivergence_minimization` — alternative training
  loop that consumes the same `TrainableKernelBase` contract.

## Adding a new kernel

1. Decide if the output is continuous or finite.
2. Continuous: subclass `TrainableKernelBase`; implement `log_prob`
   and `_draw`; override `draw_with_log_prob` if the two can share a
   forward pass (see `GaussianKernel`). Accept `cond` as
   `int | Encoding` and pass it straight to `super().__init__`.
3. Finite: subclass `FiniteKernelInterface` together with
   `TrainableKernelBase` (in that MRO order — see `FiniteKernel`,
   `ConstantKernel`, `FixedFiniteKernel`); implement
   `construct_log_probs(cond)` returning a normalised `(N, K)` table
   over a `FiniteSampleSpace` you pass to `super().__init__`. If the
   kernel naturally exposes one logit table per index value, also mix
   in `IndexedInterface` and implement `construct_log_prob_table`.
4. Use `basic_model_factory(input_encoding, output_encoding)` for the
   default parameter networks so users can swap encodings without
   re-implementing the kernel.
5. Numerical stability: clip extreme log-probabilities to finite
   values, expose a `max_chi` outlier mask if the distribution has
   heavy tails, and prefer in-place log-space arithmetic over
   `exp`/`log` round trips.
6. Re-use the `&` / `+` / `|` operators where possible — the combinator
   classes are usually preferable to writing a bespoke joint kernel.
