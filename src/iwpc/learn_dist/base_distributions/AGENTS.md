# AGENTS — `learn_dist/base_distributions`

## Contract (`SamplableBaseModel`)

- ABC in `sampleable_base_model.py`. Subclasses must implement:
  - `draw(num_samples: int) -> np.ndarray` returning shape `(num_samples, self.dimension)`.
  - `_log_prob(x: np.ndarray) -> np.ndarray` taking shape `(num_samples, self.dimension)` and returning `(num_samples,)`; assume `total_volume == 1` here.
- Public `log_prob(x)` is concrete: adds `np.log(self.total_volume)` to `_log_prob(x)`. Always override `_log_prob`, never `log_prob`. Edge case: if `x.shape[0] == 0` it short-circuits to `np.zeros((0, self.dimension))` (note: that shape is arguably wrong — flag if touched).
- `__init__(dimension, total_volume=1.0, *args, **kwargs)` stores both. `dimension` is the event dimension `D`; there is no separate batch shape — the leading axis of `x`/samples is the only batch axis.
- Optional classmethod `fit(*args, **kwargs)` for sample-based parameter estimation; default raises `NotImplementedError`. `Cauchy.fit` is explicitly not implemented.
- Everything is numpy-side. `Exponential` and `MultivariateNormal` do import `torch` for unused attributes / sampling helpers — do not assume tensors flow through.

## Operators (combinators)

- `a & b` -> `ConcatenatedBaseModel`. Independent product. `dimension = sum(sub.dimension)`. `_log_prob` slices `x` along the last axis using cumulative sub-dimensions and sums sub-`log_prob`s (so `total_volume` factors multiply correctly). `__and__` is overridden on `ConcatenatedBaseModel` to flatten chains.
- `a + b` -> `MixtureBaseModel`. Requires all sub-models share `dimension`. Mixing weights = normalised `total_volume`s. `draw` uses `np.random.multinomial` then permutes. `_log_prob` uses `scipy.special.logsumexp` over sub-`log_prob`s (note: each sub `log_prob` already bakes in `log(total_volume)`, so the logsumexp gives the unnormalised mixture density; the outer `log_prob` then adds the parent `total_volume=1`, which is correct only because `MixtureBaseModel.__init__` does not pass through a custom `total_volume`).
- `c * model` -> deep copy with scaled `total_volume`. Only float / 0-D ndarray accepted.

## Cross-package consumers

- `iwpc/learn_dist/classifier_reweighting.py` — takes a `SamplableBaseModel` as the proposal distribution.
- `iwpc/learn_dist/kernels/` — trainable kernels (`gaussian_kernel`, `multivariate_gaussian_kernel`, `two_sided_exponential_kernel`, mixture/concatenated/conditioned kernels) consume base distributions as the source of latent noise. They do not import the base models directly here today; coupling is at user-code level (instantiate base, draw samples, push through kernel). Mirror the operator algebra (`&`, `+`) when building matching kernel structures.
- `iwpc/learn_dist/fdivergence_minimization/` — the training loop is agnostic; base distributions appear via the kernels.

## Warts

- Filename typo: `multivaraite_normal_base_model.py` (should be `multivariate_...`). The class inside is `MultivariateNormalBaseModel`. Any rename is a public-API break — coordinate with `learn_dist/__init__.py` re-exports if/when fixed.
- `CauchyBaseModel.fit` references `Optional` without importing it — calling it would `NameError` before the `NotImplementedError`. Harmless today.
- `HistogramBaseModel.draw` smears samples uniformly within bins.
- `log_prob`'s empty-input branch returns shape `(0, dimension)` rather than `(0,)` — inconsistent with the populated branch.
