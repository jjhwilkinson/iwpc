# `iwpc.modules` — agent notes

The estimator hierarchy. `FDivergenceEstimator` (`fdivergence_base.py:14`) is
the abstract Lightning base; everything else specialises it for a particular
variational representation.

## Subclass contract

A concrete estimator implements three abstract methods
(`fdivergence_base.py:57-93`):

- `_configure_metrics(self)` — called from `__init__`
  (`fdivergence_base.py:54`). MUST set `self.val_Df` and `self.val_Df_err`
  (typically from `iwpc.metrics.WeightedMeanMetric`, which supports `[0]`=mean,
  `[1]`=stderr indexing — see `naive.py:24-27`).
  `self.val_Df_sig = val_Df / val_Df_err` is computed for you right after
  (`fdivergence_base.py:55`).
- `_calculate_batch_loss(self, batch) -> Tensor` — return the **negative** of
  the train estimate of $D_f$ (the quantity to minimise).
- `_accumulate_validation_Df(self, batch)` — update the val metrics.

## What the base class gives you for free

- `training_step` (`fdivergence_base.py:95`): runs `_calculate_batch_loss`,
  logs `train_loss`, NaN-guards, stashes `self.prev_batch` for debugging.
- `validation_step` (`fdivergence_base.py:121`): runs
  `_accumulate_validation_Df` then logs `val_Df`, `val_Df_err`, `val_Df_sig`
  on epoch. These names are what `calculate_divergence` /
  `ModelCheckpoint` / `EarlyStopping` / the LR scheduler monitor with
  `mode="max"` (higher = tighter lower bound).
- `configure_optimizers` (`fdivergence_base.py:140`): Adam +
  `ReduceLROnPlateau(monitor="val_Df", mode="max")`. Pass `lr_decay_factor=None`
  to disable scheduling.
- `forward(x)` returns `self.model(x[0])` — treats input as a batch tuple.

## Batch / label contract

Batches are `(features, labels, weights)` with **labels 0 = p, 1 = q**
(`naive.py:43-44,65` — uses `iwpc.utils.split_by_mask`).
`AsymmetryEstimator` ignores `labels` because it operates on a single
distribution (`asymmetry_estimator.py:56,70`).

## Numerical stability

Naive estimator clips $\log(p/q)$ to `[-14, 14]` before exponentiation in
**both** the train loss (`naive.py:47`) and validation accumulation
(`naive.py:63`). Follow the same pattern in new estimators.

## Cross-package deps

- `iwpc.divergences.DifferentiableFDivergence` — `naive_estimate_given_log`,
  `calculate_naive_rep_summands_given_log_by_label`,
  `calculate_naive_{p,q}_summands_given_log` are the loss/val primitives.
- `iwpc.metrics.WeightedMeanMetric` — builds `val_Df` / `val_Df_err`.
- `iwpc.models.utils.basic_model_factory` — used by
  `GenericNaiveVariationalFDivergenceEstimator` (`naive.py:98`); accepts either
  an `int`/`tuple` input shape or an `iwpc.encodings.Encoding`.
- `iwpc.symmetries.GroupAction` — `AsymmetryEstimator` calls
  `group.symmetrize(...)` to Haar-average the q-summand
  (`asymmetry_estimator.py:30`).

Downstream consumer: `iwpc.calculate_divergence.calculate_divergence` wraps any
`FDivergenceEstimator` in a Lightning `Trainer` and reloads the best `val_Df`
checkpoint.

## `utility_modules/`

Not estimators. `IndependentSumModule` is a generic `nn.Module` that runs a
list of sub-networks on configurable feature-index subsets and averages their
outputs (optionally post-encoded). Use it as the `model` of an
`FDivergenceEstimator` when the log-ratio decomposes additively over feature
groups.

## Adding a new estimator

1. Subclass `FDivergenceEstimator`; forward `model`, `divergence`,
   `initial_learning_rate`, `lr_patience`, `lr_decay_factor` to `super().__init__`.
2. In `_configure_metrics`, assign `self.val_Df` and `self.val_Df_err`.
3. In `_calculate_batch_loss`, return the **negative** train estimate using
   `self.divergence`'s `naive_*` helpers and the label convention; clip
   exponentiated log-ratios to `[-14, 14]`.
4. In `_accumulate_validation_Df`, update your metrics — do NOT log manually.
5. If you wrap a model factory, call `self.save_hyperparameters()` after
   `super().__init__` (`naive.py:100`).

## Gotchas

- `asymmetry_estimator.py:77-81` is dead code after a `return`; don't copy it.
- `self.model(x)` is called on raw features inside loss / val paths — only
  `forward` unpacks the batch tuple.
