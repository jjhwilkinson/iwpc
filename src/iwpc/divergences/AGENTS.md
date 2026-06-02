# `iwpc.divergences` — agent notes

The end-to-end divergence-estimation flow. Math + estimator hierarchy + training entry points all live here as of 0.10.0 (calculate_divergence and run_reweight_loop moved out of `iwpc/` top-level; FDivergenceEstimator hierarchy moved out of `iwpc.modules/`).

## Public re-exports (`__init__.py`)

`DifferentiableFDivergence`, `KLDivergence`, `JensenShannonDivergence`, `FDivergenceEstimator`, `NaiveVariationalFDivergenceEstimator`, `GenericNaiveVariationalFDivergenceEstimator`, `AsymmetryEstimator`, `calculate_divergence`, `DivergenceResult`, `run_reweight_loop`.

## Math layer

### Dual numpy/torch contract

Every concrete `DifferentiableFDivergence` MUST implement six abstract hooks (`base.py:54-158`):
- `_f_torch`, `_f_np`
- `_f_conj_torch`, `_f_conj_np`
- `_f_dash_given_log_torch`, `_f_dash_given_log_np`

Torch and numpy implementations must be numerically equivalent — accumulators run the numpy path on cached arrays; estimators run the torch path during training. Drift between them silently corrupts results. Never call the underscore methods directly; always go through `f`, `f_conj`, `f_dash_given_log`.

### Dispatch

`_np_or_torch(x, torch_fn, np_fn)` (`base.py:33-52`) keys off `isinstance(x, torch.Tensor)`; anything else (incl. `np.ndarray`, scalars, lists) routes to the numpy branch.

### Log-stable parameterisation

`f_dash` is taken at `p/q = exp(log_p_over_q)` and easily overflows. The interface exposes `f_dash_given_log(log_x)` so subclasses rewrite the derivative in log-space.
- KL: `1 + log_x` (`kl_divergence.py:27-31`).
- JSD: uses `logaddexp` (`jensen_shannon_divergence.py:27-31`).

Callers clip `log(p/q)` to `[-14, 14]` before passing it in (`naive.py:47,63`); new divergences should remain finite on that interval.

### Naive representation helpers (on the base class)

- `calculate_naive_p_summands_given_log(log_p_over_q) = f_dash_given_log(log_p_over_q)`.
- `calculate_naive_q_summands_given_log(log_p_over_q) = f_conj(f_dash_given_log(log_p_over_q))`.
- `calculate_naive_rep_summands_given_log_by_label(log_p_over_q, label)` splits via `iwpc.utils.split_by_mask`. Convention: **label 0 = p, label 1 = q** (`base.py:244-252`).
- `naive_estimate_given_log(log_p_over_q, label, weights)` returns a weighted scalar lower-bound estimator (no error bars; for SE use `iwpc.accumulators.DfAccumulator`).

## Estimator hierarchy

### Subclass contract (`fdivergence_base.py:57-93`)

- `_configure_metrics(self)` — MUST set `self.val_Df` and `self.val_Df_err` (typically from `iwpc.metrics.WeightedMeanMetric`, which supports `[0]` = mean, `[1]` = stderr — see `naive.py:24-27`). `self.val_Df_sig = val_Df / val_Df_err` is computed for you.
- `_calculate_batch_loss(self, batch) -> Tensor` — return the **negative** of the train estimate of `D_f` (the quantity to minimise).
- `_accumulate_validation_Df(self, batch)` — update the val metrics; do NOT log manually.

### What the base class gives you for free

- `training_step` (`fdivergence_base.py:95`): runs `_calculate_batch_loss`, logs `train_loss`, NaN-guards, stashes `self.prev_batch`.
- `validation_step` (`fdivergence_base.py:121`): runs `_accumulate_validation_Df` then logs `val_Df`, `val_Df_err`, `val_Df_sig` on epoch. These names are what `calculate_divergence`, `ModelCheckpoint`, `EarlyStopping`, and the LR scheduler monitor with `mode="max"`.
- `configure_optimizers` (`fdivergence_base.py:140`): Adam + `ReduceLROnPlateau(monitor="val_Df", mode="max")`. Pass `lr_decay_factor=None` to disable scheduling.
- `forward(x)` returns `self.model(x[0])` — treats input as a batch tuple.

### Batch / label contract

`(features, labels, weights)` with **labels 0 = p, 1 = q** (`naive.py:43-44,65` — uses `iwpc.utils.split_by_mask`). `AsymmetryEstimator` ignores `labels` because it operates on a single distribution (`asymmetry_estimator.py:56,70`).

### Numerical stability

`log(p/q)` clipped to `[-14, 14]` in **both** train loss (`naive.py:47`) and validation accumulation (`naive.py:63`).

## Training entry points

### `calculate_divergence(module, data_module, patience=20, resume_training_from=None, log_dir=cwd, name=None, trainer_kwargs=None)`

Args order: **module first, data_module second** (`calculate_divergence.py:40`). Builds `ModelCheckpoint(monitor="val_Df", mode="max")` + `EarlyStopping(monitor="val_Df", mode="max", patience)` + `LearningRateMonitor`, runs `trainer.fit`, reloads the best checkpoint, validates, returns `DivergenceResult(divergence, divergence_stderr, data_module, best_module, best_model_checkpoint_path, trainer)`. `.sig = divergence / divergence_stderr` is a property.

### `run_reweight_loop(estimator, data_module, ...)`

Only works with `PandasDirDataModule` (uses `.transform` / `.reweight` / `.copy` and the `ds_info['tags']` history). When `result.sig > min_sig`, appends a `p_over_q_{i}` column, scales the weight column by `min(p/q, q/p)` clipped at 1, decays the LR, re-runs. `calculate_total_divergence` reads the final dataset, multiplies the `p_over_q_*` columns, and feeds the product through a `LabeledBinaryNaiveDfAccumulator`.

## Cross-package deps

- `iwpc.metrics.WeightedMeanMetric` — builds `val_Df` / `val_Df_err`.
- `iwpc.models.utils.basic_model_factory` — used by `GenericNaiveVariationalFDivergenceEstimator` (`naive.py:98`); accepts an `int`/`tuple` input shape or an `iwpc.encodings.Encoding`.
- `iwpc.encodings.Encoding` — `GenericNaiveVariationalFDivergenceEstimator(input=...)` forwards to the factory.
- `iwpc.symmetries.GroupAction` — `AsymmetryEstimator` calls `group.symmetrize(...)` (`asymmetry_estimator.py:30`).
- `iwpc.data_modules.PandasDirDataModule` — `run_reweight_loop` is hard-coupled.
- `iwpc.accumulators.LabeledBinaryNaiveDfAccumulator` — `calculate_total_divergence` post-loop.

## Adding a new estimator

1. Subclass `FDivergenceEstimator`; forward `model`, `divergence`, `initial_learning_rate`, `lr_patience`, `lr_decay_factor` to `super().__init__`.
2. In `_configure_metrics`, assign `self.val_Df` and `self.val_Df_err`.
3. In `_calculate_batch_loss`, return the **negative** train estimate using `self.divergence`'s `naive_*` helpers and the label convention; clip exponentiated log-ratios to `[-14, 14]`.
4. In `_accumulate_validation_Df`, update your metrics — do NOT log manually.
5. If you wrap a model factory, call `self.save_hyperparameters()` after `super().__init__` (`naive.py:100`).
6. Re-export from `divergences/__init__.py` if public.

## Adding a new divergence

1. Subclass `DifferentiableFDivergence`, call `super().__init__(name, short_name)`.
2. Implement the six `_*_torch` / `_*_np` hooks. Mirror the math exactly across backends.
3. Prefer log-space identities (`torch.logaddexp`, `torch.log1p`, `np.logaddexp`, `np.log1p`).
4. Re-export from `__init__.py`.

## Don'ts

- Don't introduce per-instance state that differs between numpy and torch paths (JSD's `self.log_two` is torch-only; the numpy branch hardcodes `np.log(2.)` — follow that pattern).
- Don't add a divergence whose `f_conj` domain excludes the outputs of `f_dash_given_log` on the clipped log-ratio range — training will NaN.
- Don't bypass `_np_or_torch` by type-checking in subclasses.
- Don't call `self.model(x)` on the batch tuple — only `forward` unpacks it.
