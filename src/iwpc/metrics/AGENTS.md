# iwpc.metrics — AGENTS.md

## Public API
- `StatMetric(ndim: int)` — `torchmetrics.Metric` tracking count, sum, and outer-product sum of an `ndim`-dim feature vector.
  - `update(*arrs: TensorOrNDArray) -> None` — pass `ndim` 1-D arrays of equal length; each is one feature.
  - `compute() -> Tensor` — alias for `means`; shape `(ndim,)`.
  - `means: Tensor` — running mean, shape `(ndim,)`.
  - `cov: Tensor` — running covariance `E[xx^T] - E[x]E[x]^T`, shape `(ndim, ndim)`.
  - State (registered via `add_state`, `dist_reduce_fx="sum"`): `N`, `sums`, `outer_prod_sums`.
- `WeightedMeanMetric()` — subclass of `StatMetric(2)` for scalar weighted-mean tracking.
  - `update(weights: TensorOrNDArray, samples: TensorOrNDArray) -> None` — internally calls `super().update(weights, weights * samples)`.
  - `compute() -> Tuple[Tensor, Tensor]` — returns `(weighted_mean, weighted_stderr)`.
  - `weighted_mean: Tensor` — `means[1] / means[0]` = `sum(w*x) / sum(w)`.
  - `weighted_stderr: Tensor` — delta-method ratio standard error from `cov / N`.

## Invariants
- `StatMetric.update` stacks inputs as `torch.stack([as_tensor(a) for a in arrs]).T`, so every `arrs[i]` must be 1-D and the same length, and `len(arrs) == ndim`.
- All state tensors are summed across distributed ranks (`dist_reduce_fx="sum"`); never overwrite them, only accumulate.
- `WeightedMeanMetric` fixes `ndim=2`: feature 0 is the weight, feature 1 is `weight * sample`. Do not call `StatMetric.update` on it directly with arbitrary arity.
- `__init__.py` is empty — import from the leaf modules (`iwpc.metrics.stat_metric`, `iwpc.metrics.weighted_mean_metric`).

## Numerical tricks
- Outer product accumulated as `(samples[:, :, None] * samples[:, None, :]).sum(0)` so covariance can be reconstructed from running sums without storing samples.
- `weighted_stderr` uses the standard ratio-of-means delta-method formula `|r| * sqrt(var_a/a^2 + var_b/b^2 - 2*cov_ab/(a*b))` with `a = sum(w)/N`, `b = sum(w*x)/N`, and `cov / N` as the covariance of the sample means. This is correct only in the large-`N` regime; small batches give noisy stderr.

## Conventions
- Inputs accept `TensorOrNDArray` (`Union[Tensor, ndarray]`); arrays are coerced via `torch.as_tensor`.
- Numpy is used inside `cov` (`np.newaxis`) — works because `means` is a torch `Tensor` and `np.newaxis is None`.
- These metrics are device-agnostic via `torchmetrics.Metric`; state lives wherever the parent `LightningModule` places it.

## Gotchas
- `StatMetric.update` takes *positional* per-feature arrays, not a single `(N, ndim)` batch. Calling `metric.update(samples)` with a 2-D tensor will silently mis-shape the state.
- `WeightedMeanMetric.update(weights, samples)` — order matters; swapping them computes `mean(w) / mean(s*w)` instead.
- `compute()` on `StatMetric` only returns the means, not the covariance; callers wanting covariance must read `.cov` explicitly.
- Calling `compute()` before any `update()` divides by `N=0` and yields NaNs/Infs; guard upstream.

## Cross-refs
- Imports from: `iwpc.types` (`TensorOrNDArray`), `torchmetrics.Metric`, `torch`, `numpy`.
- Used by:
  - `iwpc.modules.naive.NaiveVariationalFDivergenceEstimator` — `val_p_accumulator`, `val_q_accumulator` for the two expectation summands of `val_Df`.
  - `iwpc.modules.asymmetry_estimator` — `val_accumulator`.
  - `iwpc.accumulators.Df_accumulator.DfAccumulator` — `p_accumulator`, `q_accumulator` for offline divergence estimation with stderr.
