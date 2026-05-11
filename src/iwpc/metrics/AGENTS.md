# AGENTS — `iwpc.metrics`

- `StatMetric(ndim)` — `torchmetrics.Metric` tracking `N`, `sums` (shape `(ndim,)`), `outer_prod_sums` (shape `(ndim, ndim)`). All three states reduce with `dist_reduce_fx="sum"`, so distributed training works for free. `compute()` returns `means`; covariance via the `.cov` property.
- `WeightedMeanMetric` — subclass of `StatMetric(2)`. `update(weights, samples)` forwards `(weights, weights * samples)` to the parent. `compute()` returns `(weighted_mean, weighted_stderr)`; the stderr formula propagates the full 2x2 covariance through `means[1] / means[0]`, so weight and sample uncertainties are both captured.

## Subclass contract

- Anything subclassing `StatMetric` must keep `update` providing a list of length-`ndim` 1D tensors (it stacks them with `torch.stack([...]).T`). Mismatched lengths or higher-dim inputs will silently produce garbage.
- `WeightedMeanMetric` hardcodes `ndim=2`; do not pass it elsewhere.

## Cross-package consumers

- `src/iwpc/divergences/naive.py` — `NaiveVariationalFDivergenceEstimator` keeps two `WeightedMeanMetric`s for the two variational summands; this is how `val_Df` and `val_Df_err` are produced.
- `src/iwpc/divergences/asymmetry_estimator.py` — same pattern for the asymmetry estimator.
- `src/iwpc/accumulators/Df_accumulator.py` — `DfAccumulator` uses `WeightedMeanMetric` to roll up offline divergence estimates with stderrs.

Nothing else imports from this package; new estimators that need `val_Df` should follow the naive module's pattern and log `(mean, stderr)` from one of these.

## Gotchas

- The stderr expression in `weighted_mean_metric.py` divides by `means[0]` and `means[1]`; if either summand mean is zero (e.g. all-zero weights or all-zero samples in early steps) you will get NaNs/Infs. The naive estimator avoids this by only logging after the validation epoch ends.
- These metrics accumulate forever until `reset()` is called. Lightning calls `reset()` automatically between train/val epochs; manual users must too.
- `update` calls `torch.as_tensor` on each arg, so numpy inputs work but device placement is the caller's responsibility.
