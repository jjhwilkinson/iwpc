# iwpc.metrics

Streaming `torchmetrics.Metric` accumulators used by the rest of `iwpc` to compute running means, covariances, and weighted means with proper standard errors across training and validation batches. They are the building blocks behind the `val_Df` / `val_Df_err` numbers reported by every `FDivergenceEstimator`.

## What this provides

- `StatMetric` — accumulates the count, sum, and outer-product sum of an `ndim`-dimensional feature vector across batches and exposes the running `means` (length `ndim`) and `cov` (`ndim x ndim`).
- `WeightedMeanMetric` — `StatMetric(2)` specialised to a scalar with per-sample weights; reports the cumulative weighted mean and its standard error using the delta-method ratio formula.

## Usage

```python
import torch
from iwpc.metrics.weighted_mean_metric import WeightedMeanMetric

metric = WeightedMeanMetric()

# Stream batches of (weights, samples) — e.g. p/q ratios * f-conjugate values
for _ in range(10):
    weights = torch.rand(1024)
    samples = torch.randn(1024)
    metric.update(weights, samples)

mean, stderr = metric.compute()
print(f"weighted mean = {mean:.4f} +/- {stderr:.4f}")
```

`StatMetric` is used the same way for general feature vectors:

```python
from iwpc.metrics.stat_metric import StatMetric

stat = StatMetric(ndim=3)
stat.update(x, y, z)            # three 1-D tensors, all the same length
mu = stat.means                 # shape (3,)
sigma = stat.cov                # shape (3, 3)
```

## Related

- `iwpc.modules.naive` and `iwpc.modules.asymmetry_estimator` — use `WeightedMeanMetric` to track the two expectation summands of the variational lower bound, then publish `val_Df` and `val_Df_err`.
- `iwpc.accumulators.Df_accumulator` — uses two `WeightedMeanMetric`s (one over p-samples, one over q-samples) to compute `accumulated_df` / `accumulated_df_stderr` from precomputed probability ratios.