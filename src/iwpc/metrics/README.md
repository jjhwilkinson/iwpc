# `iwpc.metrics`

Small `torchmetrics.Metric` subclasses used to accumulate running statistics across training/validation batches with proper distributed reduction. `WeightedMeanMetric` is what backs the `val_Df` / `val_Df_err` numbers that the rest of `iwpc` early-stops and checkpoints on.

## Layout

- `stat_metric.py` — `StatMetric(ndim)`: tracks `N`, `sums`, and `outer_prod_sums` of a length-`ndim` feature vector; exposes `.means` and `.cov`.
- `weighted_mean_metric.py` — `WeightedMeanMetric()`: a 2-feature `StatMetric` that takes `(weights, samples)` and computes a weighted mean plus a properly propagated standard error.

## Usage

`WeightedMeanMetric` is the typical entry point. It is updated once per batch with `(weights, scalar_per_sample)`:

```python
from iwpc.metrics.weighted_mean_metric import WeightedMeanMetric

metric = WeightedMeanMetric()
metric.update(weights, log_p_over_q)        # any (N,) tensors
mean, stderr = metric.compute()
```

`NaiveVariationalFDivergenceEstimator` keeps two of these to track the two summands of the variational lower bound and logs their difference as `val_Df` (see `src/iwpc/modules/naive.py`).

`StatMetric` can be used directly when you want both means and a covariance, e.g. to monitor running statistics of a multidimensional feature:

```python
from iwpc.metrics.stat_metric import StatMetric

stat = StatMetric(ndim=3)
stat.update(x, y, z)         # three (N,) tensors
means = stat.means           # shape (3,)
cov   = stat.cov             # shape (3, 3)
```
