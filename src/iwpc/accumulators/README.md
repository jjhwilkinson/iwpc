# `iwpc.accumulators`

## Purpose

Post-hoc divergence estimation from precomputed probability ratios `p(x)/q(x)`.
A trained `FDivergenceEstimator` only stores running expectations during
validation; the accumulators in this package let you re-evaluate the naive
variational lower bound on a saved dataset, with proper sample-mean standard
errors, and optionally attribute the divergence to user-chosen feature axes via
binning. The 4-panel diagnostic plots in the top-level README come from
`BinnedDfAccumulator.plot`.

Labels follow the package-wide convention: **0 = p, 1 = q**.

## Layout

Two largely independent hierarchies live here.

### `DfAccumulator` hierarchy (`Df_accumulator.py`)

- `DfAccumulator` — abstract base. Holds a `DifferentiableFDivergence` and a
  `clip_log_p_over_q=(-14, 14)` numeric-stability bound (matches the clip used
  inside `NaiveVariationalFDivergenceEstimator`). Subclasses implement
  `update(...)`, `accumulated_df`, `accumulated_df_stderr`. The base class
  derives `sig = accumulated_df / accumulated_df_stderr` and a `__str__`.
- `LabeledBinaryNaiveDfAccumulator` — the concrete implementation used
  everywhere. Holds two `WeightedMeanMetric`s (one per label) and updates them
  from the naive-rep summands of the supplied divergence. Signature:
  `update(p_over_q, labels, weights)`. `accumulated_df` is the difference of
  weighted means; `accumulated_df_stderr` adds the two stderrs in quadrature.

### Binned-stat utilities

A small stack of histogram-with-stats accumulators that `BinnedDfAccumulator`
composes:

- `BinnedStatAccumulator(num_statistics, bins)` — tracks count, sum, and outer-
  product sum of `num_statistics` features per bin; exposes `mean_hist`,
  `cov_hist`, `corr_hist`.
- `BinnedWeightedStatAccumulator(bins)` (subclass with `num_statistics=2`) —
  tracks `(weights, weights*values)`; exposes `weighted_mean_hist` and
  `weighted_stderr_hist`. `WeightedHistogramAccumulator` is a thin wrapper that
  fixes `values = weights`.
- `WeightedBinnedStatAccumulator(num_statistics, bins)` — variant tracking a
  weighted sum and the full covariance via a single combined
  `BinnedStatAccumulator`; exposes `weighted_mean_covariance_hist`.
- `HistogramAccumulator(bins, bin_labels=None)` — D-dim weighted histogram with
  `weight_sum_hist`, `normalised_weight_sum_hist`, `mean`, `stds`, and a 1D/2D
  `.plot()`. Construct from scalars via `HistogramAccumulator.from_scalars`.

### `BinnedDfAccumulator` (`binned_Df_accumulator.py`)

Top-level diagnostic. Takes a `ScalarFunction` or list of them (the package's
projection-from-DataFrame abstraction) plus a divergence, and runs two passes
over a `PandasDirDataModule`:

- `update_train(samples, labels, weights, p_over_q)` builds reference
  histograms of p, q, and the network-implied marginalised p and q.
- `update_val(samples, labels, weights, log_p_over_q)` accumulates per-bin
  conditional divergence summands, the global divergence, and the marginalised
  divergence (the part captured by the chosen scalars).

The helper `BinnedDfAccumulator.evaluate(datamodule, p_over_q_cols)` drives the
two passes and computes `log p/q` by multiplying the listed reweight columns
(`construct_p_over_q` clips the product to `[1e-6, 1e6]`).

Plot panels are implemented for **1 or 2** scalars only; `plot` raises
`NotImplementedError` for higher dimensions even though the accumulation is
N-dim.

### `utils.py`

`construct_bin_number(_regular_bins)`, `construct_binned_statistic_result_regular_bins`,
and `faster_binned_statistic_dd_without_overflow` — fast paths for regular bins
on top of `scipy.stats.binned_statistic_dd`, plus an `is_regular_bins` check.

## Usage

### 1. Re-evaluate divergence from a trained estimator

```python
import numpy as np, torch
from iwpc.accumulators.Df_accumulator import LabeledBinaryNaiveDfAccumulator
from iwpc.divergences import KLDivergence

acc = LabeledBinaryNaiveDfAccumulator(KLDivergence())
module.eval()
with torch.no_grad():
    for x, y, w in val_loader:
        log_p_over_q = module(x).cpu().numpy().ravel()
        acc.update(np.exp(log_p_over_q), y.cpu().numpy(), w.cpu().numpy())
print(acc)  # KL(p, q) >= <df> +- <err> (<sig>)
print(acc.accumulated_df, acc.accumulated_df_stderr, acc.sig)
```

`p_over_q` is the ratio itself (not the log); the clip is applied internally.

### 2. Cumulative divergence after a reweight loop

`run_reweight_loop` returns a `LabeledBinaryNaiveDfAccumulator` built by
`calculate_total_divergence`, which multiplies the chain of `p_over_q_i`
columns and feeds the product through `update(...)` once per file.

### 3. Attribute divergence to feature axes

```python
from iwpc.accumulators.binned_Df_accumulator import BinnedDfAccumulator
from iwpc.divergences import JensenShannonDivergence

bda = BinnedDfAccumulator(
    scalars=[scalar_pt, scalar_eta],   # ScalarFunction instances, each with .bins
    divergence=JensenShannonDivergence(),
    p_name="data", q_name="MC",
)
bda.evaluate(datamodule, p_over_q_cols=["p_over_q_0", "p_over_q_1"])
fig, axes = bda.plot(title="JS divergence by (pT, eta)")
# Inspect: bda.perp_df_hist (per-bin conditional Df),
#          bda.perp_df_err_hist, bda.weighted_df_avg, bda.variability_chi_sq_dof,
#          bda.global_df_accumulator.accumulated_df,
#          bda.marginalised_df_accumulator.accumulated_df.
```

For a single scalar `plot` produces four 1D panels (val/train hists,
conditional `Df` vs scalar with weighted average, learned p/q histograms,
val/train/learned ratios); for two scalars, four 2D heatmaps.
