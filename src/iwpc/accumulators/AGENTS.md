# AGENTS — `iwpc.accumulators`

Dense notes for contributors. Read `README.md` first.

## Contract / signatures (verified against source)

- `DfAccumulator.__init__(divergence, clip_log_p_over_q=(-14, 14))`. The default
  clip range matches `NaiveVariationalFDivergenceEstimator` exactly; do not
  silently widen it.
- `LabeledBinaryNaiveDfAccumulator.update(p_over_q, labels, weights)` — note
  **`p_over_q` not `log_p_over_q`**. The accumulator does
  `exp(clip(log(p_over_q), *clip_log_p_over_q))` internally, then calls
  `divergence.calculate_naive_rep_summands_given_log_by_label(p_over_q, labels)`
  and routes label==0 to `p_accumulator`, label==1 to `q_accumulator` (both
  `WeightedMeanMetric` from `iwpc.metrics`).
- `BinnedDfAccumulator.update_train(samples, labels, weights, log_p_over_q)` and
  `update_val(samples, labels, weights, log_p_over_q)` both take the **log
  ratio**. Mixture weights `p/(p+q)` and `q/(p+q)` are recovered via
  `expit(log_p_over_q)` and `expit(-log_p_over_q)` respectively (numerically
  stable across the full range). `update_val` works in log-space throughout
  (subtracts the marginalised log ratio rather than dividing).
- `BinnedDfAccumulator.evaluate(datamodule, p_over_q_cols)` requires a
  `PandasDirDataModule` (uses `file_iter`, `feature_spec[1][0]` as the label
  column, `weight_col`, `train_files`, `validation_files`). The per-file
  pipeline takes `log(construct_p_over_q(df, p_over_q_cols))` and feeds that
  log ratio to both `update_train` and `update_val`. `construct_p_over_q` clips
  the product to `[1e-6, 1e6]`.

## Stat-accumulator stack

Composition is: `HistogramAccumulator` (`num_statistics=1`),
`BinnedWeightedStatAccumulator` (`num_statistics=2`, hand-derived
`weighted_stderr_hist` from cov of the two underlying stats), and the newer
`WeightedBinnedStatAccumulator` (any `num_statistics`, weighted covariance via
Jacobian propagation). `BinnedDfAccumulator` uses the older variants for
backward compatibility; new code should prefer
`WeightedBinnedStatAccumulator` when it needs >1 weighted statistic.

All bin operations go through `utils.faster_binned_statistic_dd_without_overflow`
which **requires regular bins** (asserted in
`construct_bin_number_regular_bins`). Irregular bins will raise.

## Plotting

1D and 2D only. `BinnedDfAccumulator.plot` and `HistogramAccumulator.plot`
both raise `NotImplementedError` for >2 binned dimensions; the accumulation
itself is N-dim, so you can still read `perp_df_hist` directly.

## Cross-package consumers

- `iwpc.reweight_loop.calculate_total_divergence` instantiates
  `LabeledBinaryNaiveDfAccumulator` and feeds it the product of all
  `p_over_q_i` columns; the returned `RunReweightLoopResult` carries it as
  `final_divergence_accumulator`. Anything that changes that constructor or
  `update` signature breaks the reweight loop.
- Top-level `README.md` documents `BinnedDfAccumulator` as the diagnostic
  plot generator — keep `plot()` returning `(Figure, (Axes, Axes, Axes, Axes))`.

## Adding a new accumulator

1. Subclass `DfAccumulator`. Implement `update`, `accumulated_df`,
   `accumulated_df_stderr`. Use `WeightedMeanMetric` (or
   `WeightedBinnedStatAccumulator`) for running means; do **not** roll your own
   stderr.
2. Respect `self.clip_log_p_over_q` in `update`; clip before exponentiation as
   in `LabeledBinaryNaiveDfAccumulator`.
3. Dispatch summand calculation through `self.divergence` (e.g.
   `calculate_naive_p_summands_given_log`) rather than computing the f-conjugate
   manually — keeps numpy/torch dispatch consistent.
4. Numpy-style docstrings on every public method. Don't add to `__init__.py`
   unless re-export is intentional (currently empty).
