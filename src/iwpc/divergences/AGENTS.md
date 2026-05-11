# `iwpc.divergences` — agent notes

## Role
- Defines the f-divergence interface (`DifferentiableFDivergence`) and the two concrete divergences shipped today: `KLDivergence`, `JensenShannonDivergence`.
- Pure math: no Lightning, no torch.nn, no I/O. Consumed by `iwpc.modules` (training) and `iwpc.accumulators` (post-hoc estimation).

## Dual numpy/torch contract
- Every concrete subclass MUST implement all six abstract hooks (`base.py:54-158`):
  - `_f_torch`, `_f_np`
  - `_f_conj_torch`, `_f_conj_np`
  - `_f_dash_given_log_torch`, `_f_dash_given_log_np`
- The torch and numpy implementations must be numerically equivalent — accumulators run the numpy path on cached arrays; estimators run the torch path during training. Drift between them silently corrupts results.
- Never call the underscore methods directly. Always go through `f`, `f_conj`, `f_dash_given_log`.

## Dispatch
- `_np_or_torch(x, torch_fn, np_fn)` (`base.py:33-52`) keys off `isinstance(x, torch.Tensor)`; anything else (including `np.ndarray`, scalars, lists) routes to the numpy branch.
- This means `f(some_python_float)` runs through numpy ops — that's the intended fallback.

## Log-stable parameterisation
- `f_dash` is taken at `p/q`, which is `exp(log_p_over_q)` and easily overflows. The interface exposes `f_dash_given_log(log_x)` instead so subclasses can rewrite the derivative in log-space.
  - KL: `1 + log_x` (`kl_divergence.py:27-31`).
  - JSD: uses `logaddexp` (`jensen_shannon_divergence.py:27-31`), with a precomputed `self.log_two = torch.log(torch.tensor(2.))` for the torch path.
- Callers (`NaiveVariationalFDivergenceEstimator` in `modules/naive.py`) clip `log(p/q)` to `[-14, 14]` before passing it in; new divergences should remain finite on that interval.

## Naive representation helpers (on the base class)
- `calculate_naive_p_summands_given_log(log_p_over_q) = f_dash_given_log(log_p_over_q)`.
- `calculate_naive_q_summands_given_log(log_p_over_q) = f_conj(f_dash_given_log(log_p_over_q))`.
- `calculate_naive_rep_summands_given_log_by_label(log_p_over_q, label)` splits via `iwpc.utils.split_by_mask` and returns `(p_summands, q_summands)`. Convention: **label 0 = p, label 1 = q** (matches batch contract repo-wide; the base-class docstring's "True/False" wording at `base.py:244-252` reads backwards but the implementation matches the repo convention because `split_by_mask` returns `(true, false)` and unpacking is `(q,), (p,)`).
- `naive_estimate_given_log(log_p_over_q, label, weights)` returns a weighted scalar lower-bound estimator (no error bars; for SE use `iwpc.accumulators.DfAccumulator`).

## Adding a new divergence
1. Subclass `DifferentiableFDivergence`, call `super().__init__(name, short_name)` (e.g. `"Reverse-KL", "rKL"`).
2. Implement the six `_*_torch` / `_*_np` hooks. Mirror the math exactly across backends.
3. Prefer log-space identities (`torch.logaddexp`, `torch.log1p`, `np.logaddexp`, `np.log1p`) so `_f_dash_given_log_*` stays finite for `log_x` in `[-14, 14]`.
4. Re-export from `__init__.py`.
5. No tests exist; smoke-test by running `examples/parity_example.py` with `divergence=YourDivergence()`.

## Cross-package consumers
- `iwpc.modules.naive.NaiveVariationalFDivergenceEstimator` — calls `divergence.calculate_naive_rep_summands_given_log_by_label(...)` inside its training/validation step; tracks the two expectations with `WeightedMeanMetric` and reports their difference as `val_Df`.
- `iwpc.accumulators.DfAccumulator` and friends — call `f_dash_given_log` / `f_conj` on numpy arrays of cached `log(p/q)` estimates to produce divergence point estimates with standard errors.
- `iwpc.reweight_loop.run_reweight_loop` — divergence-agnostic but inherits the same contract through `calculate_divergence`.
- `iwpc.learn_dist` — reuses `DifferentiableFDivergence` for the distribution-learning loss; same numpy/torch dual-implementation rule applies.

## Don'ts
- Don't introduce per-instance state that differs between the numpy and torch paths (JSD's `self.log_two` is a torch tensor used only in the torch branch; the numpy branch hardcodes `np.log(2.)` — follow that pattern).
- Don't add a divergence whose `f_conj` domain excludes the outputs of `f_dash_given_log` on the clipped log-ratio range — training will produce NaNs.
- Don't bypass `_np_or_torch` by type-checking in subclasses.
