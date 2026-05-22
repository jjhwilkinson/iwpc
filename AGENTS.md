# AGENTS.md — `iwpc`

Top-level agent notes. Each sub-package under `src/iwpc/` has its own
`AGENTS.md` with the precise contracts, signatures, and gotchas — start with
the relevant one before editing. This file collects the cross-cutting rules
and a map into them.

## What this package is

`iwpc` is two related flows on top of PyTorch Lightning:

1. **Divergence estimation** — lower-bound estimates of f-divergences (KL,
   Jensen–Shannon, …) between two empirical distributions, implementing
   [arXiv:2405.06397](https://arxiv.org/abs/2405.06397). Entry point
   `iwpc.divergences.calculate_divergence`; iterative variant
   `iwpc.divergences.run_reweight_loop`.
2. **Distribution learning** (`iwpc.learn_dist`) — trainable conditional
   kernels and sampleable base distributions, fitted either against
   unlabelled data or by minimising a `DifferentiableFDivergence` against a
   target. Independent of `calculate_divergence` and uses its own batch
   conventions.

Both flows share the network builder (`iwpc.models.basic_model_factory`), the
feature/output `Encoding`s, the `GroupAction` symmetry wrappers, and the
`DifferentiableFDivergence` math layer.

## Sub-package map

| Path | AGENTS.md scope |
|---|---|
| [`divergences/`](src/iwpc/divergences/AGENTS.md) | `DifferentiableFDivergence` math layer, `FDivergenceEstimator` hierarchy, `calculate_divergence`, `run_reweight_loop`. |
| [`learn_dist/`](src/iwpc/learn_dist/AGENTS.md) | How kernels, base distributions, and trainers compose. Different batch layout from the divergence flow. |
| [`learn_dist/kernels/`](src/iwpc/learn_dist/kernels/AGENTS.md) | `TrainableKernelBase`, structural combinators (`Concatenated`, `Branching`, `Mixture`, `Permutation`, …), finite kernels, unlabelled trainers. |
| [`learn_dist/base_distributions/`](src/iwpc/learn_dist/base_distributions/AGENTS.md) | `SamplableBaseModel` contract; analytic + histogram priors. |
| [`learn_dist/fdivergence_minimization/`](src/iwpc/learn_dist/fdivergence_minimization/AGENTS.md) | `FDivergenceMinimizingKernelTrainer` and the `fdivergence_gradient_surrogate_loss`. |
| [`encodings/`](src/iwpc/encodings/AGENTS.md) | `Encoding` subclass contract, `&` concatenation, per-encoding gotchas. |
| [`symmetries/`](src/iwpc/symmetries/AGENTS.md) | `GroupAction`, Haar averaging, `.symmetrize` / `.complement`, `*` / `&` composition. |
| [`models/`](src/iwpc/models/AGENTS.md) | `basic_model_factory` composition order; `RunningNormLayer` / `RunningDeNormLayer`. |
| [`utility_modules/`](src/iwpc/utility_modules/AGENTS.md) | `IndependentSumModule`. |
| [`data_modules/`](src/iwpc/data_modules/AGENTS.md) | Batch contract, train/val split mechanics, `ds_info.yml`, `PandasDirDataModule` mutation API. |
| [`datasets/`](src/iwpc/datasets/AGENTS.md) | `PandasDataset`, `PandasFileListDataset`, `StructuredDataSpec`. |
| [`accumulators/`](src/iwpc/accumulators/AGENTS.md) | Post-hoc divergence + binned attribution with proper stderrs. |
| [`visualise/`](src/iwpc/visualise/AGENTS.md) | 1D/2D sweep plotters (matplotlib + Bokeh). |
| [`scalars/`](src/iwpc/scalars/AGENTS.md) | Label / LaTeX-label / bin-array value objects. |
| [`metrics/`](src/iwpc/metrics/AGENTS.md) | `WeightedMeanMetric`, `StatMetric` — produce `val_Df` / `val_Df_err`. |

## Global conventions (divergence flow)

- **Batch layout:** `(features, labels, weights)` everywhere. Use
  `iwpc.utils.split_by_mask` to fan out by label rather than hand-rolling
  boolean indexing.
- **Label convention:** `0 = p`, `1 = q`. Same in batches, accumulators, and
  the reweight loop. The docstring in
  `divergences/base.py:calculate_naive_rep_summands_given_log_by_label` reads
  backwards — implementation matches the rest of the repo.
- **Validation metric:** always `val_Df` (higher is better — it is a lower
  bound). `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau` all
  monitor it with `mode="max"`. Subclasses MUST set `self.val_Df` and
  `self.val_Df_err` in `_configure_metrics`; `val_Df_sig = val_Df / val_Df_err`
  is computed for free.
- **`log(p/q)` clipping:** clipped to `[-14, 14]` before exponentiation in
  both training loss and validation accumulation. New estimators / divergences
  must remain finite on that interval.
- **numpy ↔ torch dispatch:** `DifferentiableFDivergence` exposes the same
  math through `_*_torch` and `_*_np` hooks; dispatch via
  `_np_or_torch(x, torch_fn, np_fn)` keyed on `isinstance(x, torch.Tensor)`.
  Never call the underscore methods directly; never branch on type yourself.
  Torch and numpy paths must be numerically equivalent — drift silently
  corrupts results because accumulators use the numpy branch.

`learn_dist` does **not** follow the `(features, labels, weights)` /
`0=p, 1=q` batch convention — each trainer there defines its own batch shape.
See [`learn_dist/AGENTS.md`](src/iwpc/learn_dist/AGENTS.md).

## Public API surface

Anything re-exported from `iwpc/__init__.py` or a sub-package `__init__.py` is
public. Treat these as stable; coordinate with the user before changing
signatures:

- `DifferentiableFDivergence`, `KLDivergence`, `JensenShannonDivergence`
- `FDivergenceEstimator`, `NaiveVariationalFDivergenceEstimator`,
  `GenericNaiveVariationalFDivergenceEstimator`, `AsymmetryEstimator`
- `calculate_divergence`, `DivergenceResult`, `run_reweight_loop`
- `Encoding` (and the concrete encodings)
- `GroupAction` (and concrete groups)
- `basic_model_factory`, `basic_model_factory_sum`
- The data modules (`BinaryPandasDataModule`, `BinaryNumpyDataModule`,
  `PandasDirDataModule`)
- The kernel base classes and the `learn_dist` trainers

## Repo-wide workflow

- Branch off `main` with `fix/`, `feat/`, `chore/`, or `docs/`. Never commit
  directly to `main`. One logical change per branch.
- Commits: imperative mood, ≤72-char subject, body explains *why*.
- PR body: one-sentence summary, "What changed" bullets, "How to verify"
  (cite which `examples/` script you ran — there is no test suite).
- Don't bundle unrelated changes (formatting, deps, drive-by refactors).
- Pause and ask before: adding a dependency (update `pyproject.toml` **and**
  `requirements.txt` together), changing public API surface, or deleting
  more than ~50 lines.
- Don't stage anything under `dist/`, `lightning_logs/`, or `*.pkl` dataset
  files. Bumping the `pyproject.toml` version and rebuilding is the correct
  path for `dist/`.
- Match the surrounding file's style. Public classes and methods use
  numpy-style docstrings; don't reformat untouched code.

## Smoke tests

`tests/` is empty. The `examples/` scripts double as integration tests:

- `examples/parity_example.py` — reproduces paper plots; canonical smoke
  test for the divergence flow.
- `examples/example_reweight_loop.py` — canonical reweight-loop walkthrough;
  exercises `PandasDirDataModule` mutation API.
- `examples/multidimensional_function_visualiser_example.py` — exercises the
  visualisers.

Set `DISABLE_IWPC_WELCOME=1` before importing `iwpc` to silence the banner
from `iwpc/__init__.py`. `calculate_divergence` writes per-run subdirs under
`<log_dir>/lightning_logs/` (default `log_dir=cwd`); these plus `*.pkl` and
any `*_reweighted/` directories are gitignored.

## Don'ts

- Don't bypass `_np_or_torch` by type-checking in subclasses.
- Don't call estimator `_calculate_batch_loss` /
  `_accumulate_validation_Df` /  `_configure_metrics` directly — they are
  hooks, the base class drives them.
- Don't manually log `val_Df` / `val_Df_err` in `_accumulate_validation_Df`;
  the base class does it on epoch end.
- Don't use `run_reweight_loop` with anything but `PandasDirDataModule` — it
  hard-couples to `.transform` / `.reweight` / `.copy` and the
  `ds_info['tags']` history.
- Don't mutate `PandasDirDataModule` files in place — go through
  `.transform` / `.reweight` so the tag history stays honest.
- Don't introduce per-instance state that diverges between numpy and torch
  paths of a `DifferentiableFDivergence`.