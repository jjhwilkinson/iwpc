# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`iwpc` is a research Python package that estimates lower bounds on f-divergences (KL, Jensen-Shannon, …) between two distributions p and q given samples from each. It implements the framework from https://arxiv.org/abs/2405.06397 on top of PyTorch Lightning, and the same machinery is reused for distribution learning, reweighting, and symmetry-aware modelling.

The package is published to PyPI as `iwpc`. Source lives in `src/iwpc/`; `tests/` is currently empty (no test runner is configured).

## Common commands

```bash
pip install -e .                  # editable install for development
python -m build                   # build wheel + sdist into dist/ (uses pyproject.toml)
tensorboard --logdir lightning_logs   # monitor a training run
```

When running examples or any code that imports `iwpc`, set `DISABLE_IWPC_WELCOME=1` to suppress the ASCII banner printed from `src/iwpc/__init__.py`.

`calculate_divergence` writes per-run subdirectories under `<log_dir>/lightning_logs/` (default `log_dir=cwd`). These plus `*.pkl` and `sample_dataset_example_reweighted/` are gitignored.

## Architecture

The whole package is organised around a single training contract: batches are tuples `(features, labels, weights)` where `labels==0` marks samples from p and `labels==1` marks samples from q. Almost every component (loss, accumulator, data module, reweighting) assumes this layout.

### Training entry point

`src/iwpc/calculate_divergence.py:calculate_divergence` is the canonical way to fit a divergence estimator. It builds a Lightning `Trainer` with `ModelCheckpoint(monitor="val_Df", mode="max")`, `EarlyStopping(monitor="val_Df")`, and `LearningRateMonitor`, runs `trainer.fit`, reloads the best checkpoint, validates, and returns a `DivergenceResult` (divergence, stderr, best module, trainer, checkpoint path). Anything tracked as `val_Df`/`val_Df_err` will be picked up automatically.

### Estimator hierarchy (`src/iwpc/modules/`)

- `FDivergenceEstimator` (`fdivergence_base.py`) — abstract `LightningModule`. Subclasses implement `_configure_metrics` (must set `val_Df` and `val_Df_err`), `_calculate_batch_loss`, and `_accumulate_validation_Df`. Provides Adam + `ReduceLROnPlateau(monitor="val_Df")` and the standard training/validation steps.
- `NaiveVariationalFDivergenceEstimator` (`naive.py`) implements the naive variational representation from the paper: clips `log(p/q)` to `[-14, 14]` for numerical stability, and tracks the two expectation summands with `WeightedMeanMetric` so `val_Df` is the difference of running means.
- `GenericNaiveVariationalFDivergenceEstimator` wires the above to a model produced by `models.utils.basic_model_factory`. Pass either an input dim or an `Encoding` instance as `input`.

### Divergences (`src/iwpc/divergences/`)

`DifferentiableFDivergence` exposes a generating function `f`, its Legendre conjugate `f_conj`, and `f_dash_given_log` (derivative parameterised by `log(x)` for stability). Each concrete divergence (`KLDivergence`, `JensenShannonDivergence`) implements both numpy and torch versions of these — the base class auto-dispatches via `_np_or_torch` based on input type. When adding a new divergence, implement all four `_*_torch` / `_*_np` methods; do not call them directly.

### Encodings (`src/iwpc/encodings/`)

`Encoding` is an `nn.Module` that transforms inputs into a representation the network can exploit (e.g. `ContinuousPeriodicEncoding` maps θ → (cos θ, sin θ); `AbsEncoding` enforces evenness). The `&` operator builds a `ConcatenatedEncoding` that applies sub-encodings to adjacent feature slices and concatenates the result, e.g. `TrivialEncoding(1) & ContinuousPeriodicEncoding()` for `(r, θ)`. `basic_model_factory` accepts an `Encoding` as its `input` and inserts it as the first layer.

### Data modules (`src/iwpc/data_modules/`)

- `PandasDataModule` / `BinaryPandasDataModule` (`pandas_data_module.py`) — wrap a single DataFrame (or a (p_df, q_df) pair, which auto-adds a `__label` column). 50/50 train/val split via `sklearn.train_test_split`.
- `PandasDirDataModule` (`pandas_directory_data_module.py`) — for datasets too large for memory. The directory must contain `file_0.pkl … file_{N-1}.pkl` plus a `ds_info.yml` listing `file_sizes`. Train/val split is by file, taking the first `ceil(N*split)` files for training, so the on-disk file ordering must already be unbiased. The `ds_info` dict also stores a `tags` history; use `.transform(...)`/`.reweight(...)` (which add tags) rather than mutating files in place. Set `use_in_memory_dataset=True` to concatenate everything into shared-memory tensors when it fits.
- `BinaryNumpyDataModule` (`numpy_data_module.py`) — analogous wrapper for numpy arrays.

### Reweight loop (`src/iwpc/reweight_loop.py`)

`run_reweight_loop` repeatedly calls `calculate_divergence`; whenever the resulting significance exceeds `min_sig`, it adds a new `p_over_q_{i}` column to the dataset, multiplies the weight column by `min(p/q, q/p)` (clipped at 1) to wipe out the learnt feature, and re-runs with a decayed learning rate. This produces a `PandasDirDataModule` with a chain of reweight columns; `calculate_total_divergence` reconstructs the cumulative divergence by taking the product of those columns. This loop only works with `PandasDirDataModule` because it relies on `.transform`/`.reweight`/`.copy` and on tags.

### Accumulators (`src/iwpc/accumulators/`)

`DfAccumulator` and subclasses estimate divergences from precomputed probability ratios with proper standard errors (`accumulated_df`, `accumulated_df_stderr`). `BinnedDfAccumulator` partitions by user-chosen variables to attribute divergence to specific features and produces the diagnostic plots in the README — currently 1D and 2D only.

### Symmetry-aware models (`src/iwpc/symmetries/`)

`GroupAction.symmetrize(model)` and `.complement(model)` wrap a model so its output is invariant under (or lives in the orthogonal complement of) a group action, by averaging the model over a batch of action elements drawn from the Haar measure. `basic_model_factory` accepts `symmetries=` and `complement_symmetries=` lists and applies these wrappers after construction.

### Visualisers (`src/iwpc/visualise/`)

`MultidimensionalFunctionVisualiser` and the `bokeh_*` variants render high-dimensional learned functions by sweeping selected dimensions; useful sanity checks for trained estimators.

### `learn_dist/`

A separate sub-package that uses the same divergence machinery for distribution learning and conditional ML — kernels, base distributions (gaussian, uniform, histogram, …), and an f-divergence-minimising training loop. Independent of the main divergence-estimation flow but built on the same `DifferentiableFDivergence` and Lightning trainer scaffolding.

## Conventions worth knowing

- Labels: **0 = p, 1 = q** consistently across batches, accumulators, and the reweight loop. `split_by_mask` (in `utils.py`) is used everywhere to fan out arrays by label.
- Validation metric is always `val_Df` (higher is better — it is a *lower bound*). Early stopping, checkpointing, and the LR scheduler all monitor it with `mode="max"`.
- Numerical stability: `log(p/q)` is clipped to `[-14, 14]` before exponentiation in the naive estimator and accumulators. New estimators should follow the same pattern.
- The codebase mixes numpy and torch deliberately — divergence functions exist in both. Use `_np_or_torch` rather than branching on type yourself.
- Local helper variables must either have names that make their meaning obvious (prefer longer descriptive names like `num_sample_outcomes` over `M`, `log_prob_table` over `table`) or be inlined at the use site. Single-letter names buried in error messages or shape descriptions are not acceptable.

## Workflow

- Never commit directly to `main`. Branch off it with a `fix/`, `feat/`, `chore/`, or `docs/` prefix and one logical change per branch.
- Commits: imperative mood, first line under 72 chars, body explains *why*. PR title mirrors the primary commit message; PR body needs a one-sentence summary, a "What changed" bullet list, and a "How to verify" section. Note in the PR body what you ran.
- Don't bundle unrelated changes (formatting, dep bumps, drive-by refactors) into a feature/fix PR — open a separate one.
- Pause and ask before: adding a new dependency (update `pyproject.toml` *and* `requirements.txt` together), changing public API surface (anything exported from `iwpc/` and re-exported via subpackage `__init__.py` files — `DifferentiableFDivergence`, `FDivergenceEstimator`, `Encoding`, the data modules, `calculate_divergence`, `run_reweight_loop`), or deleting more than ~50 lines.
- Don't touch built artefacts in `dist/` (the published wheels — bumping the version in `pyproject.toml` and rebuilding is the correct path), anything under `lightning_logs/`, or `*.pkl` dataset files. None should be staged.
- Match the surrounding file's style — this codebase uses numpy-style docstrings on every public class and method. New public surface should follow suit; don't reformat untouched code.
