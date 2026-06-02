# IWPC

A PyTorch Lightning framework for modelling experiments — composing trainable conditional kernels (detector responses, smearing, instrument models), sampleable base distributions, physical symmetries, and feature encodings — plus the original divergence-estimation toolkit for quantifying differences between modelled and observed distributions.

Install with `pip install iwpc`. Some familiarity with [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) is recommended.

Set `DISABLE_IWPC_WELCOME=1` to silence the ASCII banner printed from `iwpc/__init__.py`.

---

## Packages

Each sub-package has its own `README.md` (human-oriented) and `AGENTS.md` (terse notes for coding agents).

| Sub-package | Useful for                                                                                                                                                                                                                                                                                       |
|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [`learn_dist/`](src/iwpc/learn_dist/README.md) | Learning conditional kernels `k(y\|x)`, including from unlabelled data via a co-trained density-ratio estimator. The largest sub-package; see its children below.                                                                                                                                |
| [`learn_dist/kernels/`](src/iwpc/learn_dist/kernels/README.md) | Parametric trainable conditional distributions `k(y\|x)` — detector responses, smearing kernels, mixture/branching models, finite-support distributions, and the trainers that fit them to unlabelled data via a co-trained density-ratio estimator.                                             |
| [`learn_dist/base_distributions/`](src/iwpc/learn_dist/base_distributions/README.md) | Fixed sampleable measures (uniform, Cauchy, exponential, multivariate normal, histogram) used as priors over true variables or as latent noise feeding a kernel.                                                                                                                                 |
| [`learn_dist/fdivergence_minimization/`](src/iwpc/learn_dist/fdivergence_minimization/README.md) | Trainer that fits a kernel by minimising a chosen f-divergence against a target, using a co-trained `log_p_over_q_model` to estimate the density ratio.                                                                                                                                                    |
| [`divergences/`](src/iwpc/divergences/README.md) | Lower-bound estimation of f-divergences (KL, Jensen–Shannon, ...) between two empirical distributions: the math, the `FDivergenceEstimator` LightningModule hierarchy, `calculate_divergence`, and the iterative reweight loop. Implements [arXiv:2405.06397](https://arxiv.org/abs/2405.06397). |
| [`encodings/`](src/iwpc/encodings/README.md) | Composable feature rewrites that bake physical priors into a network at the input or output: periodicity, evenness/oddness, log/exp, matrix structure, simplex / unit-sphere constraints, masking.                                                                                               |
| [`symmetries/`](src/iwpc/symmetries/README.md) | Two-layer group-action library: vector-space `GroupAction` / `GroupActionElement` primitives on `R^dim`, plus a thin `SeparableGroupAction` / `SeparableGroupActionElement` function-space wrapper. Haar-averaging makes a network invariant under (or orthogonal to) a separable action — rotations, parity, sign-flips, finite groups, products thereof.                                                                                                                                 |
| [`models/`](src/iwpc/models/README.md) | `basic_model_factory` — composes an input `Encoding`, an auto-normalised MLP body, an output `Encoding`, and optional symmetry wrappers into a single `nn.Sequential`. The canonical network builder used by every estimator and kernel.                                                         |
| [`utility_modules/`](src/iwpc/utility_modules/README.md) | Small `nn.Module` helpers — currently just `IndependentSumModule` for functions that decompose additively over disjoint feature groups.                                                                                                                                                          |
| [`data_modules/`](src/iwpc/data_modules/README.md) | Lightning `DataModule` adapters: in-memory pandas (`BinaryPandasDataModule`), in-memory numpy (`BinaryNumpyDataModule`), and on-disk sharded pickle directories (`PandasDirDataModule`, required by `run_reweight_loop`).                                                                        |
| [`datasets/`](src/iwpc/datasets/README.md) | `torch.utils.data.Dataset` implementations backing the data modules.                                                                                                                                                                                                                             |
| [`accumulators/`](src/iwpc/accumulators/README.md) | Post-hoc analysis of a trained estimator: divergence estimates with proper standard errors, and per-bin attribution of divergence to chosen feature axes via `BinnedDfAccumulator` (the source of the canonical 1D / 2D diagnostic plots).                                                       |
| [`visualise/`](src/iwpc/visualise/README.md) | Interactive 1D / 2D sweep plotters (matplotlib for scripts, Bokeh for shareable HTML) for sanity-checking trained functions — divergence estimators, kernels, or any `Callable[[NDArray], NDArray]`.                                                                                             |
| [`scalars/`](src/iwpc/scalars/README.md) | Value-objects bundling a label, LaTeX label, and bin array. Plotting / binning metadata for the visualisers and `BinnedDfAccumulator`.                                                                                                                                                           |
| [`metrics/`](src/iwpc/metrics/README.md) | `torchmetrics.Metric` accumulators (`WeightedMeanMetric`, `StatMetric`) — produce `val_Df` and `val_Df_err` for the divergence flow.                                                                                                                                                             |

Examples live in [`examples/`](examples/) — `parity_example.py` reproduces the paper plots; `example_reweight_loop.py` is the canonical reweight-loop walkthrough; `multidimensional_function_visualiser_example.py` demos the visualisers.
Unfortunately these are all out of date and likely need to be updated.

---

## Help and citation

For questions or suggestions please reach out to [Jeremy J. H. Wilkinson](mailto:jero.wilkinson@gmail.com).

If the divergence-estimation flow has been helpful in your research, please cite [arXiv:2405.06397](https://arxiv.org/abs/2405.06397).
