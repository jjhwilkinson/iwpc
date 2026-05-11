# IWPC

`iwpc` implements the methods of [arXiv:2405.06397](https://arxiv.org/abs/2405.06397) for estimating a lower bound on the f-divergence (Kullback–Leibler, Jensen–Shannon, ...) between two distributions p and q from samples drawn from each. The same machinery is reused for **dataset reweighting** and **distribution learning** (density estimation and conditional kernels).

Install with `pip install iwpc`. The package is organised around [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) — some familiarity with `LightningModule` / `LightningDataModule` / `Trainer` is recommended.

The plots in the [original paper](https://arxiv.org/abs/2405.06397) are reproduced by [`examples/parity_example.py`](examples/parity_example.py).

---

## Quick start

The canonical end-to-end flow — fit a critic, get a divergence lower bound, attribute it to feature axes with diagnostic plots — lives in [`iwpc.divergences`](src/iwpc/divergences/README.md):

```python
from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule
from iwpc.divergences import (
    calculate_divergence,
    GenericNaiveVariationalFDivergenceEstimator,
    KLDivergence,
)

dm = BinaryPandasDataModule(p_df=p_df, q_df=q_df, feature_cols=["x", "y"])
estimator = GenericNaiveVariationalFDivergenceEstimator(input=2, divergence=KLDivergence())
result = calculate_divergence(estimator, dm, trainer_kwargs={"max_epochs": 200})
print(result.divergence, "±", result.divergence_stderr)
```

See [`iwpc.divergences/README.md`](src/iwpc/divergences/README.md) for the full walkthrough — including using `Encoding`s to bake in priors, the `run_reweight_loop` driver for large or hard datasets, and the `BinnedDfAccumulator` diagnostic plots that explain *how* the network distinguishes p from q.

For the inverse problem — fitting a parametric distribution to a set of samples — see [`iwpc.learn_dist/README.md`](src/iwpc/learn_dist/README.md).

---

## What's in the package

Each sub-package has its own `README.md` (human-oriented) and `AGENTS.md` (terse notes for coding agents).

| Sub-package | What it does |
|---|---|
| [`divergences/`](src/iwpc/divergences/README.md) | The end-to-end divergence-estimation flow: math (`DifferentiableFDivergence`, `KLDivergence`, `JensenShannonDivergence`), the `FDivergenceEstimator` `LightningModule` hierarchy, `calculate_divergence`, and `run_reweight_loop`. |
| [`data_modules/`](src/iwpc/data_modules/README.md) | `LightningDataModule` adapters: `BinaryPandasDataModule` (in-memory DataFrames), `BinaryNumpyDataModule` (ndarrays), and `PandasDirDataModule` (sharded pickle directory, required for the reweight loop). |
| [`datasets/`](src/iwpc/datasets/README.md) | `torch.utils.data.Dataset` implementations backing the data modules. |
| [`encodings/`](src/iwpc/encodings/README.md) | Composable `nn.Module` feature rewrites (periodic, even/odd, log/exp, matrix-shape, ...). Used as the first/last layer of a network to bake in priors. |
| [`models/`](src/iwpc/models/README.md) | `basic_model_factory` — the canonical builder that wires an input `Encoding`, a normalised MLP body, an output `Encoding`, and optional symmetry wrappers into a single `nn.Sequential`. |
| [`symmetries/`](src/iwpc/symmetries/README.md) | Haar-averaging wrappers (`SymmetrizedModel`, `ComplementModel`) that make a network invariant under (or orthogonal to) a user-declared `GroupAction`. |
| [`utility_modules/`](src/iwpc/utility_modules/README.md) | Small `nn.Module` helpers — currently just `IndependentSumModule` for log-ratios that decompose additively over disjoint feature groups. |
| [`metrics/`](src/iwpc/metrics/README.md) | Lightweight `torchmetrics.Metric` accumulators (`WeightedMeanMetric`, `StatMetric`) used to produce `val_Df` and `val_Df_err`. |
| [`accumulators/`](src/iwpc/accumulators/README.md) | Post-hoc divergence estimation from precomputed log-ratios with proper standard errors. `BinnedDfAccumulator` produces the 1D / 2D diagnostic plots. |
| [`visualise/`](src/iwpc/visualise/README.md) | Interactive 1D / 2D sweep plotters (matplotlib + Bokeh backends) for sanity-checking a trained estimator's learned function. |
| [`scalars/`](src/iwpc/scalars/README.md) | Tiny value-objects bundling a label, LaTeX label, and bin array — used as plotting / binning metadata by the visualisers and the binned accumulator. |
| [`learn_dist/`](src/iwpc/learn_dist/README.md) | The distribution-learning side of `iwpc`: trainable conditional kernels `k(y\|x)`, sampleable base distributions, and an f-divergence-minimising training loop. Independent of `calculate_divergence` but reuses the divergence machinery. |
| [`learn_dist/kernels/`](src/iwpc/learn_dist/kernels/README.md) | ~25 trainable kernels (Gaussian, mixture, finite-support, branching, ...) + adversarial trainers. |
| [`learn_dist/base_distributions/`](src/iwpc/learn_dist/base_distributions/README.md) | Fixed sampleable distributions (uniform, Cauchy, exponential, multivariate normal, histogram) used as latent noise sources for kernels. |
| [`learn_dist/fdivergence_minimization/`](src/iwpc/learn_dist/fdivergence_minimization/README.md) | Adversarial trainer that fits a kernel to minimise a chosen `DifferentiableFDivergence` against a target. |

---

## Conventions

A handful of conventions are assumed everywhere in the divergence-estimation flow:

- **Batch contract:** `(features, labels, weights)`. **`labels == 0` marks samples from `p`, `labels == 1` marks samples from `q`.** The reweight loop, accumulators, and `split_by_mask` all assume this layout.
- **Validation metric:** `val_Df` (the divergence lower bound — higher is better) and `val_Df_err` (its standard error). Early stopping, `ModelCheckpoint`, and the LR scheduler all monitor `val_Df` with `mode="max"`.
- **Numerical stability:** `log(p/q)` is clipped to `[-14, 14]` before exponentiation in the naive estimator and in accumulators. Stay consistent with this in any new estimator or accumulator.
- **Banner suppression:** set `DISABLE_IWPC_WELCOME=1` to silence the ASCII banner printed from `iwpc/__init__.py`.

The `learn_dist/` sub-package uses different per-trainer batch contracts and logs `val_loss` / `val_divergence` instead of `val_Df` — see its README.

---

## Help and citation

For questions or suggestions please reach out to [Jeremy J. H. Wilkinson](mailto:jero.wilkinson@gmail.com).

If `iwpc` has been helpful in your research, please cite [arXiv:2405.06397](https://arxiv.org/abs/2405.06397).
