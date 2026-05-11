# `iwpc.modules`

## Purpose

This sub-package defines the **estimator hierarchy** at the heart of `iwpc`: the
trainable `LightningModule`s that learn a lower bound on an f-divergence
$D_f(p, q)$ from samples, following the variational construction in
[arXiv:2405.06397](https://arxiv.org/abs/2405.06397).

All estimators share a common subclass contract anchored on `FDivergenceEstimator`:

- Batches are tuples `(features, labels, weights)` with the package-wide
  convention **`labels == 0` for samples from `p`** and **`labels == 1` for
  samples from `q`**.
- The validation metric `val_Df` is the running estimate of the f-divergence
  lower bound, and `val_Df_err` is its standard error. **Higher `val_Df` is
  better** — checkpointing, early stopping, and the LR scheduler all monitor it
  with `mode="max"` (see `calculate_divergence`).
- A subclass implements three abstract methods:
  - `_configure_metrics` — must set `self.val_Df` and `self.val_Df_err`.
  - `_calculate_batch_loss(batch)` — returns the negative of the train estimate.
  - `_accumulate_validation_Df(batch)` — updates the validation accumulators.

Adam + `ReduceLROnPlateau(monitor="val_Df", mode="max")`, the training step,
the validation step, and the `train_loss` / `val_Df` / `val_Df_err` /
`val_Df_sig` logging are all provided by the base class for free.

## Layout

- `fdivergence_base.py` — `FDivergenceEstimator`, the abstract Lightning base
  class that fixes the training contract, the optimizer, and the LR scheduler.
- `naive.py` — `NaiveVariationalFDivergenceEstimator` implements the naive
  variational representation (Eq. 7 of the paper) using two
  `WeightedMeanMetric` accumulators for the p- and q-side expectations.
  `GenericNaiveVariationalFDivergenceEstimator` wires it to a network built by
  `models.utils.basic_model_factory` and accepts either an input shape or an
  `Encoding` as its `input` argument.
- `asymmetry_estimator.py` — `AsymmetryEstimator` estimates the f-divergence
  between `p` and its image under a `GroupAction`, by Haar-averaging the
  q-side summand so the network only has to learn the asymmetric component.
- `utility_modules/independent_sum_module.py` — `IndependentSumModule`,
  a generic `nn.Module` helper (not an estimator) that evaluates a list of
  sub-networks on configurable feature subsets and returns the (optionally
  encoded) mean of their outputs. Useful for building composite log-ratio
  models passed to an estimator.

## Usage

### Estimate a KL lower bound with a plain MLP

```python
from iwpc.calculate_divergence import calculate_divergence
from iwpc.data_modules.numpy_data_module import BinaryNumpyDataModule
from iwpc.divergences import KLDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator

module = GenericNaiveVariationalFDivergenceEstimator(
    input=9,                            # input shape; can also be an Encoding
    divergence=KLDivergence(),
    initial_learning_rate=1e-3,
    model_factory_kwargs={"hidden_layer_sizes": (128, 64, 64, 64, 64)},
)
data_module = BinaryNumpyDataModule(p_samples, q_samples,
                                    dataloader_kwargs={"batch_size": 1024})
result = calculate_divergence(module, data_module, patience=10,
                              trainer_kwargs={"max_epochs": 50})
print(result.divergence, "+/-", result.divergence_stderr)
```

`calculate_divergence` selects the checkpoint with the highest `val_Df` and
returns a `DivergenceResult` carrying the trained module.

### Plug in an `Encoding` for the input layer

```python
from iwpc.encodings import TrivialEncoding, ContinuousPeriodicEncoding
from iwpc.divergences import JensenShannonDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator

# (r, theta) -> (r, cos theta, sin theta)
encoding = TrivialEncoding(1) & ContinuousPeriodicEncoding()
module = GenericNaiveVariationalFDivergenceEstimator(
    input=encoding,
    divergence=JensenShannonDivergence(),
)
```

When `input` is an `Encoding`, `basic_model_factory` inserts it as the first
layer of the model and infers the network's input dimensionality from it.

### Measure the asymmetry of `p` under a group action

```python
from iwpc.modules.asymmetry_estimator import AsymmetryEstimator
from iwpc.models.utils import basic_model_factory
from iwpc.divergences import KLDivergence

model = basic_model_factory(input=3, output=1)
module = AsymmetryEstimator(
    group=my_group_action,              # an iwpc.symmetries.GroupAction
    model=model,
    divergence=KLDivergence(),
)
# Feed a single-distribution data module — labels are ignored by this estimator.
```

The q-side summand is averaged over a Haar sample of `my_group_action` at every
evaluation, so the learned scalar field is forced to capture only the
asymmetric component of `log(p/q)`.

## Writing your own estimator

Subclass `FDivergenceEstimator`, implement the three abstract methods above,
ensure that `_configure_metrics` assigns `self.val_Df` and `self.val_Df_err`
(typically derived from `WeightedMeanMetric` accumulators in
`iwpc.metrics`), and respect the labels-0=p / labels-1=q contract. Anything
logged as `val_Df` will be picked up automatically by `calculate_divergence`.
