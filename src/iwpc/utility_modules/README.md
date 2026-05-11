# `iwpc.utility_modules`

Small `nn.Module` helpers that don't belong to any other sub-package. Today this is a one-class home.

## Layout

- `independent_sum_module.py` — `IndependentSumModule`. Evaluates a list of sub-networks on configurable feature-index subsets and returns the (optionally output-encoded) mean of their outputs.

## Usage

Use `IndependentSumModule` as the `model` of an `FDivergenceEstimator` when the log-ratio `log(p/q)` decomposes additively over disjoint feature groups (e.g. groups of features that you believe to be independent under both `p` and `q`).

```python
from torch.nn import Sequential, Linear, ReLU
from iwpc.utility_modules.independent_sum_module import IndependentSumModule

# Two disjoint subnets: features [0, 1] -> scalar, features [2, 3, 4] -> scalar.
sub_a = Sequential(Linear(2, 32), ReLU(), Linear(32, 1))
sub_b = Sequential(Linear(3, 32), ReLU(), Linear(32, 1))

model = IndependentSumModule(
    sub_models=[sub_a, sub_b],
    feature_indices=[[0, 1], [2, 3, 4]],
)
```

Use `iwpc.models.basic_model_factory_sum` for the common case where each sub-model is itself built by `basic_model_factory`.
