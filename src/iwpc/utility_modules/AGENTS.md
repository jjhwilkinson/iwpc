# `iwpc.utility_modules` — agent notes

Catch-all for small `nn.Module` helpers that aren't estimators, encodings, or symmetries. Currently holds one class.

## `IndependentSumModule` (`independent_sum_module.py`)

- Wraps a list of sub-models plus a parallel list of feature-index subsets.
- `forward(x)`: indexes `x` per sub-model, evaluates each, optionally pushes the per-sub-model output through an `Encoding`, then **means** (not sums) the outputs.
- Consumers: `iwpc.models.basic_model_factory_sum` produces an `IndependentSumModule` from a list of spec dicts (`models/utils.py:204+`). Useful when the log-ratio decomposes additively over disjoint feature groups under both `p` and `q`.
- Constructor accepts `output_encoding=` (an `iwpc.encodings.Encoding`) applied to each sub-model's output before averaging.

## Cross-package

- `iwpc.encodings.Encoding` — optional output encoding.
- `iwpc.models.utils.basic_model_factory_sum` — the factory that builds these.

## Adding new utility modules

Drop a new file here. If the helper is broadly useful, also re-export from this package's `__init__.py` (currently empty). Don't bloat the package with single-use modules — push those into the consumer package instead.
