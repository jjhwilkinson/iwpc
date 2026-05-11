# `iwpc.divergences`

## Purpose

This sub-package defines the f-divergences that `iwpc` knows how to estimate. Each divergence is described by its generating function `f`, its Legendre conjugate `f_conj`, and the derivative `f_dash` (parameterised by `log(x)` for numerical stability). The naive variational representation used throughout the package — `D_f(p, q) >= E_p[f'(p/q)] - E_q[f*(f'(p/q))]` — is built from these three pieces, so adding a new divergence to `iwpc` reduces to subclassing `DifferentiableFDivergence` and implementing those functions in both numpy and pytorch. The base class auto-dispatches on input type so callers (estimators in `iwpc.modules`, accumulators in `iwpc.accumulators`) never have to branch.

## Layout

- `base.py` — `DifferentiableFDivergence`, the abstract interface. Declares the six `_*_torch` / `_*_np` hooks subclasses must implement, the three public `f` / `f_conj` / `f_dash_given_log` dispatchers, and the helpers (`calculate_naive_p_summands_given_log`, `calculate_naive_q_summands_given_log`, `calculate_naive_rep_summands_given_log_by_label`, `naive_estimate_given_log`) that the training loop and accumulators call.
- `kl_divergence.py` — `KLDivergence`. `f(x) = x log x`, `f*(x) = exp(x - 1)`, `f'(log x) = 1 + log x`.
- `jensen_shannon_divergence.py` — `JensenShannonDivergence`. `f(x) = 0.5 (x log x - (x+1) log((x+1)/2))`, with a `logaddexp`-based stable form for `f'`.
- `__init__.py` — re-exports `DifferentiableFDivergence`, `KLDivergence`, `JensenShannonDivergence`.

## Usage

### Evaluating a divergence's generating function

```python
import torch
import numpy as np
from iwpc.divergences import KLDivergence

kl = KLDivergence()

# Torch input -> torch output (auto-dispatch).
x_t = torch.tensor([0.5, 1.0, 2.0])
print(kl.f(x_t))                         # x * log(x)
print(kl.f_conj(x_t))                    # exp(x - 1)
print(kl.f_dash_given_log(torch.log(x_t)))  # 1 + log(x)

# Numpy input -> numpy output, same code path.
x_n = np.array([0.5, 1.0, 2.0])
print(kl.f(x_n))
```

`f`, `f_conj` and `f_dash_given_log` are the only methods you should call directly; the underscore-prefixed `_f_torch` / `_f_np` etc. are implementation hooks.

### Plugging a divergence into an estimator

`GenericNaiveVariationalFDivergenceEstimator` takes a `DifferentiableFDivergence` instance and uses its naive representation as the training objective:

```python
from iwpc.divergences import JensenShannonDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator
from iwpc.calculate_divergence import calculate_divergence
from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule

dm = BinaryPandasDataModule(p_df=p_df, q_df=q_df, feature_cols=["x"])
module = GenericNaiveVariationalFDivergenceEstimator(
    input=1,
    divergence=JensenShannonDivergence(),
)
result = calculate_divergence(module, dm)
print(result.divergence, result.divergence_stderr)
```

The estimator's network outputs `log(p/q)`, which is fed straight into
`divergence.calculate_naive_rep_summands_given_log_by_label(log_p_over_q, label)`
to form the two expectation summands the loss averages.

### Computing the naive estimator by hand from precomputed `log(p/q)`

```python
import torch
from iwpc.divergences import KLDivergence

kl = KLDivergence()

# label: 0 = sample from p, 1 = sample from q.
log_p_over_q = torch.tensor([1.2, -0.4, 0.0, 0.8])
label        = torch.tensor([0,    1,    0,   1])
weights      = torch.ones(4)

Df_hat = kl.naive_estimate_given_log(log_p_over_q, label, weights)
print(float(Df_hat))  # a lower bound on D_KL(p || q)
```

For analysis with proper error bars, prefer one of the accumulators in
`iwpc.accumulators` (e.g. `DfAccumulator`, `BinnedDfAccumulator`) which consume
the same `DifferentiableFDivergence` interface.
