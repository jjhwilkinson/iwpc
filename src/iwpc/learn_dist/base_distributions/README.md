# `iwpc.learn_dist.base_distributions`

## Purpose

Fixed (non-trainable) sampleable measures on `R^D`. These act as the
building blocks of the `learn_dist` distribution-learning machinery:
they are the noise / prior distributions fed into trainable kernels
(e.g. a Gaussian or uniform base whose samples are pushed through a
normalising-flow-style kernel), the proposal distributions in
`classifier_reweighting`, and the targets for closed-form `fit(...)`
baselines.

Every distribution implements a common contract via
`SamplableBaseModel`: a numpy `draw(num_samples)` and a numpy
`log_prob(x)` returning per-sample log densities. Each model carries a
`total_volume` scalar so unnormalised / sub-probability measures
compose cleanly under the `+` (mixture) operator.

## Layout

Abstract base:

- `sampleable_base_model.py` — `SamplableBaseModel` (ABC).

Concrete distributions (one file each):

- `cauchy_base_model.py` — `CauchyBaseModel` (1D, `loc`, `hwhm`).
- `exponential_base_model.py` — `ExponentialBaseModel` (1D, `loc`, `scale`).
- `histogram_base_model.py` — `HistogramBaseModel` (N-D, fit from samples).
- `multivaraite_normal_base_model.py` — `MultivariateNormalBaseModel` (filename has a typo; class spelled correctly).
- `uniform_base_model.py` — `UniformBaseModel` (1D, `low`, `high`).

Structural combinators (defined in `sampleable_base_model.py`):

- `ConcatenatedBaseModel` — independent product; build with `a & b`.
- `MixtureBaseModel` — weighted mixture; build with `a + b`; weights come from each sub-model's `total_volume`.
- Scalar `__rmul__` — `c * model` returns a copy with `total_volume *= c`.

## Usage

### Sample and evaluate a uniform distribution

```python
import numpy as np
from iwpc.learn_dist.base_distributions.uniform_base_model import UniformBaseModel

u = UniformBaseModel(low=-1.0, high=1.0)
x = u.draw(1000)                 # shape (1000, 1)
log_p = u.log_prob(x)            # shape (1000,)
```

### Build a joint base distribution by concatenation

For a 3D base with `(radius ~ Exponential, x, y ~ Normal)`:

```python
import numpy as np
from iwpc.learn_dist.base_distributions.exponential_base_model import ExponentialBaseModel
from iwpc.learn_dist.base_distributions.multivaraite_normal_base_model import (
    MultivariateNormalBaseModel,
)

r = ExponentialBaseModel(loc=0.0, scale=1.0)
xy = MultivariateNormalBaseModel(means=np.zeros(2), cov=np.eye(2))

base = r & xy                    # ConcatenatedBaseModel, dimension == 3
samples = base.draw(512)         # shape (512, 3)
log_p = base.log_prob(samples)   # shape (512,)
```

### Build a mixture with custom weights

```python
from iwpc.learn_dist.base_distributions.uniform_base_model import UniformBaseModel

left = 0.3 * UniformBaseModel(-2.0, 0.0)   # __rmul__ scales total_volume
right = 0.7 * UniformBaseModel(0.0, 2.0)
mix = left + right                          # MixtureBaseModel, weights (0.3, 0.7)
samples = mix.draw(1024)                    # shape (1024, 1)
```

`HistogramBaseModel.fit(x, bins, weights)` and
`MultivariateNormalBaseModel.fit(x, weights)` are useful when you want
the base distribution to roughly match a dataset before fitting a
trainable kernel on top of it.
