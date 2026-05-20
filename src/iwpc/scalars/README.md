# `iwpc.scalars`

Lightweight value-objects describing "a scalar quantity": its display label, optional LaTeX label, and a bin array for plotting/histogramming. `ScalarFunction` extends `Scalar` with a callable that derives the quantity from raw data. These are used by the visualisers and `BinnedDfAccumulator` to label axes and to define histogram binning consistently across plots.

## Layout

- `scalar.py` — `Scalar(label, latex_label=None, bins=None)`: pure value object.
- `scalar_function.py` — `ScalarFunction(fn, label, latex_label=None, bins=None, color_palette=Viridis256)`: `Scalar` plus a `fn(...)` derivation and a bokeh color palette. Instances are callable: `sf(df)` simply returns `fn(df)`.

## Usage

A plain `Scalar` is just metadata for a column that already exists on a DataFrame:

```python
import numpy as np
from iwpc.scalars.scalar import Scalar

theta = Scalar(label='theta', latex_label=r'$\theta$',
               bins=np.linspace(-np.pi, np.pi, 50))
```

A `ScalarFunction` adds a derivation rule. Both forms below appear in `examples/example_reweight_loop.py`:

```python
import numpy as np
from iwpc.scalars.scalar_function import ScalarFunction

angle_scalar  = ScalarFunction(lambda df: df['angles'],
                               'angle', latex_label=r'$\theta$',
                               bins=np.linspace(-np.pi, np.pi, 50))
radius_scalar = ScalarFunction(lambda df: (df['x']**2 + df['y']**2)**0.5,
                               'r', bins=np.linspace(0.5, 1.5, 50))

r_values = radius_scalar(some_df)   # callable; equivalent to fn(some_df)
```

The visualisers in `iwpc.visualise` accept lists of `ScalarFunction`s to drive their plot axes; `BinnedDfAccumulator` uses the `bins` attribute to define its partitioning.
