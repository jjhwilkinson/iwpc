# `iwpc.visualise`

Interactive plotters for high-dimensional functions — sanity-check trained
divergence estimators, learned reweighters, conditional distributions, or any
`Callable[[NDArray], NDArray]` whose input is a `(N, k)` feature matrix.

A visualiser **sweeps** one (1D) or two (2D) selected input dimensions over a
grid and **holds the remaining dimensions fixed** at slider-controlled values.
Switching the swept dimension and the held-fixed values is done interactively.

## Layout

### The `Visualisable` contract (`visualisable.py`)

`Visualisable` is the abstract interface a user-defined object can implement so
that `BokehFunctionVisualiser.visualise(obj)` can wrap it without any glue.
Subclasses must provide:

- `get_input_scalars() -> List[Scalar]` — names, LaTeX labels, and bin ranges
  for each input feature (used for axis labels and slider min/max).
- `get_output_scalars() -> List[ScalarFunction]` — derived scalars to plot from
  the raw function output (e.g. `log(p/q) -> p/q`).
- `evaluate_for_visualiser(x: NDArray) -> Any` — evaluates the function on a
  `(N, k)` batch.
- Optional `center_point` property — default slider positions.

`Scalar` and `ScalarFunction` live in `iwpc.scalars`.

### Matplotlib backend (`MultidimensionalFunctionVisualiser*`)

- `MultidimensionalFunctionVisualiser` (abstract) — GUI skeleton: sliders for
  every input scalar, a reset button, radio buttons to pick the swept axes and
  plotted output, and the shared `setup`/`update` lifecycle.
- `MultidimensionalFunctionVisualiser1D` — line plot, one swept input.
- `MultidimensionalFunctionVisualiser2D` — heat-map, two swept inputs (cmap
  configurable via `cmap=`).

These are constructed directly from a `fn`, `input_scalars`, and
`output_scalars`; they do **not** consume `Visualisable`.

### Bokeh backend (`BokehFunctionVisualiser*`)

- `BokehFunctionVisualiser` (abstract) — web-based UI: per-axis pickers,
  per-axis resolution spinners, sliders for fixed dimensions, freeze
  switches for input/output axes, custom-range output spinners, and a reset
  button. Function evaluation is batched (`batch_eval_size`, default 32768).
  The classmethod `BokehFunctionVisualiser.visualise(fn: Visualisable, **kw)`
  builds an instance from a `Visualisable`.
- `BokehFunctionVisualiser1D` — line/scatter plot of one swept input.
- `BokehFunctionVisualiser2D` — heat-map with x/y cross-section line profiles;
  clicking a point pins the cross-hair, clicking inside an axis label region
  jumps to an embedded 1D tab (`BokehFunctionVisualiser1D`) at that slice.

## Usage

### 1. Visualise a trained estimator's learned `log(p/q)` (matplotlib)

```python
import numpy as np
from matplotlib import pyplot as plt
from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.visualise.multidimensional_function_visualiser_1D import MultidimensionalFunctionVisualiser1D

# `result` came from calculate_divergence(...)
model = result.best_module.model
fn = lambda x: model(torch.as_tensor(x, dtype=torch.float32)).detach().numpy()

input_scalars  = [Scalar(f'x{i}', fr'$x_{{{i}}}$', bins=np.linspace(-3, 3, 100)) for i in range(k)]
output_scalars = [ScalarFunction(lambda y: y, 'log p/q', r'$\log(p/q)$')]

vis = MultidimensionalFunctionVisualiser1D(fn, input_scalars, output_scalars)
plt.show()  # IMPORTANT: keep `vis` bound — GC will freeze the figure otherwise
```

### 2. 2D sweep with the other dimensions held fixed (matplotlib)

```python
from iwpc.visualise.multidimensional_function_visualiser_2D import MultidimensionalFunctionVisualiser2D

vis = MultidimensionalFunctionVisualiser2D(
    fn, input_scalars, output_scalars,
    center_point=[0.0] * k,   # other dims pinned here; movable via sliders
    cmap='viridis',
)
plt.show()
```

The x-axis and y-axis radio buttons pick the two swept dimensions; the
remaining `k-2` dimensions stay at their current slider values.

### 3. Bokeh server with the `Visualisable` shortcut

```python
# server.py — run with `bokeh serve --show server.py`
from bokeh.io import curdoc
from iwpc.visualise.bokeh_function_visualiser_2D import BokehFunctionVisualiser2D

class MyEstimator(Visualisable):
    def get_input_scalars(self):  ...
    def get_output_scalars(self): ...
    def evaluate_for_visualiser(self, x):
        return self.model(torch.as_tensor(x).float()).detach().numpy()

vis = BokehFunctionVisualiser2D.visualise(MyEstimator())
curdoc().add_root(vis.root)
```

## Picking a backend

- **matplotlib (`MultidimensionalFunctionVisualiser*`)** — local, in-process,
  trivial in a script or notebook (`plt.show()`). Best during model
  development. Bind the visualiser to a name or matplotlib will GC it and the
  GUI will freeze (see warning in the class docstring).
- **bokeh (`BokehFunctionVisualiser*`)** — richer interaction (hover tooltips,
  custom output range, freeze-axes, 2D click-to-cross-section, batched
  evaluation, embedded 1D tab in the 2D viewer), and hostable via
  `bokeh serve` so collaborators can poke at the model without running Python
  locally. Use this once you want to share a trained estimator.

## See also

- `examples/multidimensional_function_visualiser_example.py` — matplotlib 1D
  and 2D viewers on a 3D `sin(r)/r` test function.
- `iwpc.scalars` — the `Scalar` / `ScalarFunction` types consumed here.
