# AGENTS — `iwpc.scalars`

- `Scalar(label, latex_label=None, bins=None)` — three-field value object. `latex_label` defaults to `label`. `bins` is an arbitrary `NDArray` (typically a `np.linspace`); callers assume it is sorted and equally spaced (`BinnedDfAccumulator` uses `bins[0]`/`bins[-1]`/length).
- `ScalarFunction(fn, label, latex_label=None, bins=None, color_palette=Viridis256)` — `Scalar` subclass that adds a callable `fn` and a bokeh `color_palette`. Instance call `__call__(*args, **kwargs)` returns `fn(*args, **kwargs)`; the input convention is "whatever the consumer passes" — usually a `DataFrame`, sometimes raw arrays.

## Subclass contract

- Anything subclassing `Scalar` should keep `label`, `latex_label`, and `bins` as public attributes (visualisers and accumulators read them directly).
- `ScalarFunction.fn` should accept the same input shape that the surrounding visualiser/accumulator passes to it (typically a `DataFrame` of model outputs, but `examples/multidimensional_function_visualiser_example.py` passes raw arrays — context-dependent).

## Cross-package consumers

- `src/iwpc/accumulators/binned_Df_accumulator.py` — `BinnedDfAccumulator` takes `scalars: ScalarFunction | List[ScalarFunction]` to define histogram axes (1D and 2D only).
- `src/iwpc/accumulators/histogram_accumulator.py` — `Scalar` used for axis labelling/bin specification.
- `src/iwpc/visualise/visualisable.py` — `Visualisable.get_output_scalars()` returns `List[ScalarFunction]`.
- `src/iwpc/visualise/bokeh_function_visualiser.py`, `bokeh_function_visualiser_2D.py`, `multidimensional_function_visualiser.py`, `multidimensional_function_visualiser_1D.py`, `multidimensional_function_visualiser_2D.py` — all take `output_scalars: List[ScalarFunction]`.
- `examples/parity_example.py`, `examples/example_reweight_loop.py`, `examples/multidimensional_function_visualiser_example.py` — instantiate `ScalarFunction`s for plotting.

## Gotchas

- `Scalar.__init__` does not store `bins` via `np.asarray`; passing a plain Python list will trip up downstream code that calls `.shape` or indexing arithmetic. Always pass an `ndarray`.
- `ScalarFunction` imports `bokeh.palettes.Viridis256` at module load, so importing this package pulls in bokeh. Be aware in environments without bokeh installed.
- `color_palette` is only consulted by the bokeh visualisers; the matplotlib path (`multidimensional_function_visualiser_*.py`) ignores it.
- `latex_label` falling back to `label` happens via `latex_label or label`, so an empty string `""` is treated as missing — pass `None` instead.
