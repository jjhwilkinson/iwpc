# AGENTS.md — `iwpc.visualise`

Dense notes for agents touching this package.

## Role

Interactive plotters that sweep 1 or 2 input dimensions of a high-D function
and hold the rest fixed via sliders. Two backends, same mental model:
matplotlib (`MultidimensionalFunctionVisualiser{1D,2D}`) and bokeh
(`BokehFunctionVisualiser{1D,2D}`).

## Sweep mechanics

- Input is `x: NDArray` of shape `(N, k)`; output is `(N, ...)`.
- 1D: pick one input index `i`; the swept axis spans
  `np.linspace(scalar.bins[0], scalar.bins[-1], num_plot_points)`; all other
  columns are filled with the corresponding slider values, then `fn(input)`.
- 2D: same with a `meshgrid` over two indices `(i, j)`; reshape back to
  `(num_plot_points, num_plot_points, ...)`.
- `center_point` (defaults to bin midpoint per scalar) sets initial slider
  positions; the reset button restores it.
- Output is post-processed by the selected `ScalarFunction` (e.g. `exp` to get
  `p/q` from `log p/q`). Axis ranges come from `output_scalar.bins` if set,
  otherwise auto-fit to finite outputs ±10%.

## `Visualisable` contract

Implement these to plug into `BokehFunctionVisualiser.visualise(obj)`:

- `get_input_scalars() -> List[Scalar]` — order matches feature columns.
- `get_output_scalars() -> List[ScalarFunction]` — derived plot quantities.
- `evaluate_for_visualiser(x: NDArray) -> Any` — pure callable, batched.
- `center_point` (optional override).

`BokehFunctionVisualiser.visualise` only sets defaults via `kwargs.setdefault`,
so callers can still override any of these. Matplotlib backend does NOT
consume `Visualisable` — call its constructor directly.

## 1D vs 2D

- 1D: line plot, one x-axis radio (matplotlib) / Select (bokeh).
- 2D matplotlib: `imshow` heatmap, x/y radios, cmap configurable.
- 2D bokeh: heatmap with cross-section line profiles on right/below; click in
  plot pins crosshair, click in axis label region opens the embedded
  `BokehFunctionVisualiser1D` tab at that slice (see
  `handle_main_figure_click`/`configure_1d_panel`).

## matplotlib vs bokeh — when to pick which

- matplotlib: scripts and notebooks; quick local QC of a fresh checkpoint.
  **Must bind the instance to a name** or it gets GC'd and the GUI freezes
  (class docstring repeats this).
- bokeh: `bokeh serve` apps, sharing, hover tooltips, freeze switches,
  custom output range, batched evaluation (`batch_eval_size=32768`,
  `selected_input_parameter_resolution=256` by default), tqdm progress bar.
  Pick this once the estimator is interesting enough to share.

## Cross-package consumers

- Typical wrap: `fn = lambda x: module.model(torch.as_tensor(x).float()).detach().numpy()`
  on a `result.best_module` from `iwpc.calculate_divergence.calculate_divergence`.
- `learn_dist` modules and `iwpc.modules.*` estimators expose a `model`
  attribute that is the network to feed in.
- `iwpc.scalars.{Scalar, ScalarFunction}` are required types — every public
  entry point needs them.

## Adding a new visualiser

1. Pick a backend. For matplotlib subclass `MultidimensionalFunctionVisualiser`
   and implement `plot_dimension`, `setup_radio_buttons`, `setup_plot`,
   `update_plot`, `update_axes`. For bokeh subclass `BokehFunctionVisualiser`
   and implement `setup_figure`, `setup`, `setup_input_scalar_pickers`,
   `update_input_axes`, `update_output_axes`, `_update_output`.
2. Reuse `setup_settings_column` (bokeh) and `setup_sliders` (matplotlib)
   rather than rebuilding the sidebar.
3. Bokeh evaluation must batch via `self.batch_eval_size` and respect
   `reuse_previous_output` (skip recompute when only the output scalar
   changes).
4. Wire defaults from `Visualisable` through the existing `visualise`
   classmethod — do not add a new constructor signature.
5. Don't reformat untouched code; numpy-style docstrings on public methods.
