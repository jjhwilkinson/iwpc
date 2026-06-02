# `iwpc.models` — agent notes

## `basic_model_factory` (utils.py)

Full signature:

```python
basic_model_factory(
    input: Union[Encoding, Shape],
    output: Union[Encoding, Shape] = 1,
    hidden_layer_sizes: Iterable[int] = (128, 64, 64, 64, 64),
    dropout: float = 0.,
    batch_norm: bool = False,
    initial_layers: Optional[Iterable[nn.Module]] = None,
    final_layers: Optional[Iterable[nn.Module]] = None,
    symmetries: Optional[Union[SeparableGroupAction, Iterable[SeparableGroupAction]]] = None,
    complement_symmetries: Optional[Union[SeparableGroupAction, Iterable[SeparableGroupAction]]] = None,
    activation: Callable = LeakyReLU,
    debug_name: str | None = None,
) -> Sequential
```

Composition order (top to bottom of the resulting `nn.Sequential`):

1. `initial_layers` (with the input `Encoding` prepended if supplied; must have
   `is_vector_output=True`).
2. `Flatten()`.
3. `RunningNormLayer(input_size)` — auto-normalises flattened input.
4. `len(hidden_layer_sizes)` repetitions of `make_layer_group`:
   optional `Dropout` -> `Linear` -> `activation()` -> optional `BatchNorm1d`.
5. Optional `LambdaLayer(debug_print)` if `debug_name` is set.
6. Final `Linear(hidden[-1], prod(output_shape))`.
7. `LambdaLayer` reshape back to `output_shape`.
8. `final_layers` (with the output `Encoding` appended if supplied; must have
   `is_vector_input=True`).

After the `Sequential` is built, `symmetries` and `complement_symmetries` are
coerced via `_coerce_group_action` (single `SeparableGroupAction`, or iterable
folded with `*`) and the model is wrapped by `group.symmetrize(model)` then
`group.complement(model)` respectively. Composition operators `*` (joint
action on same space) and `&` (direct product on disjoint dim slices) are how
multi-symmetry models are described declaratively. Both operators are defined
on `SeparableGroupAction` and delegate per-side to the underlying
vector-space `GroupAction`s.

`basic_model_factory_sum(specs, output, **common_spec)` builds one
`basic_model_factory` per spec dict (overlaying `common_spec`), forces each
sub-model's `output` to `output.input_shape`, and wraps them in an
`IndependentSumModule` whose final output encoding is `output` (an int becomes
`TrivialEncoding(int)`).

## `layers.py`

- `LambdaLayer(fn)` — wraps any callable so it lives inside `nn.Sequential`.
  Used internally for `debug_print` and the output reshape.
- `RunningNormLayer(input_shape, max_samples=1_000_000)` — registers buffers
  `sum_`, `sq_sum_`, `N_`. Updates running mean/std only when `self.training`
  and `N_ < max_samples`; once frozen prints the final shift/scale once.
  Always applies `(x - shift) / scale` in forward (train and eval). Avoids
  divide-by-zero by falling back to `scale=1` when variance is non-positive.
  This is what makes the factory's models robust to un-pre-processed inputs.
- `RunningDeNormLayer(input_shape, one_epoch_only=False)` — inverse of the
  above: `x * scale + shift`, with running stats accumulated on the *post*
  values. Intended as the last layer so the preceding network trains against a
  normalised target. `one_epoch_only=True` freezes stats after the first
  training epoch (tracked via `prev_training`/`current_epoch`).
- `ConstantScaleLayer(shift=None, scale=None)` — fixed affine layer; either
  arg may be `None` (stored as NaN, skipped at forward time) or 0D/1D
  array-like.

## Cross-package consumers

- `iwpc.modules.naive.GenericNaiveVariationalFDivergenceEstimator` passes its
  `input` straight into `basic_model_factory` — primary consumer in the
  divergence-estimation flow.
- `iwpc.symmetries.separable_group_action.SeparableGroupAction.symmetrize` /
  `.complement` are invoked here; any new symmetry types should keep `*`/`&`
  and the `symmetrize`/`complement` API stable. Note: `symmetrize` /
  `complement` are only defined on the function-space `SeparableGroupAction`,
  not on the bare vector-space `GroupAction`.
- `iwpc.modules.utility_modules.independent_sum_module.IndependentSumModule`
  backs `basic_model_factory_sum`.
- `iwpc.learn_dist` reuses these models (and `RunningDeNormLayer` in
  particular) for distribution-learning networks where the target is
  un-normalised.
- Examples (`examples/parity_example.py`, `example_reweight_loop.py`) call
  `basic_model_factory` directly and double as integration tests.

## Editing rules

- Public surface: `basic_model_factory`, `basic_model_factory_sum`, all four
  layers. Don't change the signature without coordinating with the modules
  above. Keep numpy-style docstrings on any new public function.
- Don't move the `Flatten` + `RunningNormLayer` pair — downstream relies on
  inputs being auto-normalised.
