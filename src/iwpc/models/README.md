# `iwpc.models`

Network factory and small helper layers used to build the MLPs that parameterise
`iwpc`'s variational divergence estimators.

## Purpose

- `basic_model_factory` — the canonical way to build an MLP for an
  `FDivergenceEstimator`. Wires together an optional input `Encoding`, a stack
  of `Linear`/activation/(dropout)/(batch-norm) groups, optional output
  encoding, and optional `SeparableGroupAction` symmetry/complement wrappers.
- `basic_model_factory_sum` — convenience that sums independent sub-models
  (each with its own encodings/symmetries) into a single `IndependentSumModule`.
- `layers.py` — small reusable modules: `LambdaLayer`, `RunningNormLayer`,
  `RunningDeNormLayer`, `ConstantScaleLayer`.

## Layout

| File        | Contents                                                        |
|-------------|-----------------------------------------------------------------|
| `utils.py`  | `basic_model_factory`, `basic_model_factory_sum`, `make_layer_group`, `debug_print` |
| `layers.py` | `LambdaLayer`, `RunningNormLayer`, `RunningDeNormLayer`, `ConstantScaleLayer` |

Every model returned by `basic_model_factory` begins with a `Flatten` followed
by a `RunningNormLayer` sized to the flattened input. That layer learns its
shift/scale buffers from the first ~1M training samples it sees, so inputs are
auto-normalised without any explicit pre-processing step.

## Usage

### 1. Plain MLP from an input dimension

```python
from iwpc.models.utils import basic_model_factory

# 5-dim input, scalar output, default hidden sizes (128, 64, 64, 64, 64)
model = basic_model_factory(input=5)
```

### 2. With an input `Encoding`

The encoding is inserted as the first layer; the network input size is taken
from `encoding.output_shape`.

```python
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from iwpc.models.utils import basic_model_factory

# (r, theta) -> (r, cos theta, sin theta) -> MLP -> scalar
enc = TrivialEncoding(1) & ContinuousPeriodicEncoding()
model = basic_model_factory(
    input=enc,
    hidden_layer_sizes=(64, 64, 64),
    dropout=0.1,
    batch_norm=True,
)
```

### 3. With symmetries

`symmetries=` makes the network invariant under the supplied
`SeparableGroupAction`(s); `complement_symmetries=` constrains the output to
live in the symmetrised complement (i.e. is forced to vanish on the symmetric
part). An iterable is folded together via `*` (joint action on the same
space).

```python
from iwpc.models.utils import basic_model_factory
from iwpc.symmetries import SeparableGroupActionElement  # plus other concrete actions

parity = SeparableGroupActionElement.from_prod_add(input_prod=[-1.0], output_dim=1).to_group()

model = basic_model_factory(
    input=4,
    symmetries=[parity],                     # invariant under parity
    complement_symmetries=other_action,      # orthogonal to other_action
)
```

### 4. Direct use of helper layers

`RunningNormLayer` / `RunningDeNormLayer` track input/output statistics during
training and apply (de)normalisation during both training and validation; the
running buffers freeze after `max_samples`. `ConstantScaleLayer` applies a
fixed affine transform. `LambdaLayer` lifts any callable into `nn.Sequential`.

```python
from torch.nn import Linear, Sequential
from iwpc.models.layers import RunningNormLayer, LambdaLayer, ConstantScaleLayer

model = Sequential(
    RunningNormLayer(input_shape=3),
    Linear(3, 1),
    LambdaLayer(lambda x: x.squeeze(-1)),
    ConstantScaleLayer(shift=0.0, scale=2.0),
)
```

### 5. Summing independent sub-models

```python
from iwpc.models.utils import basic_model_factory_sum

model = basic_model_factory_sum(
    specs=[
        {"input": enc_a, "symmetries": [g1]},
        {"input": enc_b, "complement_symmetries": [g2]},
    ],
    output=1,
    hidden_layer_sizes=(64, 64, 64),
)
```

See `iwpc.modules.GenericNaiveVariationalFDivergenceEstimator` for the typical
consumer: it forwards an `input` (int or `Encoding`) directly into
`basic_model_factory`.
