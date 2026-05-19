# `iwpc.encodings`

## Purpose

An `Encoding` is a small `nn.Module` that rewrites raw input features into a
representation a downstream network can exploit, *without* having to store the
transformed values in the dataset itself. Typical uses:

- enforce periodicity: feed `(cos theta, sin theta)` rather than `theta`;
- enforce evenness in a feature: feed `|x|` instead of `x`;
- reshape a flat vector into a matrix for structural layers;
- mask features out entirely.

Encodings plug in as the first layer of a model via
`iwpc.models.utils.basic_model_factory(input=my_encoding, ...)`, which reads
`encoding.output_shape` to size the rest of the network. Output-side encodings
are also supported (constrained to vector-input encodings).

## Layout

Every encoding subclasses `Encoding` (in `encoding_base.py`) and exposes
`input_shape`, `output_shape`, `forward(x) -> Tensor`. Inputs/outputs are tensors
of shape `(..., *input_shape)` / `(..., *output_shape)`; the leading batch dims
are arbitrary.

Identity / structural:

- `TrivialEncoding(d)`: pass-through; `(d,) -> (d,)`. Used as a placeholder
  alongside other encodings in a `ConcatenatedEncoding`.
- `NopeEncoding(d)`: consumes `d` input features and emits zero outputs. Useful
  for masking.
- `MatrixEncoding(d1, d2=d1)`: reshapes `(d1*d2,) -> (d1, d2)`.
- `AntiSymmetricMatrixEncoding(d)`: reshapes `(d*d,) -> (d, d)` and
  antisymmetrises via `0.5 * (M - M^T)`.
- `ConcatenatedEncoding([e1, ..., eN])`: applies each `ei` to its adjacent slice
  of an input feature vector and concatenates the outputs. Built implicitly by
  the `&` operator. Sub-encodings must each consume and produce 1D vectors.

Periodic / angular:

- `ContinuousPeriodicEncoding(range_=(-pi, pi))`: `(1,) -> (2,)`, maps a scalar
  to `(cos, sin)` so any continuous function on the encoded pair is continuous
  and periodic in the original variable, including across the boundary.
- `PeriodicEncoding(range_)`: `(1,) -> (1,)`, wraps the input into `range_` via
  modulo. Enforces periodicity *inside* the range but allows discontinuity at
  the boundary.
- `SphericalUnitVectorEncoding()`: `(2,) -> (3,)`, takes `(polar, azimuth)` and
  returns the corresponding 3D unit vector.

Symmetry-enforcing on a scalar feature:

- `AbsEncoding(d)`: `|x|`, enforces evenness `f(x) = f(-x)`.
- `SignEncoding(d)`: `sign(x)`.

Non-linear remaps:

- `LogEncoding(d, base=-1)`: natural log by default; `base > 0` selects a base.
  Inputs must be positive.
- `ExponentialEncoding(d)`: `exp(x)`.
- `ReciprocalEncoding(d)`: `1/x`.
- `LogSoftmaxEncoding(num_classes)`: log-softmax over the last dimension;
  intended for encoding a discrete probability distribution from logits.

## Usage

### Mixed `(r, theta)` input

Concatenate a pass-through on the radius with a periodic encoding on the angle:

```python
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding

input_encoding = TrivialEncoding(1) & ContinuousPeriodicEncoding()
# Maps (r, theta) -> (r, cos(theta), sin(theta)).
# input_shape == [2], output_shape == [3].
```

### Plugging into `basic_model_factory`

`basic_model_factory` accepts an `Encoding` as its `input` argument and inserts
it as the first layer of the sequential model. The downstream network is sized
from `encoding.output_shape`:

```python
from iwpc.models.utils import basic_model_factory

model = basic_model_factory(
    input=input_encoding,             # (r, theta) -> 3 features
    output=1,                          # scalar critic
    hidden_layer_sizes=(64, 64, 64),
)
```

This is the recommended path for handing an encoded representation to a
`GenericNaiveVariationalFDivergenceEstimator` (see
`iwpc.modules.naive`).

### Enforcing an even function and masking a nuisance feature

```python
from iwpc.encodings.abs_encoding import AbsEncoding
from iwpc.encodings.nope_encoding import NopeEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding

# Three features: signed coordinate u (force evenness), nuisance n (drop),
# and a passthrough v.
enc = AbsEncoding(1) & NopeEncoding(1) & TrivialEncoding(1)
# input_shape == [3], output_shape == [2]; the second input column is ignored.
```

Any continuous function the network learns on top of this encoding is
automatically even in `u` and independent of `n`.
