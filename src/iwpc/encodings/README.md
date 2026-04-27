# iwpc.encodings

Composable input-transformation layers (`nn.Module` subclasses) that re-express
features in a form a downstream network can exploit — periodic angles as
`(cos, sin)` pairs, even-symmetry inputs as absolute values, log/exp/reciprocal
remappings, masking-out of dimensions, and reshaping vectors into matrices. They
are normally placed at the start of a sequential model so the dataset itself
does not need to carry the transformed columns.

## What this provides

- `Encoding` — abstract base class. Subclass and implement `_encode(x)`; declare
  `input_shape` / `output_shape` in `__init__`. Encodings compose with the `&`
  operator (see Usage below).
- `ConcatenatedEncoding(sub_encodings)` — applies a list of encodings to
  adjacent slices of the input feature vector and concatenates the outputs.
  Built automatically by `&`; nested concatenations are auto-flattened.
- `TrivialEncoding(dimension)` — identity. Use as a placeholder slot inside a
  concatenation when a feature should pass through unchanged.
- `NopeEncoding(dimension)` — drops `dimension` features from the output. Useful
  for masking inputs out of a model entirely while keeping the dataset schema.
- `AbsEncoding(dimension)` — element-wise `|x|`. Enforces evenness in the learnt
  function under `x -> -x`.
- `SignEncoding(dimension)` — element-wise `sign(x)`.
- `ReciprocalEncoding(dimension)` — element-wise `1/x`.
- `LogEncoding(dimension, base=-1)` — element-wise logarithm; `base=-1`
  (default sentinel) selects the natural log.
- `ExponentialEncoding(dimension)` — element-wise `exp(x)`.
- `LogSoftmaxEncoding(num_classes)` — log-softmax over the last axis. Encodes a
  vector of logits as a discrete log-probability distribution.
- `ContinuousPeriodicEncoding(range_=(-pi, pi))` — maps a single angle `theta`
  to `(cos(2*pi*theta/period), sin(2*pi*theta/period))`. Any continuous function
  of the two outputs is automatically continuous and periodic in `theta`,
  including across the boundary.
- `PeriodicEncoding(range_)` — applies the modulus to wrap an input back into
  `range_`. Periodic but discontinuous at the boundary.
- `SphericalUnitVectorEncoding()` — takes `(theta, phi)` (polar, azimuthal) and
  returns the corresponding 3D unit vector `(sin theta cos phi, sin theta sin phi, cos theta)`.
- `MatrixEncoding(dimension, dimension2=None)` — reshapes a length
  `dimension * dimension2` vector into a `(dimension, dimension2)` matrix.
- `AntiSymmetricMatrixEncoding(dimension)` — reshapes a length `dimension**2`
  vector into a `(dimension, dimension)` matrix and antisymmetrises it via
  `0.5 * (M - M.T)`.

## Usage

Every encoding is a regular `nn.Module`, so it can be called directly or
dropped into a `nn.Sequential`. The intended usage is to build a single input
encoding for a feature vector by composing per-feature encodings with `&`, and
then to hand the result to `iwpc.models.utils.basic_model_factory` (which
inserts it as the first layer of the network):

```python
import torch
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.encodings.continuous_periodic_encoding import ContinuousPeriodicEncoding
from iwpc.encodings.abs_encoding import AbsEncoding
from iwpc.models.utils import basic_model_factory

# Feature vector layout: (radius r, angle theta, displacement z) where the
# learnt function should be continuous-periodic in theta and even in z.
input_encoding = (
    TrivialEncoding(1)              # r passes through (1 -> 1)
    & ContinuousPeriodicEncoding()  # theta -> (cos, sin) (1 -> 2)
    & AbsEncoding(1)                # z -> |z| (1 -> 1)
)
# input_shape = 3, output_shape = 4

x = torch.tensor([[2.0, 3.0, -1.5]])
encoded = input_encoding(x)         # shape (1, 4)

# Use as the first layer of a network.
model = basic_model_factory(input=input_encoding, output=1, hidden_widths=[64, 64])
```

The `&` operator returns a `ConcatenatedEncoding` whose `input_shape` is the sum
of the sub-encodings' input dimensions and whose `output_shape` is the sum of
their output dimensions. The j-th sub-encoding is applied to the input slice
`x[..., cum_in[j]:cum_in[j+1]]`. Nested concatenations are auto-flattened, so
`(A & B) & C` and `A & (B & C)` are equivalent.

## Extending

To add a new encoding, subclass `Encoding`, declare the input and output shapes
to `super().__init__`, and implement `_encode(self, x: Tensor) -> Tensor`:

```python
from torch import Tensor
from iwpc.encodings.encoding_base import Encoding

class SquareEncoding(Encoding):
    """Element-wise x ** 2."""
    def __init__(self, dimension: int):
        super().__init__(dimension, dimension)

    def _encode(self, x: Tensor) -> Tensor:
        return x ** 2
```

Notes:

- Shapes can be ints (vector encoding) or array-likes (higher-rank output, e.g.
  `MatrixEncoding`). Only encodings with vector input *and* vector output may
  participate in `&` / `ConcatenatedEncoding`; this is enforced at construction.
- Register any non-trainable constants with `self.register_buffer(...)` so they
  travel with `.to(device)` / state-dict save/load (see
  `ContinuousPeriodicEncoding.period`).
- Public classes/methods follow numpy-style docstrings (see `CLAUDE.md`).

## Related

- `iwpc.models.utils.basic_model_factory` — accepts an `Encoding` as its
  `input` (and optionally `output`) and inserts it at the appropriate end of
  the MLP.
- `iwpc.learn_dist.kernels` — `FiniteKernel`, `MultivariateGaussianKernel`,
  `AddCondKernel` etc. all accept an `Encoding` for their conditioning inputs.
- `iwpc.symmetries` — an alternative way to bake structure into a model
  (averaging over a group action) when the structure is not expressible as a
  per-feature reparametrisation.
