# AGENTS - `iwpc.encodings`

## Subclass contract

An `Encoding` is an `nn.Module` + `ABC`. To add one:

1. Call `super().__init__(input_shape, output_shape)`. Shapes may be an `int`
   (vector) or a list/tuple (higher-rank). They are stored as int buffers
   `self.input_shape` / `self.output_shape`, so they move with `.to(device)`.
2. Implement `_encode(self, x: Tensor) -> Tensor`. `x` has shape
   `(..., *input_shape)`; return `(..., *output_shape)`. `forward` simply calls
   `_encode`.
3. Do NOT override `forward` or `__and__` unless you have a strong reason.
4. Match numpy-style docstrings on every public class/method (see CLAUDE.md).

`is_vector_input` / `is_vector_output` test whether the corresponding shape is
1D. Non-vector encodings (e.g. `MatrixEncoding`, `AntiSymmetricMatrixEncoding`)
cannot participate in `&` / `ConcatenatedEncoding`.

## `&` operator and `ConcatenatedEncoding`

- `a & b` returns `ConcatenatedEncoding.merge(a, b)`. Nested
  `ConcatenatedEncoding`s are auto-flattened (un-curried), so `(a & b) & c` and
  `a & (b & c)` produce the same flat list of sub-encodings.
- `ConcatenatedEncoding` requires every sub-encoding to be vector-in /
  vector-out; the constructor raises `ValueError` otherwise.
- It stores `cum_input_shapes` and, at forward time, slices the last axis of
  `x` into adjacent chunks `x[..., low:high]` per sub-encoding, then
  concatenates the outputs along `dim=-1`. Input feature ordering matters and
  must match the order of the `&` chain.
- The last-axis size check is enforced: shape mismatch raises `ValueError`.

## Interaction with `basic_model_factory`

`iwpc.models.utils.basic_model_factory(input=..., output=..., ...)`:

- `input` may be an `Encoding`; it must satisfy `is_vector_output`, and gets
  inserted as the first layer. `input_shape` for the model is taken from
  `encoding.output_shape[0]`.
- `output` may be an `Encoding`; it must satisfy `is_vector_input`, and is
  appended as the final layer. `output_shape` is `encoding.input_shape[0]`.
- `independent_sum_model_factory` similarly accepts an output `Encoding` (or
  int, which gets wrapped in `TrivialEncoding`).

So: use vector-in/vector-out encodings (Trivial, ContinuousPeriodic, Abs, Log,
LogSoftmax, etc.) for `input=` and `output=`. Higher-rank-output encodings
(Matrix, AntiSymmetricMatrix) are only useful inside custom models that consume
matrix-shaped tensors.

## Per-encoding gotchas

- `ContinuousPeriodicEncoding`: scales by `2*pi / period` first; pass `range_`
  matching your data's actual period (default `(-pi, pi)`). Input must have a
  trailing `1` axis, not be a flat `(N,)` tensor.
- `PeriodicEncoding`: only wraps via modulo; does not enforce continuity across
  the boundary. Prefer `ContinuousPeriodicEncoding` when a NN consumes the
  result.
- `SphericalUnitVectorEncoding._encode` indexes with `x[:, 0]` / `x[:, 1]`, so
  it expects a 2D input `(N, 2)`, not the general `(..., 2)` advertised by the
  base class. Don't feed it higher-rank batches.
- `LogEncoding`: `base` defaults to `-1`, which the code treats as "natural
  log" (`log_base = 1`). Positive `base` selects a real base. Inputs must be
  positive; no clamping is performed.
- `ReciprocalEncoding` / `LogEncoding`: no guard against zero/negative input.
  Compose with `AbsEncoding` or clamp upstream if needed.
- `NopeEncoding._encode` returns `torch.zeros((x.shape[0], 0), ...)`. It
  assumes a single leading batch dim; nested batch axes will be flattened.
- `AntiSymmetricMatrixEncoding`: input is the flat `d*d` vector, output is
  shape `(d, d)`; diagonal is zero by construction.
- `LogSoftmaxEncoding`: applies along `dim=-1`. Inputs are logits, outputs are
  log-probabilities summing (in exp) to 1.

## Convention recap

- 0 = p, 1 = q labels are unrelated to encodings; encodings act on feature
  columns only.
- Don't reformat existing encodings or change public shapes — they are part of
  the public API surface (see CLAUDE.md "public API surface" rule).
