# iwpc.encodings — AGENTS.md

## Public API
- `Encoding(input_shape: int | ArrayLike, output_shape: int | ArrayLike)` — abstract `nn.Module` base.
  - `_encode(x: Tensor) -> Tensor` — abstract; subclasses implement.
  - `forward(x: Tensor) -> Tensor` — calls `_encode`.
  - `is_vector_input -> bool` — true iff `input_shape` is rank-1 (a length).
  - `is_vector_output -> bool` — true iff `output_shape` is rank-1.
  - `__and__(other: Encoding) -> ConcatenatedEncoding` — `self & other`; delegates to `ConcatenatedEncoding.merge`.
  - Buffers: `input_shape: IntTensor`, `output_shape: IntTensor`.
- `ConcatenatedEncoding(sub_encodings: List[Encoding])` — applies sub-encodings to adjacent slices and concatenates outputs.
  - `merge(a: Encoding, b: Encoding) -> ConcatenatedEncoding` — classmethod; auto-flattens if either operand is itself a `ConcatenatedEncoding`.
  - Buffer: `cum_input_shapes: IntTensor` of length `len(sub_encodings) + 1`.
- `TrivialEncoding(dimension: int)` — identity; `dim -> dim`.
- `NopeEncoding(dimension: int)` — drops all features; `dim -> 0`.
- `AbsEncoding(dimension: int)` — `torch.abs(x)`; `dim -> dim`.
- `SignEncoding(dimension: int)` — `x.sign()`; `dim -> dim`.
- `ReciprocalEncoding(dimension: int)` — `1 / x`; `dim -> dim`.
- `LogEncoding(dimension: int, base: float = -1)` — `log_base(x)`; `base < 0` sentinel selects natural log; `dim -> dim`.
- `ExponentialEncoding(dimension: int)` — `torch.exp(x)`; `dim -> dim`.
- `LogSoftmaxEncoding(num_classes: int)` — log-softmax over last axis via `logits - logits.logsumexp(-1, keepdim=True)`; `n -> n`.
- `ContinuousPeriodicEncoding(range_: Tuple[float, float] = (-pi, pi))` — `theta -> (cos(2*pi*theta/period), sin(...))`; `1 -> 2`. Buffer: `period`.
- `PeriodicEncoding(range_: Tuple[float, float])` — modulo wrap into `range_`; `1 -> 1`. Buffer: `range`.
- `SphericalUnitVectorEncoding()` — `(theta, phi) -> (sin theta cos phi, sin theta sin phi, cos theta)`; `2 -> 3`.
- `MatrixEncoding(dimension: int, dimension2: int | None = None)` — reshape `d*d2` vector to `(d, d2)`; `dimension2` defaults to `dimension`. Buffers: `dimension`, `dimension2`.
- `AntiSymmetricMatrixEncoding(dimension: int)` — reshape length-`d*d` vector to `(d, d)` matrix `M`, then return `0.5 * (M - M.T)`. Buffer: `dimension`.

## Invariants
- `_encode` accepts shape `(..., *input_shape)` and returns shape `(..., *output_shape)`. Batch axes are preserved unchanged (the only exception is `NopeEncoding` which still preserves a leading batch axis but produces zero feature columns).
- `ConcatenatedEncoding` raises `ValueError` at construction if any sub-encoding has non-vector input or non-vector output.
- `ConcatenatedEncoding._encode` raises `ValueError` if `x.shape[-1] != input_shape[-1]`.
- `ConcatenatedEncoding.merge` un-curries: nested `ConcatenatedEncoding` operands are flattened into a single sub-encoding list; `(A & B) & C == A & (B & C)` structurally.
- All shape parameters are stored as `IntTensor` buffers via `register_buffer`, so they follow `.to(device)` and round-trip through `state_dict`.

## Numerical tricks
- `LogSoftmaxEncoding` uses `logits - logits.logsumexp(-1, keepdim=True)` rather than `log(softmax(...))` for numerical stability.
- `LogEncoding` uses `base < 0` as a sentinel for "natural log" (skips the `log(base)` division) — passing a real negative base is undefined behaviour.

## Conventions
- `&` is the canonical composition operator. Reach for it instead of manually instantiating `ConcatenatedEncoding`.
- `TrivialEncoding(k)` is the conventional placeholder for "feed these `k` features through unchanged" inside a `&`-chain.
- `NopeEncoding(k)` is the conventional way to mask `k` features out of a model while keeping the dataset schema fixed.
- Non-trainable constants are stored with `register_buffer`, not as plain attributes — needed so they travel with `.to(device)` and `state_dict`.
- Numpy-style docstrings on every public class and method (repo-wide convention; see root `CLAUDE.md`).

## Gotchas
- Only vector-in / vector-out encodings can participate in `&`. `MatrixEncoding` and `AntiSymmetricMatrixEncoding` produce rank-2 outputs and therefore cannot be on either side of `&`; they are typically used as the *output* encoding of a model (e.g. via `basic_model_factory(output_encoding=...)`).
- `SphericalUnitVectorEncoding._encode` indexes with `x[:, 0]` / `x[:, 1]` (not `x[..., 0]`), so it implicitly assumes a single leading batch axis; pre-flatten any extra batch dimensions before calling it.
- `PeriodicEncoding` is *not* the continuous variant — a network on top of it is still free to be discontinuous at the wrap-around boundary. Use `ContinuousPeriodicEncoding` when continuity across the boundary is required.
- `ContinuousPeriodicEncoding` rescales by `2*pi / period`, so a non-default `range_` such as `(0, 1)` works as expected; do not pre-scale the input yourself.
- `LogEncoding(base=...)` uses `register_buffer('base', base)` directly — `base` is expected to be something `register_buffer` accepts (a tensor); the default `-1` is a sentinel and is only ever compared with `<`.
- `Encoding.__init__` calls `to_shape_tensor` on shapes, so passing a Python `int` (vector) or a list/tuple/array (higher rank) both work; the recorded `input_shape` / `output_shape` is always at least 1-D.

## Cross-refs
- Imports from: `torch`, `numpy`. No intra-iwpc imports beyond `iwpc.encodings.encoding_base` (each concrete encoding subclasses `Encoding`).
- Used by: `iwpc.models.utils.basic_model_factory` (input/output encoding slot — vector-output required for input slot, vector-input required for output slot); `iwpc.learn_dist.kernels.*` (`FiniteKernel`, `MultivariateGaussianKernel`, `AddCondKernel`, `DiracKernel`, etc. take an `Encoding` for conditioning inputs); downstream packages such as `atlas_benchmark.muons.model_inputs` build trajectory encodings via `&`.
