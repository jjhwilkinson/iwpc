# Agent notes — `iwpc.symmetries`

## Mental model

Two abstract base classes, one wrapper layer.

- `GroupAction` (the *group*): owns `input_dim`, `output_dim`, exposes
  `batch() -> Tuple[GroupActionElement, ...]`. `batch()` is the Haar-measure
  Monte-Carlo sampler — finite groups must return the full group every call (with
  identity prepended via `FiniteGroupAction.__init__`); infinite groups return a
  fresh sample on each call. Re-sampled every forward pass.
- `GroupActionElement` (an *element*): owns `input_dim`, `output_dim`, implements
  `input_space_action(x)` and `output_space_action(x)`. Restricted to actions of
  the form `[g.f](x) = g_out.(f(g_in.x))` — i.e. separable on input/output.
- `SymmetrizedModel(G, f)(x)` computes `mean_{g in G.batch()} g_out(f(g_in(x)))`;
  `ComplementModel(G, f)(x) = f(x) - SymmetrizedModel(G, f)(x)`.

## InputSpaceInvariantException

Raising it from `input_space_action` is the contract for "this element doesn't
touch the input". `SymmetrizedModel.forward` dedupes: it builds a single
unique-input list, runs `base_model` once per unique input, and stitches outputs
back using `output_indices`. Returning `x` unchanged works numerically but
forfeits the speed-up. `ProductActionElement` / `ComposedActionElement`
re-raise only if *every* sub-element is input-trivial.

## Composition operators (and their flattening)

| Op | Element level | Group level | Semantics |
|----|---------------|-------------|-----------|
| `*` | `ComposedActionElement` | `JointGroupAction` | Right-to-left compose on the same space; all sub-dims must agree |
| `&` | `ProductActionElement` | `ProductGroupAction` | Disjoint dim ranges; concatenated input/output |

All four use a `merge` classmethod that splices any operand that is already the
same wrapper, so `a * b * c` is a single 3-element `ComposedActionElement`, not a
binary tree. `FiniteGroupAction.__and__` / `__mul__` have a fast path: when both
operands are finite, they enumerate the full product / Cartesian product eagerly
via `_build_finite_product` / `_build_finite_joint` and return a flat
`FiniteGroupAction` rather than a generic wrapper. `ProdAddAction.__mul__` /
`__and__` similarly stay analytic when both operands are `ProdAddAction`.

## Special cases

- `Identity(input_dim, output_dim)` — `input_space_action` raises
  `InputSpaceInvariantException`; `output_space_action` returns `x`. Always
  prepended by `FiniteGroupAction`.
- `LambdaAction(input_fn=None, ...)` — `None` means input-trivial; pass `None`
  rather than `lambda x: x` so dedupe works.
- `ProdAddAction` — buffers always materialised at full dim (`_materialise`
  fills with `1.0` / `0.0`). `affects_input_space` is computed at construction
  and used to short-circuit `input_space_action`.
- `GroupActionElement.to_group()` — wraps a single element + identity into a
  `FiniteGroupAction`. Caller is responsible for checking the element is an
  involution; nothing enforces it.

## Cross-package consumers

- `iwpc.models.utils.basic_model_factory` (`symmetries=`, `complement_symmetries=`)
  — accepts a single `GroupAction` or an iterable. `_coerce_group_action`
  combines an iterable via `*` (joint action), so the list is reduced to one
  `JointGroupAction` and wrapped exactly once. Applied *after* model
  construction, *after* the input `Encoding`.
- `iwpc.modules.asymmetry_estimator.AsymmetryEstimator` — uses
  `group.symmetrize` on a custom callable that computes naive-q summands, i.e.
  symmetrising the *log-ratio* rather than the network directly.

## Adding a new GroupAction

1. Subclass `GroupAction`. In `__init__`, call `super().__init__(input_dim,
   output_dim)`. Both must be set; `ProductGroupAction.__init__` and similar
   sums require it.
2. Implement `batch() -> Tuple[GroupActionElement, ...]`. For continuous groups,
   draw N elements from Haar each call — the batch size *is* the Monte-Carlo
   sample count for `SymmetrizedModel`. Always include / draw the identity
   element implicitly if you want it; `FiniteGroupAction` does this for you.
3. Implement / reuse a `GroupActionElement` for each element. Raise
   `InputSpaceInvariantException` from `input_space_action` whenever the input
   action is trivial. Keep `input_dim` / `output_dim` consistent with the group.
4. If finite, prefer subclassing `FiniteGroupAction` (or constructing one with
   your non-identity elements) to inherit the `&` / `*` finite fast-paths.
5. Register sub-modules as `ModuleList` if you store element lists, so they
   move with `.to(device)`.
