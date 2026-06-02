# Agent notes — `iwpc.symmetries`

## Mental model

Two hierarchies plus one wrapper layer.

**Vector-space hierarchy** — the mathematical primitive:

- `GroupAction` (the *group*): owns `dim`, exposes `batch() -> Tuple[GroupActionElement,
  ...]`. `batch()` is the Haar-measure Monte-Carlo sampler — finite groups return the full
  group every call (identity prepended via `FiniteGroupAction.__init__`); infinite groups
  return a fresh sample on each call. Re-sampled every forward pass.
- `GroupActionElement` (an *element*): owns `dim`, implements `action(x)`. Exposes a static
  `is_identity: bool` class attribute (default `False`); concrete subclasses override it
  (statically on `Identity`, dynamically on `ProdAddAction`/`ComposedActionElement`/
  `ProductActionElement` after construction).

**Separable hierarchy** — function-space wrapper on top:

- `SeparableGroupAction` (group of `[g.f](x) = g_out.f(g_in.x)`): owns `input_dim`,
  `output_dim`, exposes `batch() -> Tuple[SeparableGroupActionElement, ...]`. Implements
  `symmetrize(f)` / `complement(f)`. Concrete subclasses: `SeparableFiniteGroupAction`,
  `SeparableProductGroupAction`, `SeparableJointGroupAction`, plus the private
  `PairedSeparableGroupAction` (the direct-product case: two independent vector-space
  groups zipped at sample time).
- `SeparableGroupActionElement`: holds `input_action: GroupActionElement` (acts on
  `R^input_dim`) and `output_action: GroupActionElement` (acts on `R^output_dim`). Exposes
  `input_space_action(x)`, `output_space_action(x)`, and `input_is_identity`. Provides
  `from_callables` and `from_prod_add` classmethod factories mirroring the legacy
  ergonomic constructors.

**Wrappers**:

- `SymmetrizedModel(G, f)(x)` computes `mean_{g in G.batch()} g_out(f(g_in(x)))`; dedupes
  evaluations of `f` for any element whose `input_is_identity` is `True`. Also catches the
  legacy `InputSpaceInvariantException` raised by user-defined elements for back-compat.
- `ComplementModel(G, f)(x) = f(x) - SymmetrizedModel(G, f)(x)`.

`G` here is a `SeparableGroupAction`. The vector-space `GroupAction` does *not* expose
`symmetrize` / `complement` — those are inherently function-space (input + output)
operations.

## Identity detection

`is_identity` is the load-bearing flag on `GroupActionElement`. Subclass guidance:

- `Identity` — `is_identity = True` as a class attribute.
- `ProdAddAction` — set as an instance attribute in `__init__` based on `prod == 1` and
  `add == 0`. Buffers always materialised at full `dim`.
- `ComposedActionElement` / `ProductActionElement` — set as instance attribute: True iff
  every sub-element is identity.
- `LambdaAction` — defaults to `False` (no way to introspect a callable).

`SeparableGroupActionElement.input_is_identity` is just `self.input_action.is_identity`. Its
`input_space_action(x)` still raises `InputSpaceInvariantException` when the input side is
identity, so any user code that wraps the legacy exception pattern keeps working. New code
should prefer reading `input_is_identity` directly.

## Composition operators (and their flattening)

Vector-space hierarchy:

| Op | Element level | Group level | Semantics |
|----|---------------|-------------|-----------|
| `*` | `ComposedActionElement` | `JointGroupAction` | Right-to-left compose on the same space; all sub-dims must agree |
| `&` | `ProductActionElement` | `ProductGroupAction` | Disjoint dim ranges; concatenated input/output |

All four use a `merge` classmethod that splices any operand that is already the same wrapper,
so `a * b * c` is a single 3-element `ComposedActionElement`, not a binary tree.
`FiniteGroupAction.__and__` / `__mul__` have a fast path: when both operands are finite, they
enumerate the full product / Cartesian product eagerly via `_build_finite_product` /
`_build_finite_joint` and return a flat `FiniteGroupAction`. `ProdAddAction.__mul__` /
`__and__` similarly stay analytic when both operands are `ProdAddAction`.

Separable hierarchy:

| Op | Element level | Group level | Semantics |
|----|---------------|-------------|-----------|
| `*` | per-side `*` on input + output | `SeparableJointGroupAction` | Joint action; per-side dims must agree |
| `&` | per-side `&` on input + output | `SeparableProductGroupAction` | Disjoint dim ranges on both sides |

Separable element operators delegate per side, so analytic fast paths from the vector layer
(e.g. `ProdAddAction * ProdAddAction = ProdAddAction`) carry over for free.
`SeparableFiniteGroupAction.__and__` / `__mul__` have the same finite × finite fast paths.

## Cross-package consumers

- `iwpc.models.utils.basic_model_factory` (`symmetries=`, `complement_symmetries=`) — accepts
  a single `SeparableGroupAction` or an iterable. `_coerce_group_action` combines an iterable
  via `*` (joint action), so the list is reduced to one `SeparableJointGroupAction` and
  wrapped exactly once. Applied *after* model construction, *after* the input `Encoding`.
- `iwpc.divergences.asymmetry_estimator.AsymmetryEstimator` — uses `group.symmetrize` on a
  custom callable that computes naive-q summands, i.e. symmetrising the *log-ratio* rather
  than the network directly.

## Adding a new vector-space `GroupAction` / `GroupActionElement`

1. Subclass `GroupAction` and call `super().__init__(dim)`.
2. Implement `batch() -> Tuple[GroupActionElement, ...]`. For continuous groups, draw N
   elements from Haar each call — the batch size *is* the Monte-Carlo sample count for the
   downstream `SymmetrizedModel`. `FiniteGroupAction` prepends `Identity` for you.
3. For elements, subclass `GroupActionElement(dim)`, implement `action(x)`, and set
   `is_identity = True` only when you genuinely return `x` unchanged. Trivial identity passes
   are how `SymmetrizedModel` skips redundant `base_model` evaluations.
4. Prefer subclassing `FiniteGroupAction` for finite groups so you inherit the `&` / `*`
   finite fast-paths.
5. Register sub-modules as `ModuleList` if you store element lists, so they move with
   `.to(device)`.

## Adding a new `SeparableGroupAction`

Usually you don't — build one by combining two vector-space groups via
`PairedSeparableGroupAction(input_group, output_group)`, or enumerate
pairs explicitly with `SeparableFiniteGroupAction(pairs, input_dim, output_dim)`. Subclass
`SeparableGroupAction` directly only if you need a custom Haar sampler whose pairs do not
factor as independent products.

## Don'ts

- Don't add `symmetrize` / `complement` to the vector-space `GroupAction`. They belong on
  `SeparableGroupAction` because they need both an input action and an output action.
- Don't bypass the `is_identity` flag by always returning `x` — set the flag so
  `SymmetrizedModel` can dedupe.
- Don't construct a `SeparableGroupActionElement` with mismatched per-side dims; the input
  and output halves are independent and may differ.
