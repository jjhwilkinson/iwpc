# `iwpc.symmetries`

## Purpose

Symmetry-aware modelling. Two distinct hierarchies:

1. **Vector-space group actions** (`GroupAction`, `GroupActionElement`) — actions on a single
   vector space `R^dim`. Element interface: `action(x) -> Tensor`. The mathematical primitive.
2. **Separable function-space group actions** (`SeparableGroupAction`,
   `SeparableGroupActionElement`) — actions on the function space of maps `R^input_dim ->
   R^output_dim` that factor as `[g.f](x) = g_out.(f(g_in.x))` for separate vector-space
   actions of `G` on the input and output spaces. Each separable element holds two
   vector-space elements internally — one acting on the input space, one on the output space.

Given a neural function `f : R^input_dim -> R^output_dim`, `SeparableGroupAction.symmetrize(f)`
makes `f` invariant under the group action via Haar averaging, and `.complement(f)` builds
the orthogonal complement of that invariant subspace. In divergence-estimation runs this lets
you bake known physical symmetries (periodicity, parity, sign-flips, …) into the critic so the
learnt lower bound only uses information the symmetry does not explain — see
`iwpc.divergences.asymmetry_estimator` for the canonical consumer.

## Layout

### Vector-space hierarchy

- **`GroupAction`** (`group_action.py`) — abstract group on `R^dim`. Provides `batch()` (Haar
  sample) and the composition operators `&` (direct product on disjoint dim ranges) and `*`
  (joint action on the same space).
- **`GroupActionElement`** (`group_action_element.py`) — abstract single element `g`.
  Implements `action(x)`. Exposes a static `is_identity: bool` flag (default `False`) that
  the separable layer reads to dedupe model evaluations. `Identity` is a ready-made element
  with `is_identity = True`. Elements compose via `*` (group multiplication) and `&` (direct
  product).

Concrete vector-space classes:

- `FiniteGroupAction` — explicit enumeration of all elements (identity is always prepended).
  Overrides `&` / `*` so finite × finite enumerates the full direct / Cartesian product.
- `ProductGroupAction`, `ProductActionElement` — direct product on disjoint dim ranges.
  Auto-flattened via `merge`.
- `JointGroupAction`, `ComposedActionElement` — same-space composition. Auto-flattened.
- `LambdaAction(dim, fn)` — element built from a single callable.
- `ProdAddAction(prod, add, dim)` — element of the form `x -> p * x + q`. Composition and
  direct product of two `ProdAddAction`s stay analytic (single `ProdAddAction`).

### Separable hierarchy

- **`SeparableGroupAction`** (`separable_group_action.py`) — abstract function-space group.
  Provides `batch()` returning `SeparableGroupActionElement`s and the model wrappers
  `symmetrize(f)` / `complement(f)`. The composition operators `&` (per-side direct product)
  and `*` (per-side joint action) zip fresh batches from the operands and combine each pair
  per side via the vector-space `&` / `*` operators.
- **`SeparableGroupActionElement`** — concrete pair wrapper holding an `input_action:
  GroupActionElement` and an `output_action: GroupActionElement`. Exposes
  `input_space_action(x)` / `output_space_action(x)` (the legacy contract) plus
  `input_is_identity` for dedupe.
- **`SeparableFiniteGroupAction`** — explicit list of `SeparableGroupActionElement` pairs
  with the identity pair prepended. Overrides `&` / `*` to enumerate the full direct /
  Cartesian product when both operands are finite.
- **`SeparableProductGroupAction`**, **`SeparableJointGroupAction`** — generic wrappers
  zipping sub-group batches per side.

### Model wrappers (`symmetrized_model.py`, `complement_model.py`)

- `SymmetrizedModel(G, f)` evaluates `(1/|batch|) sum_g g_out.(f(g_in.x))`, where `G` is a
  `SeparableGroupAction`. Dedupes calls to `f` for any element whose `input_is_identity` is
  `True`. The legacy `InputSpaceInvariantException` raised from `input_space_action` is also
  caught for back-compat with user-defined elements.
- `ComplementModel(G, f)` evaluates `f(x) - SymmetrizedModel(G, f)(x)`.

## Usage

### Defining a custom action: Z_2 sign flip on a 1D output

```python
from iwpc.symmetries import SeparableGroupActionElement

flip = SeparableGroupActionElement.from_callables(
    input_dim=2, output_dim=1, output_fn=lambda y: -y,
)
Z2 = flip.to_group()  # SeparableFiniteGroupAction containing (identity_pair, flip)
```

`Z2.batch()` returns `(identity_pair, flip)`; `SymmetrizedModel(Z2, f)(x)` evaluates to
`(f(x) - f(x)) / 2 = 0` for any `f`, and `ComplementModel(Z2, f)` reduces to `f` itself.

### Wrapping an existing model

```python
import torch.nn as nn
from iwpc.symmetries import SeparableGroupActionElement

# Z_2 parity: flip sign of x_0 on the input, leave output alone
parity = SeparableGroupActionElement.from_prod_add(
    input_prod=[-1.0, 1.0], output_dim=1,
).to_group()  # involution -> finite group with identity

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
invariant_model = parity.symmetrize(model)        # f(x_0, x_1) = f(-x_0, x_1)
odd_model       = parity.complement(model)        # output orthogonal to that subspace
```

`SeparableGroupActionElement.to_group()` is only valid when the element is an involution
(its own inverse).

### Via `basic_model_factory`

```python
from iwpc.models.utils import basic_model_factory
from iwpc.symmetries import SeparableGroupActionElement

reflection = SeparableGroupActionElement.from_prod_add(input_prod=[-1.0], output_dim=1).to_group()
sign_flip  = SeparableGroupActionElement.from_prod_add(output_prod=[-1.0], input_dim=1).to_group()

model = basic_model_factory(
    input=1, output=1, hidden_layer_sizes=(64, 64),
    symmetries=[reflection],            # f becomes invariant in x
    complement_symmetries=[sign_flip],  # f becomes odd in y (output)
)
```

The factory coerces an iterable of `SeparableGroupAction`s into a single
`SeparableJointGroupAction` (via `*`) before wrapping, so passing a list is equivalent to
one combined joint action.

### Composing actions declaratively

```python
G = G1 & G2            # direct product, acts on disjoint slices [0:G1.input_dim), [G1.input_dim: ...)
H = H1 * H2            # joint action on the same space (H1, H2 must agree on dims)
g = g1 * g2 * g3       # element multiplication, auto-flattened
```

When all operands are `SeparableFiniteGroupAction`s, `&` / `*` enumerate the full product
group eagerly; otherwise `batch()` zips fresh samples from each sub-group and combines them
per side.

### Building separable actions from two independent vector-space groups

```python
from iwpc.symmetries import PairedSeparableGroupAction, ProdAddAction

input_group  = ProdAddAction(prod=[-1.0]).to_group()  # vector-space Z_2 on R^1
output_group = ProdAddAction(prod=[-1.0]).to_group()  # vector-space Z_2 on R^1
G = PairedSeparableGroupAction(input_group, output_group)
# G.batch() zips one element from each side independently — i.e. the direct product
# separable action G_in x G_out
```

Note: `GroupActionElement.to_group()` on a vector-space element returns a
`FiniteGroupAction` (vector-space). To recover the *separable* finite group containing the
identity and this single involution, use `SeparableGroupActionElement.to_group()`.
