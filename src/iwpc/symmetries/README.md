# `iwpc.symmetries`

## Purpose

Symmetry-aware modelling. Given a neural function `f : R^M -> R^N`, this sub-package
builds wrappers that make `f` either *invariant* under a group action `G` (via
averaging over the Haar measure) or *orthogonal* to that invariant subspace (the
"complement"). The averaging is restricted to actions that factor as
`[g.f](x) = g.(f(g.x))` for separate actions of `G` on the input and output spaces.

In divergence-estimation runs this lets you bake known physical symmetries
(periodicity, parity, sign-flips, …) into the critic so the learnt lower bound
only uses information the symmetry does not explain — see
`iwpc.modules.asymmetry_estimator` for the canonical consumer.

## Layout

Three layers:

- **`GroupAction`** (`group_action.py`) — abstract group. Knows the input/output
  dims and provides `batch()`, a sample from the Haar measure used for the
  Monte-Carlo average. Implements `symmetrize(model)` / `complement(model)` and
  the composition operators `&` (direct product on disjoint dim ranges) and `*`
  (joint action on the same space).
- **`GroupActionElement`** (`group_action_element.py`) — abstract single element
  `g`. Implements `input_space_action` and `output_space_action`.
  `InputSpaceInvariantException` is the signalled-trivial case so
  `SymmetrizedModel` can de-duplicate model calls. `Identity` is a ready-made
  element. Elements compose via `*` (group multiplication) and `&` (product on
  disjoint dims).
- **Model wrappers** (`symmetrized_model.py`, `complement_model.py`) —
  `SymmetrizedModel` evaluates `(1/|batch|) sum_g g.(f(g.x))`; `ComplementModel`
  evaluates `f(x) - SymmetrizedModel(f)(x)`. Both treat `batch()` as the sample
  set per forward pass.

Concrete actions / elements:

- `FiniteGroupAction` — explicit enumeration of all elements (identity is always
  prepended). Overrides `&` / `*` to enumerate the full direct / Cartesian product
  when both sides are finite.
- `ProductGroupAction`, `ProductActionElement` — direct product on disjoint dim
  ranges. Auto-flattened via `merge`.
- `JointGroupAction`, `ComposedActionElement` — same-space composition.
  Auto-flattened.
- `LambdaAction` — element built from two arbitrary callables `(input_fn,
  output_fn)`. Set `input_fn=None` to advertise input-space invariance.
- `ProdAddAction` — element of the form `x -> p*x + q` per space. Composition and
  direct product of two `ProdAddAction`s stay analytic (single `ProdAddAction`).

## Usage

### Defining a custom action: Z_2 sign flip on a 1D output

```python
import torch
from iwpc.symmetries import FiniteGroupAction, LambdaAction

flip = LambdaAction(input_dim=2, output_dim=1, output_fn=lambda y: -y)  # input_fn=None: input invariant
Z2 = FiniteGroupAction([flip], input_dim=2, output_dim=1)
```

`Z2.batch()` returns `(Identity, flip)`; `SymmetrizedModel(Z2, f)(x)` then
evaluates to `(f(x) - f(x)) / 2 = 0` for any `f`, and `ComplementModel(Z2, f)`
reduces to `f` itself.

### Wrapping an existing model

```python
import torch.nn as nn
from iwpc.symmetries import ProdAddAction

# Periodic-style action: flip sign of x_0 on the input, leave output alone
parity = ProdAddAction(input_prod=[-1.0, 1.0], output_dim=1).to_group()  # Z_2 from an involution

model = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 1))
invariant_model = parity.symmetrize(model)        # f(x_0, x_1) = f(-x_0, x_1)
odd_model       = parity.complement(model)        # output orthogonal to that subspace
```

`to_group()` is only valid when the element is an involution (its own inverse).

### Via `basic_model_factory`

```python
from iwpc.models.utils import basic_model_factory
from iwpc.symmetries import ProdAddAction

reflection = ProdAddAction(input_prod=[-1.0], output_dim=1).to_group()
sign_flip  = ProdAddAction(output_prod=[-1.0], input_dim=1).to_group()

model = basic_model_factory(
    input=1, output=1, hidden_layer_sizes=(64, 64),
    symmetries=[reflection],            # f becomes invariant in x
    complement_symmetries=[sign_flip],  # f becomes odd in y (output)
)
```

The factory coerces an iterable of `GroupAction`s into a single `JointGroupAction`
(via `*`) before wrapping, so passing a list is equivalent to one combined joint
action.

### Composing actions declaratively

```python
G = G1 & G2            # direct product, acts on disjoint slices [0:G1.input_dim), [G1.input_dim: ...)
H = H1 * H2            # joint action on the same space (H1, H2 must agree on dims)
g = g1 * g2 * g3       # element multiplication, auto-flattened into one ComposedActionElement
```

When all operands are `FiniteGroupAction`s, `&` / `*` enumerate the full product
group eagerly; otherwise `batch()` zips fresh samples from each sub-group.
