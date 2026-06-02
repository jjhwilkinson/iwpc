from typing import Callable, Optional, TYPE_CHECKING

from numpy._typing import ArrayLike
from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action_element import GroupActionElement, Identity, InputSpaceInvariantException

if TYPE_CHECKING:
    from iwpc.symmetries.separable_finite_group_action import SeparableFiniteGroupAction


class SeparableGroupActionElement(Module):
    """
    A separable function-space action element built from a pair of vector-space ``GroupActionElement``
    instances — one acting on the input space R^input_dim and one on the output space R^output_dim. Together they
    represent a single function-space group element g where [g.f](x) = output_action(f(input_action(x)))

    SeparableGroupActionElements support declarative composition via Python operators, which delegate to the
    underlying vector-space elements per side

    >>> composed = g1 * g2   # input/output sides composed independently
    >>> product = g1 & g2    # input/output sides direct-producted on disjoint dim ranges
    """

    def __init__(self, input_action: GroupActionElement, output_action: GroupActionElement):
        """
        Parameters
        ----------
        input_action
            A ``GroupActionElement`` acting on the input space R^input_dim. Use
            ``Identity`` if the input space is invariant under this element
        output_action
            A ``GroupActionElement`` acting on the output space R^output_dim. Use
            ``Identity`` if the output space is invariant under this element
        """
        super().__init__()
        self.input_action = input_action
        self.output_action = output_action
        self.input_dim = input_action.dim
        self.output_dim = output_action.dim

    @property
    def input_is_identity(self) -> bool:
        """
        Returns
        -------
        bool
            True when the input-side action is the identity. ``SymmetrizedModel`` reads this to dedupe
            evaluations of the base model on the unchanged input
        """
        return self.input_action.is_identity

    def input_space_action(self, x: Tensor) -> Tensor:
        """
        Applies the input-side action to ``x``. For back-compat with the legacy contract, raises
        ``InputSpaceInvariantException`` when ``input_is_identity`` is True so existing
        ``except`` paths in user code continue to work
        """
        if self.input_action.is_identity:
            raise InputSpaceInvariantException()
        return self.input_action.action(x)

    def output_space_action(self, x: Tensor) -> Tensor:
        """
        Applies the output-side action to ``x``
        """
        return self.output_action.action(x)

    def to_group(self) -> "SeparableFiniteGroupAction":
        """
        Constructs a ``SeparableFiniteGroupAction`` containing the identity pair and this element. Warning:
        this is only valid if this element is an involution (its own inverse). It is the caller's responsibility to
        check this

        Returns
        -------
        SeparableFiniteGroupAction
            A SeparableFiniteGroupAction containing only this element and the identity pair
        """
        from iwpc.symmetries.separable_finite_group_action import SeparableFiniteGroupAction
        return SeparableFiniteGroupAction([self], input_dim=self.input_dim, output_dim=self.output_dim)

    def __mul__(self, other: "SeparableGroupActionElement") -> "SeparableGroupActionElement":
        """
        Composes two separable elements by composing their input-side and output-side actions independently via the
        vector-space ``*`` operator. Each side may un-curry into a ComposedActionElement or stay analytic (e.g. when
        both sides are ProdAddActions)

        Parameters
        ----------
        other
            A SeparableGroupActionElement to compose with

        Returns
        -------
        SeparableGroupActionElement
            The composed separable element
        """
        return SeparableGroupActionElement(
            input_action=self.input_action * other.input_action,
            output_action=self.output_action * other.output_action,
        )

    def __and__(self, other: "SeparableGroupActionElement") -> "SeparableGroupActionElement":
        """
        Direct product on disjoint dim ranges. The input and output sides are direct-producted via the vector-space
        ``&`` operator independently

        Parameters
        ----------
        other
            A SeparableGroupActionElement to take the direct product with

        Returns
        -------
        SeparableGroupActionElement
            The direct-product separable element acting on the concatenated input and output spaces
        """
        return SeparableGroupActionElement(
            input_action=self.input_action & other.input_action,
            output_action=self.output_action & other.output_action,
        )

    @classmethod
    def from_callables(
        cls,
        input_dim: int,
        output_dim: int,
        input_fn: Optional[Callable[[Tensor], Tensor]] = None,
        output_fn: Optional[Callable[[Tensor], Tensor]] = None,
    ) -> "SeparableGroupActionElement":
        """
        Convenience factory mirroring the legacy ``LambdaAction(input_fn=..., output_fn=...)`` constructor. ``None``
        on either side becomes ``Identity`` on that side

        Parameters
        ----------
        input_dim
            Dimensionality of the input space
        output_dim
            Dimensionality of the output space
        input_fn
            Optional callable acting on the input space. Pass ``None`` rather than an identity function to advertise
            input invariance to ``SymmetrizedModel``
        output_fn
            Optional callable acting on the output space. Pass ``None`` rather than an identity function to advertise
            output invariance

        Returns
        -------
        SeparableGroupActionElement
            A SeparableGroupActionElement wrapping the supplied callables (or Identity on either side)
        """
        from iwpc.symmetries.lambda_action import LambdaAction
        input_action = Identity(dim=input_dim) if input_fn is None else LambdaAction(dim=input_dim, fn=input_fn)
        output_action = Identity(dim=output_dim) if output_fn is None else LambdaAction(dim=output_dim, fn=output_fn)
        return cls(input_action=input_action, output_action=output_action)

    @classmethod
    def from_prod_add(
        cls,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        input_prod: Optional[ArrayLike] = None,
        input_add: Optional[ArrayLike] = None,
        output_prod: Optional[ArrayLike] = None,
        output_add: Optional[ArrayLike] = None,
    ) -> "SeparableGroupActionElement":
        """
        Convenience factory mirroring the legacy ``ProdAddAction(input_prod=..., output_prod=..., ...)`` constructor.
        Builds a ``ProdAddAction`` on each side

        Parameters
        ----------
        input_dim
            Dimensionality of the input space. May be omitted if inferrable from ``input_prod`` or ``input_add``
        output_dim
            Dimensionality of the output space. May be omitted if inferrable from ``output_prod`` or ``output_add``
        input_prod
            Multiplier constant for the input-side action. Defaults to ones if not provided
        input_add
            Additive constant for the input-side action. Defaults to zeros if not provided
        output_prod
            Multiplier constant for the output-side action. Defaults to ones if not provided
        output_add
            Additive constant for the output-side action. Defaults to zeros if not provided

        Returns
        -------
        SeparableGroupActionElement
            A SeparableGroupActionElement wrapping a ProdAddAction on each side
        """
        from iwpc.symmetries.prod_add_action import ProdAddAction
        input_action = ProdAddAction(prod=input_prod, add=input_add, dim=input_dim)
        output_action = ProdAddAction(prod=output_prod, add=output_add, dim=output_dim)
        return cls(input_action=input_action, output_action=output_action)
