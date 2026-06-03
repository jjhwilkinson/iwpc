from typing import Callable

from torch import Tensor
from torch.nn import Module

from iwpc.symmetries.group_action import GroupAction
from iwpc.symmetries.symmetrized_model import (
    REDUCTION_MEAN,
    VALID_REDUCTIONS,
    evaluate_group_action_outputs,
    reduce_orbit,
)


class ComplementModel(Module):
    """
    Group actions, G, define a projection operator S_G where S_Gf(x) = E_G[gf(x)] and expectation is taken with
    respect to the Haar measure on G. This wrapper module implements the complement projection operator on the
    base_function, 1 - S_G. Note that the averaging procedure can significantly increase model evaluation time.

    The orbit reduction is configurable via `reduction`. With the default `mean`, the module returns
    ``base(x) - mean_g[ g . base(g.x) ]`` — the orthogonal projection onto the G-anti-invariant subspace. With
    `log_mean_exp` it returns ``base(x) - log( mean_g[ exp( g . base(g.x) ) ] )`` — the log-space analogue, which
    guarantees ``mean_g[ exp( g . forward(g.x) ) ] = 1`` so `exp(forward)` is the renormalised orbit image of
    `exp(base)`. The log-space reduction is numerically bounded everywhere thanks to `torch.logsumexp`
    """
    def __init__(
        self,
        group: GroupAction,
        base_function: Callable[[...], Tensor],
        reduction: str = REDUCTION_MEAN,
    ):
        """
        Parameters
        ----------
        group
            A group action for which the resulting module should live in the symmetrized complement
        base_function
            A function
        reduction
            Either `mean` (default; linear-space complement) or `log_mean_exp` (log-space complement). See `reduce_orbit`
        """
        super().__init__()
        if reduction not in VALID_REDUCTIONS:
            raise ValueError(f"Unknown reduction {reduction!r}; expected one of {VALID_REDUCTIONS}")
        self.group = group
        self.base_model = base_function
        self.reduction = reduction

    def forward(self, input: Tensor) -> Tensor:
        """
        Evaluates the complement of the chosen orbit average. The transformed inputs for every group element and the
        original input are all passed through base_model in a single batched forward pass, so any input-dependent
        layers in base_model (running normalisation, batch normalisation, dropout, ...) apply consistently across the
        symmetrised and the original branches. Evaluating these branches in separate forward passes (as a naive
        ``base_model(input) - SymmetrizedModel(input)`` implementation would) breaks the projection identities in
        train mode

        Parameters
        ----------
        input
            An input Tensor

        Returns
        -------
        Tensor
        """
        per_action_outputs, base_input_output = evaluate_group_action_outputs(
            self.group, self.base_model, input
        )
        symmetrised = reduce_orbit(per_action_outputs, self.reduction)
        return base_input_output - symmetrised
