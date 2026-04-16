from torch import Tensor

from iwpc.divergences import DifferentiableFDivergence


class FDivergenceGradientSurrogateLoss:
    def __init__(self, divergence: DifferentiableFDivergence):
        self.divergence = divergence

    def call(
        self,
        log_q_given_base: Tensor,
        q_weights: Tensor,
        log_p_over_q: Tensor,
    ) -> Tensor:
        return q_weights * self.divergence._f_dash_given_log(log_p_over_q) * log_q_given_base
