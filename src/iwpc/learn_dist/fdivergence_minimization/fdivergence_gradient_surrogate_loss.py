from torch import Tensor

from iwpc.divergences import DifferentiableFDivergence


class FDivergenceGradientSurrogateLoss:
    """
    Surrogate loss whose gradient with respect to the kernel parameters matches the gradient of the f-divergence
    Df(p || q) = E_q[f(p/q)] with respect to those parameters, where q is the kernel-induced distribution. Built by
    multiplying the kernel's `log_q_given_base` (which carries the gradient through the score-function trick) by
    -f*(f'(p/q)), since d/dθ E_q[f(r)] = E_q[(f(r) - r f'(r)) d/dθ log q] = -E_q[f*(f'(r)) d/dθ log q] with r = p/q,
    evaluated via the divergence's stable `calculate_naive_q_summands_given_log`, and the per-sample weight
    """
    def __init__(self, divergence: DifferentiableFDivergence):
        """
        Parameters
        ----------
        divergence
            The DifferentiableFDivergence whose gradient is being approximated
        """
        self.divergence = divergence

    def call(
        self,
        log_q_given_base: Tensor,
        q_weights: Tensor,
        log_p_over_q: Tensor,
    ) -> Tensor:
        """
        Evaluate the per-sample surrogate loss

        Parameters
        ----------
        log_q_given_base
            log q(x | base sample) for each sample, shape (N,). Carries the differentiable dependence on the kernel
            parameters via the score-function trick
        q_weights
            Sample weights for the q-batch, shape (N,)
        log_p_over_q
            Detached estimate of log(p / q) at each sample, shape (N,), typically produced by a learned `log_p_over_q_model`

        Returns
        -------
        Tensor
            Per-sample surrogate loss, shape (N,). Take its mean before backpropagating
        """
        return -q_weights * self.divergence.calculate_naive_q_summands_given_log(log_p_over_q) * log_q_given_base
