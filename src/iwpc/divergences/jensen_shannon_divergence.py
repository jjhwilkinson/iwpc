import math

import numpy as np
import torch
import torch.nn.functional as F
from numpy import ndarray
from scipy.special import xlogy
from torch import Tensor

from ..types import TensorOrNDArray
from .base import DifferentiableFDivergence


class JensenShannonDivergence(DifferentiableFDivergence):
    """
    Implementation of the Jensen-Shannon divergence as described in https://arxiv.org/abs/2405.06397
    """
    def __init__(self) -> None:
        """
        Initialises the divergence with display names "Jensen-Shannon" and "JSD" and caches `log_two = ln 2` as a
        Python float so it broadcasts cleanly across dtypes and devices.
        """
        super().__init__("Jensen-Shannon", "JSD")
        self.log_two = math.log(2.)

    def _f_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The JSD generating function $f(x) = \tfrac{1}{2}\left(x \log x - (x + 1) \log \tfrac{x + 1}{2}\right)$
            evaluated in pytorch
        """
        return 0.5 * (torch.special.xlogy(x, x) - torch.special.xlogy(x + 1, (x + 1) / 2))

    def _f_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The JSD generating function $f(x) = \tfrac{1}{2}\left(x \log x - (x + 1) \log \tfrac{x + 1}{2}\right)$
            evaluated in numpy
        """
        return 0.5 * (xlogy(x, x) - xlogy(x + 1, (x + 1) / 2))

    def _f_conj_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The Legendre transform $f^*(x) = -\tfrac{1}{2}\left(\ln 2 + \log(1 - \tfrac{1}{2} e^{2x})\right)$ evaluated
            in pytorch. Defined on $x < \tfrac{1}{2} \ln 2$
        """
        return - 0.5 * (self.log_two + torch.log1p(-0.5 * torch.exp(2 * x)))

    def _f_conj_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The Legendre transform $f^*(x) = -\tfrac{1}{2}\left(\ln 2 + \log(1 - \tfrac{1}{2} e^{2x})\right)$ evaluated
            in numpy. Defined on $x < \tfrac{1}{2} \ln 2$
        """
        return - 0.5 * (np.log(2.) + np.log1p(-0.5 * np.exp(2 * x)))

    def _f_dash_given_log_torch(self, log_x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The derivative $f^'(x) = \tfrac{1}{2}(\ln 2 + \log \sigma(\log x))$ evaluated in pytorch given $\log x$.
            Uses `F.logsigmoid` for numerical stability across both tails of $\log x$
        """
        return 0.5 * (self.log_two + F.logsigmoid(log_x))

    def _f_dash_given_log_np(self, log_x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The derivative $f^'(x) = \tfrac{1}{2}(\ln 2 + \log \sigma(\log x))$ evaluated in numpy given $\log x$. Uses
            `logaddexp` for numerical stability across both tails of $\log x$
        """
        return 0.5 * (np.log(2) + log_x - np.logaddexp(log_x, 0.))

    def calculate_naive_q_summands_given_log(self, log_p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        r"""
        Returns the q-side summand $f^*(f^'(p/q)) = -\tfrac{1}{2}(\ln 2 + \log \sigma(-\log(p/q)))$ directly from
        $\log(p/q)$, bypassing the generic `f_conj(f_dash_given_log(.))` composition. The substitution
        $0.5 \exp(2 f^'(p/q)) = \sigma(\log(p/q))$ collapses the composition to a single `logsigmoid` call, which is
        numerically stable across both tails of $\log(p/q)$ — the generic form loses precision near the upper boundary
        of $f^*$'s domain as $\log(p/q) \to +\infty$

        Parameters
        ----------
        log_p_over_q
            An estimator for the log-probability ratio $\log \frac{p(x)}{q(x)}$

        Returns
        -------
        TensorOrNDArray
            The values of the q-side naive-representation summand
        """
        if isinstance(log_p_over_q, Tensor):
            return -0.5 * (self.log_two + F.logsigmoid(-log_p_over_q))
        return -0.5 * (self.log_two - np.logaddexp(log_p_over_q, 0.))
