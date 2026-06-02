import numpy as np
import torch
from numpy import ndarray
from scipy.special import xlogy
from torch import Tensor

from ..types import TensorOrNDArray
from .base import DifferentiableFDivergence


class KLDivergence(DifferentiableFDivergence):
    """
    Implementation of the Kullback-Leibler divergence as described in https://arxiv.org/abs/2405.06397
    """

    def __init__(self) -> None:
        """
        Initialises the divergence with display names "Kullback-Leibler" and "KL".
        """
        super().__init__("Kullback-Leibler", "KL")

    def _f_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The KL generating function $f(x) = x \log x$ evaluated in pytorch
        """
        return torch.special.xlogy(x, x)

    def _f_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The KL generating function $f(x) = x \log x$ evaluated in numpy
        """
        return xlogy(x, x)

    def _f_conj_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The Legendre transform $f^*(x) = \exp(x - 1)$ evaluated in pytorch
        """
        return torch.exp(x - 1)

    def _f_conj_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The Legendre transform $f^*(x) = \exp(x - 1)$ evaluated in numpy
        """
        return np.exp(x - 1)

    def _f_dash_given_log_torch(self, log_x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The derivative $f^'(x) = 1 + \log x$ evaluated in pytorch given $\log x$
        """
        return 1 + log_x

    def _f_dash_given_log_np(self, log_x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The derivative $f^'(x) = 1 + \log x$ evaluated in numpy given $\log x$
        """
        return 1 + log_x

    def calculate_naive_q_summands_given_log(self, log_p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        r"""
        Returns the q-side summand $f^*(f^'(p/q)) = p/q$ directly from $\log(p/q)$, bypassing the generic
        `f_conj(f_dash_given_log(.))` composition. The composition is `exp((1 + log_x) - 1)`; the +1/-1 round-trip
        loses the small deviation from 1 that the q-summand encodes near $\log(p/q) = 0$.

        Parameters
        ----------
        log_p_over_q
            An estimator for the log-probability ratio $\log \frac{p(x)}{q(x)}$

        Returns
        -------
        TensorOrNDArray
            The values of $p(x) / q(x)$
        """
        if isinstance(log_p_over_q, Tensor):
            return torch.exp(log_p_over_q)
        return np.exp(log_p_over_q)
