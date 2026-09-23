import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ..types import TensorOrNDArray
from .base import DifferentiableFDivergence


class ReverseKLDivergence(DifferentiableFDivergence):
    r"""
    Implementation of the reverse Kullback-Leibler divergence, $D_f(p, q) = \mathrm{KL}(q \| p)$, with generating
    function $f(x) = -\log x$ in the convention of https://arxiv.org/abs/2405.06397 ($D_f(p, q) = E_q[f(p/q)]$).

    The derivative $f^'(x) = -1/x$ diverges as $p/q \to 0$.
    """

    def __init__(self) -> None:
        """
        Initialises the divergence with display names "Reverse Kullback-Leibler" and "RKL".
        """
        super().__init__("Reverse Kullback-Leibler", "RKL")

    def _f_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The generating function $f(x) = -\log x$ evaluated in pytorch
        """
        return -torch.log(x)

    def _f_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The generating function $f(x) = -\log x$ evaluated in numpy
        """
        return -np.log(x)

    def _f_conj_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The Legendre transform $f^*(u) = -1 - \log(-u)$, defined on $u < 0$, evaluated in pytorch
        """
        return -1 - torch.log(-x)

    def _f_conj_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The Legendre transform $f^*(u) = -1 - \log(-u)$, defined on $u < 0$, evaluated in numpy
        """
        return -1 - np.log(-x)

    def _f_dash_given_log_torch(self, log_x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The derivative $f^'(x) = -1/x$ evaluated in pytorch given $\log x$
        """
        return -torch.exp(-log_x)

    def _f_dash_given_log_np(self, log_x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The derivative $f^'(x) = -1/x$ evaluated in numpy given $\log x$
        """
        return -np.exp(-log_x)

    def calculate_naive_q_summands_given_log(self, log_p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        r"""
        Returns the q-side summand $f^*(f^'(p/q)) = \log(p/q) - 1$ directly from $\log(p/q)$, bypassing the
        exponential round trip of the generic composition.

        Parameters
        ----------
        log_p_over_q
            An estimator for the log-probability ratio $\log \frac{p(x)}{q(x)}$

        Returns
        -------
        TensorOrNDArray
            The values of $\log(p(x) / q(x)) - 1$
        """
        return log_p_over_q - 1
