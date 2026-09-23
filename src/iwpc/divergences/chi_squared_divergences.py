import numpy as np
import torch
from numpy import ndarray
from torch import Tensor

from ..types import TensorOrNDArray
from .base import DifferentiableFDivergence


class PearsonChiSquaredDivergence(DifferentiableFDivergence):
    r"""
    Implementation of the Pearson $\chi^2$-divergence, $D_f(p, q) = \int (p - q)^2 / q$, with generating function
    $f(x) = (x - 1)^2$ in the convention of https://arxiv.org/abs/2405.06397 ($D_f(p, q) = E_q[f(p/q)]$).

    The derivative $f^'(x) = 2(x - 1)$ grows linearly in $p/q$, so a single region where $q$ under-populates $p$ can
    dominate a gradient built from $f^'$.
    """

    def __init__(self) -> None:
        """
        Initialises the divergence with display names "Pearson chi-squared" and "PChi2".
        """
        super().__init__("Pearson chi-squared", "PChi2")

    def _f_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The generating function $f(x) = (x - 1)^2$ evaluated in pytorch
        """
        return (x - 1) ** 2

    def _f_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The generating function $f(x) = (x - 1)^2$ evaluated in numpy
        """
        return (x - 1) ** 2

    def _f_conj_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The Legendre transform over $x \ge 0$: $f^*(u) = u + u^2 / 4$ for $u \ge -2$ and $-1$ below, evaluated
            in pytorch
        """
        return torch.where(x >= -2, x + x ** 2 / 4, torch.full_like(x, -1.))

    def _f_conj_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The Legendre transform over $x \ge 0$: $f^*(u) = u + u^2 / 4$ for $u \ge -2$ and $-1$ below, evaluated
            in numpy
        """
        return np.where(x >= -2, x + x ** 2 / 4, -1.)

    def _f_dash_given_log_torch(self, log_x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The derivative $f^'(x) = 2(x - 1)$ evaluated in pytorch given $\log x$, as $2\,\mathrm{expm1}(\log x)$
        """
        return 2 * torch.expm1(log_x)

    def _f_dash_given_log_np(self, log_x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The derivative $f^'(x) = 2(x - 1)$ evaluated in numpy given $\log x$, as $2\,\mathrm{expm1}(\log x)$
        """
        return 2 * np.expm1(log_x)

    def calculate_naive_q_summands_given_log(self, log_p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        r"""
        Returns the q-side summand $f^*(f^'(p/q)) = (p/q)^2 - 1$ directly from $\log(p/q)$, as
        $\mathrm{expm1}(2 \log(p/q))$, which keeps the small deviation from zero near $\log(p/q) = 0$.

        Parameters
        ----------
        log_p_over_q
            An estimator for the log-probability ratio $\log \frac{p(x)}{q(x)}$

        Returns
        -------
        TensorOrNDArray
            The values of $(p(x) / q(x))^2 - 1$
        """
        if isinstance(log_p_over_q, Tensor):
            return torch.expm1(2 * log_p_over_q)
        return np.expm1(2 * log_p_over_q)


class NeymanChiSquaredDivergence(DifferentiableFDivergence):
    r"""
    Implementation of the Neyman $\chi^2$-divergence, $D_f(p, q) = \int (p - q)^2 / p$, with generating function
    $f(x) = (x - 1)^2 / x$ in the convention of https://arxiv.org/abs/2405.06397 ($D_f(p, q) = E_q[f(p/q)]$).

    The derivative $f^'(x) = 1 - x^{-2}$ is bounded above by 1 but diverges as $p/q \to 0$, i.e. where $q$
    over-populates $p$.
    """

    def __init__(self) -> None:
        """
        Initialises the divergence with display names "Neyman chi-squared" and "NChi2".
        """
        super().__init__("Neyman chi-squared", "NChi2")

    def _f_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The generating function $f(x) = (x - 1)^2 / x$ evaluated in pytorch
        """
        return (x - 1) ** 2 / x

    def _f_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The generating function $f(x) = (x - 1)^2 / x$ evaluated in numpy
        """
        return (x - 1) ** 2 / x

    def _f_conj_torch(self, x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The Legendre transform $f^*(u) = 2 - 2\sqrt{1 - u}$, defined on $u < 1$ ($+\infty$ elsewhere), evaluated
            in pytorch
        """
        return torch.where(x < 1, 2 - 2 * torch.sqrt(torch.clamp(1 - x, min=0.)), torch.full_like(x, torch.inf))

    def _f_conj_np(self, x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The Legendre transform $f^*(u) = 2 - 2\sqrt{1 - u}$, defined on $u < 1$ ($+\infty$ elsewhere), evaluated
            in numpy
        """
        return np.where(x < 1, 2 - 2 * np.sqrt(np.clip(1 - x, 0., None)), np.inf)

    def _f_dash_given_log_torch(self, log_x: Tensor) -> Tensor:
        r"""
        Returns
        -------
        Tensor
            The derivative $f^'(x) = 1 - x^{-2}$ evaluated in pytorch given $\log x$, as
            $-\mathrm{expm1}(-2 \log x)$
        """
        return -torch.expm1(-2 * log_x)

    def _f_dash_given_log_np(self, log_x: ndarray) -> ndarray:
        r"""
        Returns
        -------
        ndarray
            The derivative $f^'(x) = 1 - x^{-2}$ evaluated in numpy given $\log x$, as $-\mathrm{expm1}(-2 \log x)$
        """
        return -np.expm1(-2 * log_x)

    def calculate_naive_q_summands_given_log(self, log_p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        r"""
        Returns the q-side summand $f^*(f^'(p/q)) = 2 - 2 q/p$ directly from $\log(p/q)$, as
        $-2\,\mathrm{expm1}(-\log(p/q))$, bypassing the square root of the generic composition.

        Parameters
        ----------
        log_p_over_q
            An estimator for the log-probability ratio $\log \frac{p(x)}{q(x)}$

        Returns
        -------
        TensorOrNDArray
            The values of $2 - 2 q(x) / p(x)$
        """
        if isinstance(log_p_over_q, Tensor):
            return -2 * torch.expm1(-log_p_over_q)
        return -2 * np.expm1(-log_p_over_q)
