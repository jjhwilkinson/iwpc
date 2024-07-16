from abc import ABC, abstractmethod
from typing import Union, Callable, Tuple

import torch
from numpy import ndarray
from torch import Tensor

from ..types import TensorOrNDArray
from ..utils import split_by_mask


class DifferentiableFDivergence(ABC):
    """
    Abstract class describing the interface for a differentiable f-divergence for use in the iwpc divergence framework.
    The background of the required generating function, 'f', properties are described in the paper
    https://arxiv.org/abs/2405.06397. iwpc makes heavy use of both numpy and pytorch. As a result the generating
    function methods are required to be implemented in numpy and in pytorch, although these should not be used directly.
    Helper methods described below inspect the type of a given array, and use the respective numpy/pytorch
    implementations automatically.
    """
    def __init__(self, name: str, short_name: str):
        """
        Parameters
        ----------
        name
            The name of the divergence. e.g. 'KL-divergence'
        short_name
            The short name of the divergence. e.g. 'KL'
        """
        self.name = name
        self.short_name = short_name

    def _np_or_torch(self, x: TensorOrNDArray, torch_fn: Callable, np_fn: Callable) -> TensorOrNDArray:
        """
        Returns the value of torch_fn(x) or np_fn(x) depending on the type or x, Tensor or otherwise.

        Parameters
        ----------
        x
            An array
        torch_fn
            A function which accepts and returns a pytorch Tensor
        np_fn
            A function which accepts and returns a numpy NDArray

        Returns
        -------
        TensorOrNDArray
        """
        if isinstance(x, torch.Tensor):
            return torch_fn(x)
        return np_fn(x)

    @abstractmethod
    def _f_torch(self, x: Tensor) -> Tensor:
        """
        Implementation of the f-divergence generating function (https://arxiv.org/abs/2405.06397) in pytorch

        Parameters
        ----------
        x
            A pytorch tensor

        Returns
        -------
        Tensor
            The values $f(x)$
        """
        pass

    @abstractmethod
    def _f_np(self, x: ndarray) -> ndarray:
        """
        Implementation of the f-divergence generating function (https://arxiv.org/abs/2405.06397) in numpy

        Parameters
        ----------
        x
            A numpy array tensor

        Returns
        -------
        ndarray
            The values $f(x)$
        """
        pass

    @abstractmethod
    def _f_conj_torch(self, x: Tensor) -> Tensor:
        """
        Implementation of the f-divergence generating function legendre transform (https://arxiv.org/abs/2405.06397)
        in pytorch

        Parameters
        ----------
        x
            A pytorch tensor

        Returns
        -------
        Tensor
            The values $f^*(x)$
        """
        pass

    @abstractmethod
    def _f_conj_np(self, x: ndarray) -> ndarray:
        """
        Implementation of the f-divergence generating function legendre transform (https://arxiv.org/abs/2405.06397)
        in numpy

        Parameters
        ----------
        x
            A numpy array

        Returns
        -------
        ndarray
            The values $f^*(x)$
        """
        pass

    @abstractmethod
    def _f_dash_torch(self, x: Tensor) -> Tensor:
        """
        Implementation of the f-divergence generating function first derivative (https://arxiv.org/abs/2405.06397)
        in pytorch

        Parameters
        ----------
        x
            A pytorch tensor

        Returns
        -------
        Tensor
            The values $f^'(x)$
        """
        pass

    @abstractmethod
    def _f_dash_np(self, x: ndarray) -> ndarray:
        """
        Implementation of the f-divergence generating function first derivative (https://arxiv.org/abs/2405.06397)
        in numpy

        Parameters
        ----------
        x
            A numpy array

        Returns
        -------
        ndarray
            The values $f^'(x)$
        """
        pass

    def f(self, x: TensorOrNDArray) -> TensorOrNDArray:
        """
        Returns the value of the generating function evaluated on x. Automatically uses the numpy or pytorch
        implementation depending on the type of x.

        Parameters
        ----------
        x

        Returns
        -------
        The values of $f(x)$ as a Tensor of ndarray
        """
        return self._np_or_torch(x, self._f_torch, self._f_np)

    def f_conj(self, x: TensorOrNDArray) -> TensorOrNDArray:
        """
        Returns the value of the generating function legendre transform evaluated on x. Automatically uses the numpy or
        pytorch implementation depending on the type of x.

        Parameters
        ----------
        x

        Returns
        -------
        The values of $f^*(x)$ as a Tensor of ndarray
        """
        return self._np_or_torch(x, self._f_conj_torch, self._f_conj_np)

    def f_dash(self, x: TensorOrNDArray) -> TensorOrNDArray:
        """
        Returns the value of the generating function first derivative evaluated on x. Automatically uses the numpy or
        pytorch implementation depending on the type of x.

        Parameters
        ----------
        x

        Returns
        -------
        The values of $f^'(x)$ as a Tensor of ndarray
        """
        return self._np_or_torch(x, self._f_dash_torch, self._f_dash_np)

    def calculate_naive_p_values(self, p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        """
        Evaluates the function in the expectation value over the distribution p in the naive representation of the
        f-divergence (https://arxiv.org/abs/2405.06397)

        Parameters
        ----------
        p_over_q
            An estimator for the likelihood ratio of $\frac{p(x)}{q(x)}$

        Returns
        -------
        $f^'\left(\frac{p(x)}{q(x)}\right)$ as per the notation in (https://arxiv.org/abs/2405.06397)
        """
        return self.f_dash(p_over_q)

    def calculate_naive_q_values(self, p_over_q: TensorOrNDArray) -> TensorOrNDArray:
        """
        Evaluates the function in the expectation value over the distribution q in the naive representation of the
        f-divergence (https://arxiv.org/abs/2405.06397)

        Parameters
        ----------
        p_over_q
            An estimator for the likelihood ratio of $\frac{p(x)}{q(x)}$

        Returns
        -------
        $f^*\left(f^'\left(\frac{p(x)}{q(x)}\right)\right)$ as per the notation in (https://arxiv.org/abs/2405.06397)
        """
        return self.f_conj(self.f_dash(p_over_q))

    def calculate_naive_rep_values_by_label(
        self,
        p_over_q: TensorOrNDArray,
        label: TensorOrNDArray,
    ) -> Tuple[TensorOrNDArray, TensorOrNDArray]:
        """
        Calculates the two set of values averaged over in the naive representation of the f-divergence. Samples from
        the distribution p are identified with the label 'False' and samples from q are identified with the label 'True'.

        Parameters
        ----------
        p_over_q
            An estimator for the likelihood ratio of $\frac{p(x)}{q(x)}$
        label
            An array labelling samples in p_over_q from p with 'False' and samples from q are identified with the label
            'True'


        Returns
        -------
        Tuple[TensorOrNDArray, TensorOrNDArray]
            The list of values to be averaged over in the naive representation of the f-divergence
        """
        (samples_from_q,), (samples_from_p,) = split_by_mask(label, p_over_q)
        return self.calculate_naive_p_values(samples_from_p), self.calculate_naive_q_values(samples_from_q)

    def naive_estimate(
        self,
        p_over_q: TensorOrNDArray,
        label: TensorOrNDArray,
        weights: TensorOrNDArray
    ) -> TensorOrNDArray:
        """
        Calculates an estimator for lower bound on the f-divergence between p and q given an estimate of the values of
        the likelihood ratio $\frac{p(x)}{q(x)}$. Used in the divergence training loop and does not provide an estimate
        of the error on the estimate. You likely want to use a Df accumulator for analysis instead

        Parameters
        ----------
        p_over_q
            An estimate for the likelihood ratio $\frac{p(x)}{q(x)}$
        label
            An array labelling samples in p_over_q from p with 'False' and samples from q are identified with the label
            'True'
        weights
            The weight associated with each sample. Note it is assumed that the global average weight has been
            normalised to 1

        Returns
        -------
        TensorOrNDArray
            A scalar value providing an estimate of a lower bound of $D_f(p, q)$
        """
        p_values, q_values = self.calculate_naive_rep_values_by_label(p_over_q, label)
        (q_weights,), (p_weights,) = split_by_mask(label, weights)

        Df_hat = 0.0
        if len(p_values) > 0:
            Df_hat += (p_weights * p_values).mean()
        if len(q_values) > 0:
            Df_hat -= (q_weights * q_values).mean()

        return Df_hat
