from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np

from ..divergences import DifferentiableFDivergence
from ..metrics.weighted_mean_metric import WeightedMeanMetric
from ..utils import split_by_mask


class DfAccumulator(ABC):
    """
    Abstract class for objects that track a running divergence between two distributions
    """
    def __init__(
        self,
        divergence: DifferentiableFDivergence,
        clip_log_p_over_q: Optional[Tuple[float, float]] = (-14, 14),
    ):
        """
        Parameters
        ----------
        divergence
            A DifferentiableFDivergence object
        clip_log_p_over_q
            If provided, clips the likelihood ratios passed into 'update' such that the log likelihood ratio is between
            the two bounds provided. Used to provide numerical stability as sometimes floating point errors can cause
            issues at extreme values
        """
        self.divergence = divergence
        self.clip_log_p_over_q = clip_log_p_over_q

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """
        Update the internal state of the accumulator with new values. May vary from implementation to implementation

        Parameters
        ----------
        args
        kwargs
        """
        pass

    @property
    @abstractmethod
    def accumulated_df(self) -> float:
        """
        Returns
        -------
        float
            The total f-divergence over all the data seen so far
        """
        pass

    @property
    @abstractmethod
    def accumulated_df_stderr(self) -> float:
        """
        Returns
        -------
        float
            The f-divergence sample mean standard deviation over all the data seen so far
        """
        pass

    @property
    def sig(self) -> float:
        """
        Returns
        -------
        float
            The significance of the f-divergence over all the data seen so far
        """
        return self.accumulated_df / self.accumulated_df_stderr

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            A summary of the f-divergence over the data seen so far
        """
        return f"{self.divergence.short_name} divergence >= {self.accumulated_df} +- {self.accumulated_df_stderr} ({self.sig:.2f})"


class LabeledBinaryNaiveDfAccumulator(DfAccumulator):
    """
    DfAccumulator that accumulates the naive f-divergence between two distributions p & q assuming independent samples
    from each
    """
    def __init__(
        self,
        divergence: DifferentiableFDivergence,
        p_name: str = 'p',
        q_name: str = 'q',
        clip_log_p_over_q: Optional[Tuple[float, float]] = (-14, 14),
    ):
        """
        Parameters
        ----------
        divergence
            A DifferentiableFDivergence object
        p_name
            The name of the first distribution (label 0)
        q_name
            The name of the second distribution (label 1)
        clip_log_p_over_q
            If provided, clips the likelihood ratios passed into 'update' such that the log likelihood ratio is between
            the two bounds provided. Used to provide numerical stability as sometimes floating point errors can cause
            issues at extreme values
        """
        super().__init__(divergence=divergence, clip_log_p_over_q=clip_log_p_over_q)
        self.p_name = p_name
        self.q_name = q_name

        self.p_accumulator = WeightedMeanMetric()
        self.q_accumulator = WeightedMeanMetric()

    # @profile
    def update(
        self,
        p_over_q,
        labels,
        weights,
    ) -> None:
        """
        Updates the internal state required to calculate the running f-divergence between p and q

        Parameters
        ----------
        p_over_q
            An estimate of the likelihood ratio $\frac{p(x)}{q(x)}$ for a number of samples
        labels
            The sample labels. 0 corresponds to distribution p and 1 corresponds to distribution q
        weights
            The weight of each sample

        """
        if self.clip_log_p_over_q:
            p_over_q = np.exp(np.clip(np.log(p_over_q), *self.clip_log_p_over_q))

        p_values, q_values = self.divergence.calculate_naive_rep_values_by_label(
            p_over_q,
            labels,
        )
        (p_weights,), (q_weights,) = split_by_mask(labels == 0, weights)
        self.p_accumulator.update(p_weights, p_values)
        self.q_accumulator.update(q_weights, q_values)

    @property
    def accumulated_df(self) -> float:
        """
        Returns
        -------
        float
            The total f-divergence over all the data seen so far
        """
        return self.p_accumulator.weighted_mean - self.q_accumulator.weighted_mean

    @property
    def accumulated_df_stderr(self) -> float:
        """
        Returns
        -------
        float
            The f-divergence sample mean standard deviation over all the data seen so far
        """
        return (self.p_accumulator.weighted_stderr**2 + self.q_accumulator.weighted_stderr**2) ** 0.5

    def __str__(self) -> str:
        """
        Returns
        -------
        str
            A summary of the f-divergence over the data seen so far
        """
        return f"{self.divergence.short_name}({self.p_name}, {self.q_name}) >= {self.accumulated_df} +- {self.accumulated_df_stderr} ({self.sig:.2f})"