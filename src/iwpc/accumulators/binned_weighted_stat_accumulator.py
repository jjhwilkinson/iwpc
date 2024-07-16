from typing import Optional, List

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult

from .binned_stat_accumulator import BinnedStatAccumulator


class BinnedWeightedStatAccumulator(BinnedStatAccumulator):
    """
    Accumulator that can be used track the sum-of-weights, weighted sum of a feature, and the feature-weight outer
    product matrix within a set of bins. Also provides the weighted mean and weighted mean standard deviation in each
    bin
    """
    def __init__(self, bins: List[NDArray]):
        """
        Parameters
        ----------
        bins
            A list containing a number of regularly spaced bin arrays, one for each binned feature
        """
        super().__init__(2, bins)

    def update(
        self,
        samples: NDArray,
        values: NDArray,
        weights: NDArray,
        prev_binned_statistic_result: Optional[BinnedStatisticddResult] = None,
    ) -> Optional[BinnedStatisticddResult]:
        """
        Updates the internal state to account for the weighted sum and weighted outer product of the given samples and
        their weights

        Parameters
        ----------
        samples
            A numpy array of shape (N, B) where B is the number of binned features
        values
            A numpy array of size N
        weights
            A numpy array of size N containing the weight of each sample
        prev_binned_statistic_result
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls

        Returns
        -------
        Optional[BinnedStatisticddResult]
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls
        """
        return super().update(
            samples,
            [weights, weights * values],
            prev_binned_statistic_result=prev_binned_statistic_result
        )

    @property
    def sum_of_weights_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            an array of shape (len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the sum-of-weights in each bin
        """
        return self.sum_hist[0]

    @property
    def weighted_sum_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            an array of shape (len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the weighted sum of the feature in
            each bin
        """
        return self.sum_hist[1]

    @property
    def sum_of_sq_weights_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            an array of shape (len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the sum-of-(weights^2) in each bin
        """
        return self.sq_sum_hist[0, 0]

    @property
    def weighted_mean_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            an array of shape (len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the weighted mean of the feature in
            each bin
        """
        return self.weighted_sum_hist / self.sum_of_weights_hist

    @property
    def weighted_stderr_hist(self) -> NDArray:
        """
        Returns
        -------
        Tensor
            The cumulative weighted standard error of the samples
        """
        unweighted_mean_hist = self.mean_hist
        unweighted_sample_mean_cov_hist = self.cov_hist / self.count_hist
        weighted_mean_hist = self.weighted_mean_hist

        weighted_mean_stderr_hist = np.abs(weighted_mean_hist) * np.sqrt(
            unweighted_sample_mean_cov_hist[0, 0] / unweighted_mean_hist[0]**2
            + unweighted_sample_mean_cov_hist[1, 1] / unweighted_mean_hist[1]**2
            - 2 * unweighted_sample_mean_cov_hist[0, 1] / unweighted_mean_hist[1] / unweighted_mean_hist[0]
        )
        return weighted_mean_stderr_hist


class WeightedHistogramAccumulator(BinnedWeightedStatAccumulator):
    def update(
        self,
        samples: NDArray,
        weights: NDArray,
        prev_binned_statistic_result: Optional[BinnedStatisticddResult] = None,
    ) -> Optional[BinnedStatisticddResult]:
        """
        Updates the internal state to account for the weighted sum and weighted outer product of the given samples and
        their weights

        Parameters
        ----------
        samples
            A numpy array of shape (N, B) where B is the number of binned features
        weights
            A numpy array of size N containing the weight of each sample
        prev_binned_statistic_result
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls

        Returns
        -------
        Optional[BinnedStatisticddResult]
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls
        """
        return super().update(
            samples,
            [weights, weights],
            prev_binned_statistic_result=prev_binned_statistic_result
        )