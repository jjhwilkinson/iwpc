from typing import Optional, List, Union, Iterable

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult

from iwpc.accumulators.binned_stat_accumulator import BinnedStatAccumulator


class WeightedBinnedStatAccumulator(BinnedStatAccumulator):
    """
    Metric that tracks the weighted sum and outer product sum of a list of features within a set of bins
    """
    def __init__(self, num_statistics: int, bins: Iterable[NDArray]):
        """
        Parameters
        ----------
        num_statistics
            The number of statistic features to track. Note these are not the binned features, these are the features
            for which the sum and outer product sum are tracked.
        bins
            A list containing the bins in each binned dimension. The binned features are unrelated to the statistic
            features mentioned above
        """
        self.bins = list(bins)
        self.num_statistics = num_statistics
        self.combined_accumulator = BinnedStatAccumulator(num_statistics + 1, self.bins)

    def reset(self) -> None:
        """
        Resets internal state variables
        """
        self.combined_accumulator.reset()

    def update(
        self,
        samples: NDArray,
        values: Union[List[NDArray], NDArray],
        weights: NDArray | None = None,
        prev_binned_statistic_result: Optional[BinnedStatisticddResult] = None,
    ) -> Optional[BinnedStatisticddResult]:
        """
        Updates the internal state with the sums and outer product sums of the given samples

        Parameters
        ----------
        samples
            A numpy array of shape (N, len(bins)) containing the binned features for each sample
        values
            A numpy array of shape (N, num_statistics) containing the statistic features for each sample
        weights
            An optional numpy array of shape (N) containing the weights for each sample
        prev_binned_statistic_result
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls

        Returns
        -------
        Optional[BinnedStatisticddResult]
            If the list of samples is not empty, returns a BinnedStatisticddResult object containing the indices of each
            samples' binned features for reuse in binned_statistic_dd calls
        """
        if isinstance(values, list):
            values = np.stack(values)
        if values.ndim == 1:
            values = values[None, :]
        if weights is None:
            weights = np.ones(values.shape[1])

        return self.combined_accumulator.update(
            samples,
            np.concatenate([weights[None], values * weights[None]], axis=0),
            prev_binned_statistic_result=prev_binned_statistic_result
        )

    @property
    def weighted_sum_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the F weighted average values for
            each
            statistic feature in each bin
        """
        return self.combined_accumulator.sum_hist[1:]

    @property
    def weight_sum_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the F weighted average values for
            each
            statistic feature in each bin
        """
        return self.combined_accumulator.sum_hist[0]

    @property
    def weighted_mean_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the F weighted average values for
            each
            statistic feature in each bin
        """
        return self.weighted_sum_hist / self.weight_sum_hist[None]

    @property
    def weighted_mean_covariance_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the weighted covariance matrix
            of the F statistic features in each bin
        """
        identity = np.eye(self.num_statistics)
        identity = identity.reshape((self.num_statistics, self.num_statistics, *([1] * len(self.bins))))

        ratio_jacobian = np.concat([
            - (self.weighted_sum_hist / self.weight_sum_hist[None]**2)[:, None],
            identity / self.weight_sum_hist[None, None],
        ], axis=1)

        return np.einsum(
            'ij...,jk...,mk...->im...',
            ratio_jacobian,
            self.combined_accumulator.outer_product_sum_hist,
            ratio_jacobian
        )
