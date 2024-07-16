from typing import Optional, List, Union

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult, binned_statistic_dd

from .utils import construct_binned_statistic_result_regular_bins, faster_binned_statistic_dd_without_overflow


class BinnedStatAccumulator:
    """
    Metric that tracks the sum and outer product sum of a list of features within a set of bins
    """
    def __init__(self, num_statistics: int, bins: List[NDArray]):
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
        if not (isinstance(bins, list) or isinstance(bins, tuple)):
            bins = [bins]
        super().__init__()
        self.num_statistics = num_statistics
        self.bins = bins
        self.hist_shape = tuple(x.shape[0]-1 for x in self.bins)
        self.count_hist = np.zeros((self.num_statistics,) + self.hist_shape)
        self.sum_hist = np.zeros((self.num_statistics,) + self.hist_shape)
        self.sq_sum_hist = np.zeros((self.num_statistics, self.num_statistics) + self.hist_shape)

    def update(
        self,
        samples: NDArray,
        values: Union[List[NDArray], NDArray],
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
        if values.shape[-1] == 0:
            return

        if prev_binned_statistic_result is None:
            prev_binned_statistic_result = construct_binned_statistic_result_regular_bins(samples, self.bins)

        self.count_hist += faster_binned_statistic_dd_without_overflow(
            samples,
            values,
            statistic='count',
            binned_statistic_result=prev_binned_statistic_result,
        )[0]

        self.sum_hist += faster_binned_statistic_dd_without_overflow(
            samples,
            values,
            statistic='sum',
            binned_statistic_result=prev_binned_statistic_result,
        )[0]

        sq_mat = values[:, np.newaxis] * values[np.newaxis, :]
        sq_mat_result = faster_binned_statistic_dd_without_overflow(
            samples,
            sq_mat.reshape((-1, sq_mat.shape[-1])),
            statistic='sum',
            binned_statistic_result=prev_binned_statistic_result,
        )
        self.sq_sum_hist += sq_mat_result[0].reshape(self.sq_sum_hist.shape)
        return prev_binned_statistic_result

    @property
    def mean_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the F average values for each
            statistic feature in each bin
        """
        return self.sum_hist / self.count_hist

    @property
    def cov_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An array of shape (F, F, len(bins[0]) - 1, len(bins[1]) - 1, ...) containing the covariance matrix of the F
            statistic features in each bin
        """
        mean_hist = self.mean_hist
        return self.sq_sum_hist / self.count_hist - mean_hist[:, np.newaxis] * mean_hist[np.newaxis, :]
