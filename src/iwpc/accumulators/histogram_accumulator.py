from typing import List, Optional

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult

from .binned_stat_accumulator import BinnedStatAccumulator
from ..stat_utils import normalised_weight_sum_uncertainty


class HistogramAccumulator(BinnedStatAccumulator):
    """
    Basic implementation of a D-dimensional weighted histogram that can be updated as new samples become available
    """

    def __init__(self, bins: List[NDArray]):
        """
        Parameters
        ----------
        bins
            A list of D bin arrays. Each bin array must contain regularly spaced bins
        """
        super().__init__(1, bins)

    def update(
        self,
        samples: NDArray,
        weights: Optional[NDArray] = None,
        prev_binned_statistic_result: Optional[BinnedStatisticddResult] = None
    ) -> Optional[BinnedStatisticddResult]:
        """
        Updates the internal state to include the new samples in the histogram

        Parameters
        ----------
        samples
            A numpy array of shape (N, len(bins)) containing the binned features for each sample
        weights
            A list of length N containing the weight of each sample. If None, the weights are all set to 1.
        prev_binned_statistic_result
            A BinnedStatisticddResult object containing the indices of each samples' binned features for reuse in
            binned_statistic_dd calls

        Returns
        -------
        Optional[BinnedStatisticddResult]
            If the list of samples is not empty, returns a BinnedStatisticddResult object containing the indices of each
            samples' binned features for reuse in binned_statistic_dd calls
        """
        if isinstance(samples, list):
            samples = np.asarray(samples).T
        if weights is None:
            weights = np.ones(samples.shape[0])
        return super().update(
            samples,
            weights,
            prev_binned_statistic_result=prev_binned_statistic_result
        )

    @property
    def weight_sum_hist(self) -> NDArray:
        return self.sum_hist[0]

    @property
    def normalised_weight_sum_hist(self) -> NDArray:
        return self.sum_hist[0] / self.sum_hist[0].sum()

    @property
    def weight_sum_stderr_hist(self) -> NDArray:
        return np.sqrt(self.sq_sum_hist[0])

    @property
    def normalised_weight_sum_stderr_hist(self) -> NDArray:
        return normalised_weight_sum_uncertainty(self.sum_hist, np.sqrt(self.sq_sum_hist))
