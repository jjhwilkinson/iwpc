from typing import List, Optional

import numpy as np
from matplotlib import pyplot as plt
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult

from .binned_stat_accumulator import BinnedStatAccumulator
from ..scalars.scalar import Scalar
from ..stat_utils import normalised_weight_sum_uncertainty
from ..utils import bin_centers


class HistogramAccumulator(BinnedStatAccumulator):
    """
    Basic implementation of a D-dimensional weighted histogram that can be updated as new samples become available
    """

    def __init__(self, bins: List[NDArray], bin_labels=None):
        """
        Parameters
        ----------
        bins
            A list of D bin arrays. Each bin array must contain regularly spaced bins
        bin_labels
            An optional list of bin labels used for plotting
        """
        assert bin_labels is None or len(bin_labels) == len(bins)
        super().__init__(1, bins)
        self.bin_labels = bin_labels

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
        return self.weight_sum_hist / self.weight_sum_hist.sum()

    @property
    def weight_sum_stderr_hist(self) -> NDArray:
        return np.sqrt(self.sq_sum_hist[0])

    @property
    def normalised_weight_sum_stderr_hist(self) -> NDArray:
        return normalised_weight_sum_uncertainty(self.sum_hist[0], np.sqrt(self.sq_sum_hist[0, 0]))

    def plot(self) -> None:
        """
        Plots the contents of the weighted histogram. Implemented for 1D and 2D histograms
        Parameters
        """
        if len(self.bins) > 2:
            raise NotImplementedError()

        plt.figure()
        if len(self.bins) == 1:
            bins, = self.bins
            plt.plot(bin_centers(bins), self.weight_sum_hist)
            if self.bin_labels is not None:
                plt.xlabel(self.bin_labels[0])
            plt.ylabel('Weight Sum')
        elif len(self.bins) == 2:
            bins1, bins2 = self.bins
            plt.xlabel(self.bin_labels[0])
            plt.ylabel(self.bin_labels[1])
            plt.imshow(
                self.weight_sum_hist,
                extent=(
                    bins2[0],
                    bins2[-1],
                    bins1[0],
                    bins1[-1],
                ),
                origin='lower',
                interpolation='none',
                aspect='auto',
            )

    @classmethod
    def from_scalars(cls, scalars: List[Scalar]) -> "HistogramAccumulator":
        """
        Creates an instance from a list of scalars, populating the bins and axis labels from the scalars

        Parameters
        ----------
        scalars
            A list of Scalar instances

        Returns
        -------
        HistogramAccumulator
        """
        return cls(bins=[scalar.bins for scalar in scalars], bin_labels=[scalar.latex_label for scalar in scalars])
