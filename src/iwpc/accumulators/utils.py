from typing import List

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult, binned_statistic_dd


def construct_bin_number_regular_bins(
    samples: NDArray,
    bins: List[NDArray],
) -> NDArray:
    """
    The scipy implementation of binned_statistic_dd does not seem to exploit regular bin sizing for faster bin indexing.
    This function calculates bin index of each sample assuming regular bin sizing. Note the indices may well overflow
    the shape of the bins on either side if any sample is outside the bin range. These indices should be handled
    downstream of this function

    Parameters
    ----------
    samples
        A list of samples of shape (N, F) where N denotes the batch dimension and F the feature dimension
    bins
        A list of F arrays denoting the bins in each dimension. Bins are assumed to be regular and an error is raised
        if they are not

    Returns
    -------
    NDArray
        An array of shape (N, F) giving the index of each sample within each of the F bins
    """
    assert all(np.allclose(b[1:] - b[:-1], b[1] - b[0]) for b in bins)
    bin_starts = np.asarray([b[0] for b in bins])
    bin_deltas = np.asarray([b[1] - b[0] for b in bins])
    return (samples - bin_starts[None, :]) / bin_deltas[None, :]


def construct_binned_statistic_result_regular_bins(
    samples: NDArray,
    bins: List[NDArray],
) -> BinnedStatisticddResult:
    """
    The scipy implementation of binned_statistic_dd does not seem to exploit regular bin sizing for faster bin indexing.
    This function constructs an instance of BinnedStatisticddResult containing the bin indices of each sample calculated
    rapidly assuming regular bin sizes. This object may be passed to binned_statistic_dd calls so the function may
    re-use these indices

    Parameters
    ----------
    samples
        A list of samples of shape (N, F) where N denotes the batch dimension and F the feature dimension
    bins
        A list of F arrays denoting the bins in each dimension. Bins are assumed to be regular and an error is raised
        if they are not

    Returns
    -------
    BinnedStatisticddResult
        BinnedStatisticddResult instance containing the bin indices of each sample in the format expected by
        binned_statistic_dd
    """
    bins_with_overflow = [np.concat([[2*b[0] - b[1]], b, [2*b[-1] - b[-2]]]) for b in bins]
    bin_numbers = np.ceil(construct_bin_number_regular_bins(samples, bins_with_overflow).T).astype(int)
    bin_numbers = np.clip(bin_numbers, 0, np.asarray([b.shape[0] for b in bins_with_overflow])[:, None])
    return BinnedStatisticddResult(
        None,
        bins_with_overflow,
        np.ravel_multi_index(
            bin_numbers,
            tuple(bins.shape[0] + 1 for bins in bins_with_overflow)
        ),
    )


def faster_binned_statistic_dd_without_overflow(samples, values, bins=None, statistic=None, binned_statistic_result=None):
    if binned_statistic_result is None:
        binned_statistic_result = construct_binned_statistic_result_regular_bins(samples, bins)

    result = binned_statistic_dd(
        samples,
        values,
        statistic=statistic,
        binned_statistic_result=binned_statistic_result,
    )

    slices = (slice(0, values.shape[0]),) + (( slice(1, -1),) * len(binned_statistic_result.bin_edges))
    return result.statistic[slices], result
