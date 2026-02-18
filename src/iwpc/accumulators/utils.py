from typing import List, Optional, Any

import numpy as np
from numpy._typing import NDArray
from scipy.stats._binned_statistic import BinnedStatisticddResult, binned_statistic_dd


def construct_bin_number(
    samples: np.ndarray,
    bins: List[NDArray]
) -> NDArray:
    """
    Constructs the bin number of the samples within the specified bins. If the bins are regular a faster implementation
    is used but irregular bins are supported

    Parameters
    ----------
    samples
        A numpy array of (N, F)
    bins
        A list of F numpy array containing the bins in each dimension

    Returns
    -------
    NDArray
        A numpy array of shape (N, F) containing the bin index in each dimension
    """
    bin_indices = []
    for s, b in zip(samples.T, bins):
        if is_regular_bins(b):
            binds = np.ceil((s - b[0]) / (b[1] - b[0])).astype(int)
        else:
            binds = np.searchsorted(b, s)

        bin_indices.append(binds)

    return np.stack(bin_indices, axis=1)


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
    re-use these indices. If the provided bins are irregular, binned_statistic_dd is used directly

    Parameters
    ----------
    samples
        A list of samples of shape (N, F) where N denotes the batch dimension and F the feature dimension
    bins
        A list of F arrays denoting the bins in each dimension.

    Returns
    -------
    BinnedStatisticddResult
        BinnedStatisticddResult instance containing the bin indices of each sample in the format expected by
        binned_statistic_dd
    """
    bins_with_overflow = [np.concatenate([[2*b[0] - b[1]], b, [2*b[-1] - b[-2]]]) for b in bins]
    bin_numbers = np.ceil(construct_bin_number(samples, bins_with_overflow).T).astype(int)
    bin_numbers = np.clip(bin_numbers, 0, np.asarray([b.shape[0] for b in bins_with_overflow])[:, None])
    return BinnedStatisticddResult(
        None,
        bins_with_overflow,
        np.ravel_multi_index(
            bin_numbers,
            tuple(bins.shape[0] + 1 for bins in bins_with_overflow)
        ),
    )


def faster_binned_statistic_dd_without_overflow(
    samples: np.ndarray,
    values: np.ndarray | List[np.ndarray],
    bins: Optional[List[np.ndarray]] = None,
    statistic: Optional[str, Any] =None,
    binned_statistic_result: BinnedStatisticddResult | None = None,
):
    """
    A slightly faster implementation of binned_statistic_dd when using regular bins

    Parameters
    ----------
    samples
        A numpy array of samples with shape (N, F)
    values
        A numpy array of samples with shape (N, D) or a list of length D containing numpy arrays of shape (N,)
    bins
        See the binned_statistic_dd docstring
    statistic
        See the binned_statistic_dd docstring
    binned_statistic_result
        A previous binned_statistic_dd result that can be used to re-use indices

    Returns
    -------
    BinnedStatisticddResult
    """
    if binned_statistic_result is None:
        binned_statistic_result = construct_binned_statistic_result_regular_bins(samples, bins)

    result = binned_statistic_dd(
        samples,
        values,
        statistic=statistic,
        binned_statistic_result=binned_statistic_result,
    )

    slices = (( slice(1, -1),) * len(binned_statistic_result.bin_edges))
    if values is not None:
        slices = (slice(0, values.shape[0]),) + slices
    return result.statistic[slices], result


def is_regular_bins(bins: NDArray) -> bool:
    """
    Returns whether the bin spacings are equal

    Parameters
    ----------
    bins
        A numpy array of bin edges

    Returns
    -------
    bool
        Whether the bin spacings are equal
    """
    return np.allclose((bins - np.roll(bins, 1))[1:], bins[1] - bins[0])
