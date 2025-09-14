from typing import List, Optional

import numpy as np
from scipy.stats import binned_statistic_dd

from iwpc.utils import bin_centers
from .sampleable_base_model import SamplableBaseModel


class HistogramBaseModel(SamplableBaseModel):
    """
    Base model that draws samples from an N-d histogram
    """
    def __init__(self, bins: List[np.ndarray], histogram: np.ndarray):
        """
        Parameters
        ----------
        bins
            The list of bin edges in each dimension
        histogram
            The weight sum in each bin. The histogram is automatically normalized to ensure the total volume is 1
        """
        super().__init__(len(bins))
        self.bins = bins
        self.bin_centers = [bin_centers(b) for b in bins]
        self.histogram = histogram / np.sum(histogram)
        self.p = self.histogram.flatten()

    def draw(self, num_samples: int) -> np.ndarray:
        """
        Draw a sample from the histogram. Note that samples are smeared to be uniform within each bin
        """
        flat_indices = np.random.choice(self.p.shape[0], size=num_samples, p=self.p)
        indices = np.unravel_index(flat_indices, [b.shape[0] for b in self.bin_centers])
        return np.stack([
            b[idx] + (b[1:] - b[:-1])[idx] * np.random.random(size=num_samples)
            for b, idx in zip(self.bins, indices)
        ], axis=-1)

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log probability of each sample given by self.histogram
        """
        idxs = []
        out_of_bounds_mask = np.zeros(x.shape[0], dtype=bool)
        for bins, vals in zip(self.bins, x.T):
            idxs.append(np.digitize(vals, bins) - 1)
            out_of_bounds_mask = out_of_bounds_mask | (idxs[-1] < 0) | (idxs[-1] > bins.shape[0] - 2)

        probability = np.log(self.histogram[*(np.clip(idx, 0, b.shape[0] - 2) for idx, b in zip(idxs, self.bins))])
        probability[out_of_bounds_mask] = -np.inf

        return probability

    @classmethod
    def fit(
        cls,
        x: np.ndarray,
        bins: List[np.ndarray],
        weights: Optional[np.ndarray] = None,
    ) -> "HistogramBaseModel":
        """
        Uses scipy.stats.binned_statistic_dd to compute a histogram of the data and instantiates an instance of
        HistogramBaseModel

        Parameters
        ----------
        x
            Samples of dimension D
        bins
            A list of length D giving the bin edges in each dimension
        weights
            A list of weights for each sample
        """
        result = binned_statistic_dd(x, weights, statistic='count' if weights is None else 'sum', bins=bins)
        stat = result.statistic
        stat = stat / stat.sum()
        return HistogramBaseModel(bins, stat)
