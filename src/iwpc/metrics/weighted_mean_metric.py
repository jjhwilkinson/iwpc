from typing import Tuple

import torch
from torch import Tensor

from .stat_metric import StatMetric
from ..types import TensorOrNDArray


class WeightedMeanMetric(StatMetric):
    """
    Metric which tracks a weighted mean and standard error of a scalar
    """
    def __init__(self):
        super().__init__(2)

    def update(self, weights: TensorOrNDArray, samples: TensorOrNDArray) -> None:
        """
        Updates the cumulative weighted mean and weight sums

        Parameters
        ----------
        weights
            The weight of each sample
        samples
            The value of each sample
        """
        super().update(weights, weights * samples)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """
        Returns
        -------
        Tuple[Tensor, Tensor]
            The cumulative weighted mean and standard error
        """
        return self.weighted_mean, self.weighted_stderr

    @property
    def weighted_mean(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            The cumulative weighted mean of the samples
        """
        means = self.means
        return means[1] / means[0]

    @property
    def weighted_stderr(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            The cumulative weighted standard error of the samples
        """
        means = self.means
        samples_mean_cov = self.cov / self.N

        ratio = means[1] / means[0]
        stderr = torch.abs(ratio) * torch.sqrt(
            samples_mean_cov[0, 0] / means[0]**2
            + samples_mean_cov[1, 1] / means[1]**2
            - 2 * samples_mean_cov[0, 1] / means[1] / means[0]
        )
        return stderr
