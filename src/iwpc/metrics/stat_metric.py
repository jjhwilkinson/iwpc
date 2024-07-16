import numpy as np
import torch
from torch import Tensor
from torchmetrics import Metric

from ..types import TensorOrNDArray


class StatMetric(Metric):
    """
    Metric that tracks the sum and outer product sum of a list of features. Useful for computing the mean and covariance
    of the features
    """
    def __init__(self, ndim):
        """
        Parameters
        ----------
        ndim
            The size of the feature vector
        """
        super().__init__()
        self.ndim = ndim
        self.add_state("N", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("sums", default=torch.zeros(ndim), dist_reduce_fx="sum")
        self.add_state("outer_prod_sums", default=torch.zeros((ndim, ndim)), dist_reduce_fx="sum")

    def update(self, *arrs: TensorOrNDArray) -> None:
        """
        Updates the cumulative sums and outer product sums of the features

        Parameters
        ----------
        arrs
            A list of values, one for each feature. All arrays must have the same length
        """
        samples = torch.stack([torch.as_tensor(arr) for arr in arrs]).T
        self.N += samples.shape[0]
        self.sums += samples.sum(dim=0)
        self.outer_prod_sums += (samples[:, :, None] * samples[:, None, :]).sum(axis=0)

    def compute(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            A vector of the means of the features. Has length ndim
        """
        return self.means

    @property
    def means(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            A vector of the means of the features. Has length ndim
        """
        return self.sums / self.N

    @property
    def cov(self):
        """
        Returns
        -------
        Tensor
            A matrix of the covariance of the features. Has shape (ndim, ndim)
        """
        means = self.means
        return self.outer_prod_sums / self.N - means[:, np.newaxis] * means[np.newaxis, :]
