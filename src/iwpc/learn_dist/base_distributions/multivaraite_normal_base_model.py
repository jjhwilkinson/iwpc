from typing import Optional

import numpy as np
import torch
from torch.distributions import MultivariateNormal

from .sampleable_base_model import SamplableBaseModel


class MultivariateNormalBaseModel(SamplableBaseModel):
    """
    SamplableBaseModel represents a multivariate normal distribution
    """
    def __init__(
        self,
        means: np.ndarray,
        cov: np.ndarray,
    ):
        """
        Parameters
        ----------
        means
            The mean of the multivariate normal distribution
        cov
            The covariance matrix of the multivariate normal distribution
        """
        assert cov.shape == (means.shape[0], means.shape[0])
        self.means = means
        self.cov = cov
        self.norm = MultivariateNormal(torch.as_tensor(self.means), torch.as_tensor(self.cov))
        self.log_cov_det = np.log(np.linalg.det(self.cov))
        self.cov_inv = np.linalg.inv(self.cov)
        super().__init__(self.means.shape[0])

    def draw(self, num_samples: int) -> np.ndarray:
        """
        Draw num_samples samples from the multivariate normal distribution
        """
        return np.random.multivariate_normal(self.means, self.cov, size=(num_samples,))

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        The probability of observing the given samples
        """
        diffs = (x - self.means[None, :])
        chisq = np.einsum("ij,jk,ik", diffs, self.cov_inv, diffs)
        return - 0.5 * (self.dimension * np.log(2 * np.pi) + self.log_cov_det + chisq)

    @classmethod
    def fit(cls, x: np.ndarray, weights: Optional[np.ndarray] = None) -> "MultivariateNormalBaseModel":
        """
        Fits a multivariate normal distribution to the given samples
        """
        if weights is None:
            weights = np.ones(x.shape[0])

        cov = np.cov(x.T, aweights=weights)
        return cls(
            np.sum(x * weights[:, None], axis=0) / weights.sum(),
            cov if cov.ndim == 2 else cov[None, None],
        )
