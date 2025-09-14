from typing import Optional

import numpy as np
import torch

from .sampleable_base_model import SamplableBaseModel


class ExponentialBaseModel(SamplableBaseModel):
    def __init__(
        self,
        loc: float,
        scale: float,
    ):
        """
        Parameters
        ----------
        loc
            The left-most edge of the exponential distribution
        scale
            The decay rate of the exponential distribution
        """
        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        super().__init__(1)

    def draw(self, num_samples: int) -> np.ndarray:
        """
        Draw num_samples samples from the exponential distribution
        """
        return self.loc + np.random.exponential(self.scale, size=(num_samples, 1))

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        The log probability of an exponential distribution with parameters given by self.loc and self.scale producing
        each sample
        """
        log_prob = np.log(self.scale) - self.scale * x
        log_prob[x < self.loc] = -np.inf
        return log_prob

    @classmethod
    def fit(cls, x: np.ndarray, loc: float, weights: Optional[np.ndarray] = None) -> "ExponentialBaseModel":
        """
        Fit an exponential distribution to the given data given the left-most edge of the distribution
        """
        weights = np.ones(x.shape[0]) if weights is None else weights
        scale = weights.sum() / ((x - loc) * weights).sum()
        return cls(loc, scale)
