import numpy as np

from .sampleable_base_model import SamplableBaseModel


class UniformBaseModel(SamplableBaseModel):
    """
    Base class for uniform distributions
    """
    def __init__(self, low: float, high: float):
        """
        Parameters
        ----------
        low
            The lower bound of the uniform distribution
        high
            The upper bound of the uniform distribution
        """
        super().__init__(1)
        assert low <= high

        self.low = low
        self.high = high

    def draw(self, num_samples: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=(num_samples, 1))

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        return np.full(x.shape[0], 1 / (self.high - self.low))

    @classmethod
    def fit(cls, x: np.ndarray) -> "UniformBaseModel":
        """
        Fits a uniform distribution to the data using the min and max values
        """
        return UniformBaseModel(x.min(), x.max())
