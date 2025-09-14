import numpy as np

from .sampleable_base_model import SamplableBaseModel


class CauchyBaseModel(SamplableBaseModel):
    def __init__(
        self,
        loc: float,
        hwhm: float,
    ):
        """
        Parameters
        ----------
        loc
            The centre of the Cauchy distribution
        hwhm
            The half-width at half maximum of the Cauchy distribution
        """
        self.loc = loc
        self.hwhm = hwhm
        super().__init__(1)

    def draw(self, num_samples: int) -> np.ndarray:
        """
        Draw num_samples from the Cauchy distribution centered on self.loc and width given by self.hwhm
        """
        return np.random.standard_cauchy(size=(num_samples, 1)) * self.hwhm + self.loc

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the log probability of a cauchy distribution centered on self.loc and width self.hfhm producing each sample
        """
        return -np.log(np.pi * self.hwhm) - np.log(1 + ((x - self.loc) / self.hwhm)**2)

    @classmethod
    def fit(cls, x: np.ndarray, weights: Optional[np.ndarray] = None) -> "CauchyBaseModel":
        raise NotImplementedError("This is complicated, not sure how to do it safely")
