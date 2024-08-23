import numpy as np
import torch
from torch.distributions import Cauchy

from iwpc.learn_dist.base_models.sampleable_base_model import SampleableBaseModel


class CauchyBaseModel(SampleableBaseModel):
    def __init__(
        self,
        loc,
        fwhm,
    ):
        self.loc = loc
        self.fwhm = fwhm
        self.cauchy = Cauchy(torch.as_tensor(self.loc), torch.as_tensor(self.fwhm))
        super().__init__(1)

    def draw(self, num_samples):
        return self.cauchy.sample(torch.Size([num_samples, 1])).float()

    def _log_prob(self, x):
        return torch.stack([self.cauchy.log_prob(sample) for sample in x])

    @classmethod
    def fit(cls, x, weights=None):
        if weights is None:
            weights = np.ones(x.shape[0])
        loc = (x * weights).mean() / weights.mean()
        fwhm = np.sqrt(8 * np.log(2)) * x.std()

        return cls(loc, fwhm)
