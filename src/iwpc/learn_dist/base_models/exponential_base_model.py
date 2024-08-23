import numpy as np
import torch
from torch.distributions import Exponential

from iwpc.learn_dist.base_models.sampleable_base_model import SampleableBaseModel


class ExponentialBaseModel(SampleableBaseModel):
    def __init__(
        self,
        loc,
        scale,
    ):
        self.loc = torch.tensor(loc)
        self.scale = torch.tensor(scale)
        self.expon = Exponential(rate=torch.as_tensor(self.scale))
        super().__init__(1)

    def draw(self, num_samples):
        return self.expon.sample(torch.Size([num_samples, 1])).float() + self.loc

    def _log_prob(self, x):
        return torch.stack([
            self.expon.log_prob(sample) if sample > 0 else torch.tensor(-torch.inf)
            for sample in (x - self.loc)
        ])

    @classmethod
    def fit(cls, x, loc, weights=None):
        weights = np.ones(x.shape[0]) if weights is None else weights
        scale = weights.sum() / ((x - loc) * weights).sum()
        return cls(loc, scale)
