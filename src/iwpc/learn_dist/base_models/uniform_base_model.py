import torch
from torch.distributions import Uniform

from .sampleable_base_model import SampleableBaseModel


class UniformBaseModel(SampleableBaseModel):
    def __init__(self, low, high):
        super().__init__(1)
        self.low = low
        self.high = high
        self.uniform = Uniform(low, high)

    def draw(self, num_samples):
        return self.uniform.sample(torch.Size([num_samples, 1])).float()

    def _log_prob(self, x):
        return torch.stack([self.uniform.log_prob(sample).squeeze() for sample in x])

    @classmethod
    def fit(cls, x, weights=None):
        return UniformBaseModel(x.min(), x.max())
