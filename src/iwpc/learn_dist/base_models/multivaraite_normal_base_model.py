import numpy as np
import torch
from torch.distributions import MultivariateNormal

from iwpc.learn_dist.base_models.sampleable_base_model import SampleableBaseModel


class MultivariateNormalBaseModel(SampleableBaseModel):
    def __init__(
        self,
        means,
        cov,
    ):
        assert cov.shape == (means.shape[0], means.shape[0])
        self.means = means
        self.cov = cov
        self.norm = MultivariateNormal(torch.as_tensor(self.means), torch.as_tensor(self.cov))
        super().__init__(self.means.shape[0])

    def draw(self, num_samples):
        return self.norm.sample(torch.Size([num_samples])).float()

    def _log_prob(self, x):
        return torch.stack([self.norm.log_prob(sample) for sample in x])

    @classmethod
    def fit(cls, x, weights=None):
        if weights is None:
            weights = np.ones(x.shape[0])

        cov = np.cov(x.T, aweights=weights)
        return cls(
            np.sum(x * weights[:, None], axis=0) / weights.sum(),
            cov if cov.ndim == 2 else cov[None, None],
        )
