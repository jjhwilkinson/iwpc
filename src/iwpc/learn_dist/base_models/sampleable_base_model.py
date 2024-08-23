from abc import ABC, abstractmethod

import numpy as np
import torch

rng = np.random.Generator(np.random.PCG64())


class SampleableBaseModel(ABC):
    def __init__(self, dimension):
        self.dimension = dimension

    @abstractmethod
    def draw(self, num_samples):
        pass

    @abstractmethod
    def _log_prob(self, x):
        pass

    def log_prob(self, x):
        if x.shape[0] == 0:
            return torch.zeros((0,))
        return self._log_prob(torch.as_tensor(x)).float()

    @classmethod
    def fit(cls, x):
        raise NotImplementedError()

    def __and__(self, other):
        if isinstance(other, ConcatenatedBaseModel):
            return other & self
        return ConcatenatedBaseModel([self, other])

    def __add__(self, other):
        if isinstance(other, MixtureBaseModel):
            return other + self
        return MixtureBaseModel([self, other], [0.5, 0.5])


class ConcatenatedBaseModel(SampleableBaseModel):
    def __init__(self, models):
        self.model_dimensions = [model.dimension for model in models]
        self.models = models
        super().__init__(sum(self.model_dimensions))

    def draw(self, num_samples):
        return torch.concat([model.draw(num_samples) for model in self.models], dim=1)

    def _log_prob(self, x):
        cum_dims = np.cumsum([0] + self.model_dimensions)
        return sum([model.log_prob(x[..., low:high]) for model, low, high in zip(self.models, cum_dims[:-1], cum_dims[1:])])

    def __and__(self, other):
        if isinstance(other, ConcatenatedBaseModel):
            return ConcatenatedBaseModel(self.models + other.models)
        return ConcatenatedBaseModel([self, other])


class MixtureBaseModel(SampleableBaseModel):
    def __init__(
        self,
        models,
        fracs,
    ):
        assert all(model.dimension == models[0].dimension for model in models)
        assert np.isclose(sum(fracs), 1.0).all()

        self.models = models
        self.fracs = np.asarray(fracs) / sum(fracs)
        super().__init__(models[0].dimension)

    def draw(self, num_samples):
        num_samples_per_model = rng.multinomial(num_samples, self.fracs)
        samples = torch.concat([model.draw(size) for model, size in zip(self.models, num_samples_per_model)])

        return samples[torch.randperm(num_samples)]

    def _log_prob(self, x):
        return torch.log(
            sum([frac * torch.exp(model.log_prob(x)) for frac, model in zip(self.fracs, self.models)])
        )

    def __add__(self, other):
        if isinstance(other, MixtureBaseModel):
            return MixtureBaseModel(self.models + other.models, np.concatenate([0.5 * self.fracs, 0.5 * other.fracs]))
        return MixtureBaseModel(self.models + [other], [0.5 * self.fracs] + [0.5])
