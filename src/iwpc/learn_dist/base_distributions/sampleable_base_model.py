import copy
from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
from scipy.special import logsumexp

rng = np.random.Generator(np.random.PCG64())


class SamplableBaseModel(ABC):
    """
    Base class for finite measures on R^D. Provides the interface for sampling and getting the log probability of points
    """
    def __init__(self, dimension: int, total_volume: float = 1.0, *args: Any, **kwargs: Any):
        """
        Parameters
        ----------
        dimension
            The dimension of the measure space
        total_volume
            A global constant scaling the volume of the whole space. Typically corresponds to the total volume of the
            space assuming the standard implementation is a normalised measure (total volume is 1)
        """
        self.dimension = dimension
        self.total_volume = total_volume

    @abstractmethod
    def draw(self, num_samples: int) -> np.ndarray:
        """
        Draw samples from the measure
        
        Parameters
        ----------
        num_samples
            The number of samples to draw
        
        Returns
        -------
        np.ndarray
            The samples with shape (num_samples, self.dimension)
        """
        pass

    @abstractmethod
    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the log probability of the samples
        
        Parameters
        ----------
        x
            A numpy array of shape (num_samples, dimension)
        
        Returns
        -------
        np.ndarray
            The log probability of the samples assuming self.total_volume=1
        """

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the log probability of the samples scaled to the total volume of the space
        
        Parameters
        ----------
        x
            A numpy array of shape (num_samples, dimension)
        
        Returns
        -------
        np.ndarray
            The log probability of the samples scaled to the total volume of the space
        """
        if x.shape[0] == 0:
            return np.zeros((0, self.dimension))
        return np.log(self.total_volume) + self._log_prob(x)

    @classmethod
    def fit(cls, *args, **kwargs) -> "SamplableBaseModel":
        """
        Fits the base model and instantiates a SamplableBaseModel instance. Signature left very loose to allow for
        generic fitting arguments
        
        Parameters
        ----------
        args
        kwargs
        
        Returns
        -------
        SamplableBaseModel
            The fitted SamplableBaseModel instance
        """
        raise NotImplementedError()

    def __and__(self, other: "SamplableBaseModel") -> "ConcatenatedBaseModel":
        """
        Syntactic sugar to construct a SamplableBaseModel that produced samples of dimension self.dimension + other.dimension
        by concatenating independent samples drawn from this model and the other
        """
        if isinstance(other, ConcatenatedBaseModel):
            return ConcatenatedBaseModel([self] + other.models)
        return ConcatenatedBaseModel([self, other])

    def __add__(self, other: "SamplableBaseModel") -> "MixtureBaseModel":
        """
        Syntactic sugar to construct a SamplableBaseModel that produced samples by drawn from either self and other
        with probability given by the relative size of self.total_volume and other.total_volume
        """
        if isinstance(other, MixtureBaseModel):
            return other + self
        return MixtureBaseModel([self, other])

    def __rmul__(self, other: float | np.ndarray) -> "SamplableBaseModel":
        """
        Syntactic sugar to construct a copy of this SamplableBaseModel with the total volume scaled by other
        """
        if not (isinstance(other, float) or (isinstance(other, np.ndarray) and other.ndim == 0)):
            raise TypeError(f"Cannot multiply {self} with {other}")

        copy_ = copy.deepcopy(self)
        copy_.total_volume = other * self.total_volume
        return copy_


class ConcatenatedBaseModel(SamplableBaseModel):
    """
    Utility for constructing a SamplableBaseModel that produces samples by sampling a sequence of sub-models and
    concatenating the sub-samples
    """
    def __init__(self, models: List[SamplableBaseModel]):
        """
        Parameters
        ----------
        models
            The list of SamplableBaseModel sub-models
        """
        self.model_dimensions = [model.dimension for model in models]
        self.models = models
        super().__init__(sum(self.model_dimensions))

    def draw(self, num_samples: int) -> np.ndarray:
        """
        Parameters
        ----------
        num_samples
            The number of samples to draw

        Returns
        -------
        np.ndarray
            The samples with shape (num_samples, sum(m.dimension for m in self.models))
        """
        return np.concat([model.draw(num_samples) for model in self.models], axis=1)

    def _log_prob(self, x):
        """
        Returns the log probability of the samples

        Parameters
        ----------
        x
            A numpy array of shape (num_samples, dimension)

        Returns
        -------
        np.ndarray
            The log probability of the samples assuming samples are drawn independently from each sub-model
        """
        if x.shape[1] != self.dimension:
            raise ValueError(f"Sample dimension {x.shape[1]} does not match model dimension {self.dimension}")

        cum_dims = np.cumsum([0] + self.model_dimensions)
        return sum([model.log_prob(x[..., low:high]) for model, low, high in zip(self.models, cum_dims[:-1], cum_dims[1:])])

    def __and__(self, other: "SamplableBaseModel") -> "ConcatenatedBaseModel":
        """
        Syntactic sugar to construct a SamplableBaseModel that produced samples of dimension self.dimension + other.dimension
        by concatenating independent samples drawn from this model and the other
        """
        if isinstance(other, ConcatenatedBaseModel):
            return ConcatenatedBaseModel(self.models + other.models)
        return ConcatenatedBaseModel(self.models + [other])


class MixtureBaseModel(SamplableBaseModel):
    """
    Utility for constructing a SamplableBaseModel that produces samples by drawing from one of its sub-models with
    relative probability given by the normalised ratio of each sub-models total_volume
    """
    def __init__(self, models: List[SamplableBaseModel]):
        """
        Parameters
        ----------
        models
            The list of SamplableBaseModel sub-models
        """
        assert all(model.dimension == models[0].dimension for model in models)
        super().__init__(models[0].dimension)

        self.models = models
        self.fracs = np.array([model.total_volume for model in models])
        self.fracs = self.fracs / self.fracs.sum()

    def draw(self, num_samples: int) -> np.ndarray:
        """
        For each requested sample, a sub-model is chosen with probability given by the normalised ratio of each
        sub-models total_volume and a sample is drawn

        Parameters
        ----------
        num_samples
            The number of samples to draw

        Returns
        -------
        np.ndarray
            The samples with shape (num_samples, self.dimension)
        """
        num_samples_per_model = np.random.multinomial(num_samples, self.fracs)
        samples = np.concat([model.draw(size) for model, size in zip(self.models, num_samples_per_model)], axis=0)
        return samples[np.random.permutation(num_samples)]

    def _log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Returns the log probability of the samples

        Parameters
        ----------
        x
            A numpy array of shape (num_samples, self.dimension)

        Returns
        -------
        np.ndarray
            The log probability of the samples
        """
        probs = np.concat([model.log_prob(x) for model in self.models], axis=-1)
        return logsumexp(probs, axis=-1)

    def __add__(self, other: "SamplableBaseModel") -> "MixtureBaseModel":
        """
        Syntactic sugar to construct a SamplableBaseModel that produced samples by drawn from either self and other
        with probability given by the relative size of self.total_volume and other.total_volume. If 'other' is an instance
        of MixtureBaseModel, the list of sub-models are concatenated / un-curried
        """
        if isinstance(other, MixtureBaseModel):
            return MixtureBaseModel(self.models + other.models)
        return MixtureBaseModel(self.models + [other])
