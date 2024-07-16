from typing import Tuple, Dict

import torch
from torch import Tensor

from .fdivergence_base import FDivergenceEstimator
from ..divergences import DifferentiableFDivergence
from ..metrics.weighted_mean_metric import WeightedMeanMetric
from ..models.utils import basic_model_factory
from ..types import Shape
from ..utils import split_by_mask


class NaiveVariationalFDivergenceEstimator(FDivergenceEstimator):
    """
    Contains the optimization routine for the naive variational representation of an f-divergence.
    """
    def _configure_metrics(self) -> None:
        """
        Initialises two metrics used to track the running mean of the two expectation values in Equation 7 of the paper
        https://arxiv.org/abs/2405.06397
        """
        self.val_p_accumulator = WeightedMeanMetric()
        self.val_q_accumulator = WeightedMeanMetric()
        self.val_Df = self.val_p_accumulator[0] - self.val_q_accumulator[0]
        self.val_Df_err = (self.val_p_accumulator[1] ** 2 + self.val_q_accumulator[1] ** 2) ** 0.5

    def _calculate_batch_loss(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Calculates the batch loss as described in Section 2 of the paper https://arxiv.org/abs/2405.06397

        Parameters
        ----------
        batch
            A batch containing (sample features, labels, weights)

        Returns
        -------
        Tensor
            The batch loss
        """
        x, labels, weights = batch
        labels = labels.squeeze().bool()

        log_p_over_q_hat = self.model(x)[:, 0]
        clipped_p_over_q = torch.exp(torch.clip(log_p_over_q_hat, -14, 14))
        return - self.divergence.naive_estimate(clipped_p_over_q, labels, weights)

    def _accumulate_validation_Df(self, batch: Tuple[Tensor, Tensor, Tensor]) -> None:
        """
        Calculates and updates the validation metrics tracking the global validation loss

        Parameters
        ----------
        batch
            A batch containing (sample features, labels, weights)
        """
        x, labels, weights = batch
        labels = labels.squeeze().bool()

        log_p_over_q_hat = self.model(x)[:, 0]
        clipped_p_over_q = torch.exp(torch.clip(log_p_over_q_hat, -14, 14))
        p_samples, q_samples = self.divergence.calculate_naive_rep_values_by_label(clipped_p_over_q, labels)
        (p_weights,), (q_weights,) = split_by_mask(labels == 0, weights)

        self.val_p_accumulator(p_weights, p_samples)
        self.val_q_accumulator(q_weights, q_samples)


class GenericNaiveVariationalFDivergenceEstimator(NaiveVariationalFDivergenceEstimator):
    """
    NaiveVariationalFDivergenceEstimator setup with a generic primitive function NN model that takes in inputs of shape
    input_shape and returns a single scalaer
    """
    def __init__(
        self,
        input_shape: Shape,
        divergence: DifferentiableFDivergence,
        model_factory_kwargs: Dict = None,
        **kwargs
    ):
        """
        Parameters
        ----------
        input_shape
            The shape of input samples without the batch dimension
        divergence
            A DifferentiableFDivergence
        model_factory_kwargs
            Any additional arguments to provide to the generic model factory. See docstring of basic_model_factory
        kwargs
            Any additional arguments passed to the super constructor
        """
        self.input_shape = input_shape
        self.model_factory_kwargs = model_factory_kwargs or {}
        self.kwargs = kwargs

        model = basic_model_factory(input_shape=self.input_shape, output_shape=1, **self.model_factory_kwargs)
        self.save_hyperparameters()
        super().__init__(model=model, divergence=divergence, **kwargs)
