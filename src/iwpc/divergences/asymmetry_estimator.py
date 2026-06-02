from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

from iwpc.metrics.weighted_mean_metric import WeightedMeanMetric
from .fdivergence_base import FDivergenceEstimator
from iwpc.symmetries.group_action import GroupAction


class AsymmetryEstimator(FDivergenceEstimator):
    """
    FDivergenceEstimator variant that estimates the f-divergence between a distribution p and its symmetrised image
    under a GroupAction (i.e. the asymmetry of p with respect to the group). The q-side summand is averaged over the
    group action so the model only needs to learn the asymmetric component of log(p / q)
    """
    def __init__(self, group: GroupAction, *args, **kwargs):
        """
        Parameters
        ----------
        group
            The GroupAction under which the asymmetry is being measured. The q-side summand is averaged over a Haar
            sample of this group at every evaluation
        *args, **kwargs
            Forwarded to FDivergenceEstimator.__init__
        """
        super().__init__(*args, **kwargs)
        self.group = group
        self.sym_q_fn = self.group.symmetrize(lambda x: self.divergence.calculate_naive_q_summands_given_log(self.model(x)))

    def _configure_metrics(self) -> None:
        """
        Configures `val_Df` and `val_Df_err` from a single WeightedMeanMetric whose mean and standard error track the
        validation asymmetry estimate and its uncertainty respectively
        """
        self.val_accumulator = WeightedMeanMetric()
        self.val_Df = self.val_accumulator[0]
        self.val_Df_err = self.val_accumulator[1]

    def _calculate_batch_loss(self, batch: Tuple) -> Tensor:
        """
        Returns the negative of the train estimate of the asymmetry: weighted mean of the naive p-summand minus the
        group-symmetrised q-summand

        Parameters
        ----------
        batch
            (features, labels, weights). Labels are ignored; both p and the symmetrised q are evaluated on `features`

        Returns
        -------
        Tensor
            Scalar loss
        """
        x, _, weights = batch

        return - (weights * (self.divergence.calculate_naive_p_summands_given_log(self.model(x)[:, 0]) - self.sym_q_fn(x)[:, 0])).mean()

    def _accumulate_validation_Df(self, batch: Tuple):
        """
        Updates `val_Df` / `val_Df_err` with the per-sample p-summand minus group-symmetrised q-summand for the current
        validation batch

        Parameters
        ----------
        batch
            (features, labels, weights). Labels are ignored
        """
        x, _, weights = batch

        divs = self.divergence.calculate_naive_p_summands_given_log(self.model(x)[:, 0]) - self.sym_q_fn(x)[:, 0]
        self.val_accumulator(
            weights,
            divs
        )
