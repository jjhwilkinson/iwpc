from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Module

from iwpc.metrics.weighted_mean_metric import WeightedMeanMetric
from iwpc.modules.fdivergence_base import FDivergenceEstimator
from iwpc.symmetries.group_action import GroupAction


class AsymmetryEstimator(FDivergenceEstimator):
    def __init__(self, group: GroupAction, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.group = group
        self.sym_q_fn = self.group.symmetrize(lambda x: self.divergence.calculate_naive_q_summands_given_log(self.model(x)))

    def _configure_metrics(self) -> None:
        self.val_accumulator = WeightedMeanMetric()
        self.val_Df = self.val_accumulator[0]
        self.val_Df_err = self.val_accumulator[1]

    def _calculate_batch_loss(self, batch: Tuple) -> Tensor:
        x, _, weights = batch

        return - (weights * (self.divergence.calculate_naive_p_summands_given_log(self.model(x)[:, 0]) - self.sym_q_fn(x)[:, 0])).mean()

    def _accumulate_validation_Df(self, batch: Tuple):
        x, _, weights = batch

        divs = self.divergence.calculate_naive_p_summands_given_log(self.model(x)[:, 0]) - self.sym_q_fn(x)[:, 0]
        self.val_accumulator(
            weights,
            divs
        )
        return
        self.val_accumulator(
            weights,
            self.divergence.calculate_naive_p_summands_given_log(torch.exp(self.diagonal_model(x)[:, 0])) - self.sym_q_fn(x)[:, 0]
        )
