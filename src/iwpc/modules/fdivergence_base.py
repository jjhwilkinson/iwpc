from abc import ABC, abstractmethod
from typing import Tuple

import torch
from lightning import LightningModule
from torch import optim, Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.divergences import DifferentiableFDivergence


class FDivergenceEstimator(LightningModule, ABC):
    """
    Base class for trainable f-divergence estimators as outlined in https://arxiv.org/abs/2405.06397 for estimating a
    lower bound on the f-divergence between two distribution, $D_f(p, q)$. Training is performed with a dynamic learning
    rate which reduces by a factor of lr_decay_factor if no improvement is seen in the validation divergence bound in
    lr_patience epochs
    """
    def __init__(
        self,
        model: Module,
        divergence: DifferentiableFDivergence,
        initial_learning_rate: float = 1e-3,
        lr_patience: int = 10,
        lr_decay_factor: float = 0.1,
    ):
        """
        Parameters
        ----------
        model
            The function approximator to be used in the training process to regress the log likelihood ratio
            $\log\left(\frac{p}{q}\right)$. Should output scalar value
        divergence
            A DifferentiableFDivergence implementation which determines which f-divergence is calculated
        initial_learning_rate
            The initial learning rate of the optimizer
        lr_patience
            The number of epochs to wait for an improvement in the validation estimate of the f-divergence before the
            learning is reduced by a factor of lr_decay_factor
        lr_decay_factor
            The factor by which the lr is to be reduced when no improvement is seen in the validation estimate of the
            f-divergence for lr_patience epochs
        """
        super().__init__()
        self.model = model
        self.divergence = divergence
        self.learning_rate = initial_learning_rate
        self.prev_batch = None
        self.lr_patience = lr_patience
        self.lr_decay_factor = lr_decay_factor

        self._configure_metrics()
        self.val_Df_sig = self.val_Df / self.val_Df_err

    @abstractmethod
    def _configure_metrics(self) -> None:
        """
        Configures the metrics required to accumulate the validation f-divergence across the whole dataset. Must
        configure at least two metric attributes, 'val_Df' and 'val_Df_err' that provide the validation estimate of the
        f-divergence and its corresponding standard error across the whole validation dataset
        """

    @abstractmethod
    def _calculate_batch_loss(self, batch: Tuple) -> Tensor:
        """
        Calculate the loss of the batch. In this case the loss corresponds to the negative of the train estimate of the
        f-divergence

        Parameters
        ----------
        batch
            A tuple containing the necessary information to calculate the f-divergence estimate. This may vary from
            implementation to implementation but is typically a triplet of features, labels, and weights

        Returns
        -------
        Tensor
            The loss of the batch
        """

    @abstractmethod
    def _accumulate_validation_Df(self, batch: Tuple):
        """
        Updates the values of val_df and val_df_err with the given batch loss

        Parameters
        ----------
        batch
            A tuple containing the necessary information to calculate the f-divergence estimate. This may vary from
            implementation to implementation but is typically a triplet of features, labels, and weights
        """

    def training_step(self, batch: Tuple, batch_idx: int) -> Tensor:
        """
        Implements a generic version of the training procedure described in https://arxiv.org/abs/2405.06397 including
        some logging of metrics and debugging information

        Parameters
        ----------
        batch
            A tuple containing the necessary information to calculate the f-divergence estimate. This may vary from
            implementation to implementation but is typically a triplet of features, labels, and weights
        batch_idx
            The index of the batch

        Returns
        -------
        Tensor
            The batch train loss
        """
        loss = self._calculate_batch_loss(batch)
        if torch.isnan(loss):
            print('Encountered nan. Please drop a breakpoint here to debug and use self.prev_* to diagnose')

        self.prev_batch = batch
        self.log(f"train_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """
        Implements a generic version of the validation procedure for estimating the f-divergence across the entire
        validation set including the logging of some metrics

        Parameters
        ----------
        batch
            A tuple containing the necessary information to calculate the f-divergence estimate. This may vary from
            implementation to implementation but is typically a triplet of features, labels, and weights
        batch_idx
            The index of the batch
        """
        self._accumulate_validation_Df(batch)
        self.log('val_Df', self.val_Df, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_Df_err', self.val_Df_err, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_Df_sig', self.val_Df_sig, on_step=False, on_epoch=True, prog_bar=True)
        return

    def configure_optimizers(self) -> dict:
        """
        Configures an AdamOptimizer initialised with self.learning_rate and associated with a learning rate scheduler
        that reduces the learning rate by a factor of self.lr_decay_factor if no validation improvement is seen in
        self.lr_patience epochs

        Returns
        -------
        dict
            A dictionary containing the Adam optimizer and configuration for the learning rate scheduler
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    patience=self.lr_patience,
                    factor=self.lr_decay_factor,
                ),
                "monitor": "val_Df",
                "frequency": 1,
            },
        }

    def forward(self, x: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            The log likelihood that a given sample was drawn from either category, $\log(\frac{p(x)}{q(x)})$
        """
        return self.model(x[0])
