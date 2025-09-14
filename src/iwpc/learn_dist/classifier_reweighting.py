from typing import Optional, Any, Tuple

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.cli import ReduceLROnPlateau
from torch.nn import Module
from torch.nn.functional import logsigmoid

from iwpc.models.utils import basic_model_factory
from .base_distributions.sampleable_base_model import SamplableBaseModel


class DistributionApproximator(LightningModule):
    """
    A module that can be trained to learn a data-distribution given samples by classifying between the data samples and
    samples from a known base-distribution. The trained model can be used to estimate the log probability of an arbitrary
    sample and produce weighted samples from the distribution
    """
    def __init__(
        self,
        base_distribution: SamplableBaseModel,
        base_distribution_sample_rate: int = 1,
        log_p_over_q_model: Optional[Module] = None,
    ):
        """
        Parameters
        ----------
        base_distribution
            A SamplableBaseModel to sample from. The closer the SamplableBaseModel is to the target distribution, the
            better the final result is likely to be
        base_distribution_sample_rate
            The number of samples to sample from a base distribution in each batch
        log_p_over_q_model
            The Module to use for the log probability rato between the data and base-distribution samples
        """
        super().__init__()
        self.log_p_over_q_model = (
            log_p_over_q_model if log_p_over_q_model is not None
            else basic_model_factory(base_distribution.dimension, 1)
        )
        self.base_distribution_sample_rate = base_distribution_sample_rate
        self.base_distribution = base_distribution

        self.save_hyperparameters()

    def calculate_batch_loss(self, batch: Any) -> torch.Tensor:
        """
        Calculates the BCE loss for a batch
        """
        samples, _, weights = batch
        base_samples = torch.tensor(
            self.base_distribution.draw(samples.shape[0] * self.base_distribution_sample_rate),
            dtype=torch.float,
            device=self.device,
        )

        loss = - (
            (weights * logsigmoid(self.log_p_over_q_model(samples))).mean()
            + (weights * logsigmoid(-self.log_p_over_q_model(base_samples))).mean()
        )
        return loss

    def training_step(self, batch: Any) -> torch.Tensor:
        """
        Calculates and logs the BCE loss for a training batch
        """
        loss = self.calculate_batch_loss(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        """
        Calculates and logs the BCE loss for a validation batch
        """
        loss = self.calculate_batch_loss(batch)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def learned_log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Calculates the learned log-probability for the given samples
        """
        with torch.no_grad():
            return (
                self.log_p_over_q_model(torch.as_tensor(x, device=self.device, dtype=torch.float))[:, 0]
                + self.base_distribution.log_prob(x)
            )

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler
        """
        optimizer = torch.optim.Adam(self.log_p_over_q_model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=20,
                    factor=0.5,
                    monitor='val_loss',
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def draw(self, num_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Produces num_samples weighted samples from the learned distribution. Note that the log of the weights is provided

        Parameters
        ----------
        num_samples
            The number of samples to draw

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            The samples drawn from the learned distribution, and the logarithm of the weight of each sample
        """
        samples = self.base_distribution.draw(num_samples)
        with torch.no_grad():
            log_p_over_q = self.log_p_over_q_model(torch.tensor(samples, dtype=torch.float, device=self.device))
        return (
            samples,
            log_p_over_q.cpu().detach().numpy()[:, 0]
        )
