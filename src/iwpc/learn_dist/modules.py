from typing import Optional

import torch
from lightning import LightningModule
from lightning.pytorch.cli import ReduceLROnPlateau

from iwpc.divergences import DifferentiableFDivergence
from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.base_models.sampleable_base_model import SampleableBaseModel
from iwpc.models.utils import basic_model_factory


class DistributionApproximator(LightningModule):
    def __init__(
        self,
        base_distribution: SampleableBaseModel,
        divergence: DifferentiableFDivergence,
        base_distribution_sample_rate: int = 1,
        input_encoding: Optional[Encoding] = None,
    ):
        super().__init__()
        self.log_p_over_q_model = basic_model_factory(
            input_encoding if input_encoding is not None else base_distribution.dimension,
            1,
            hidden_layer_sizes=(128, 64, 64, 64, 64),
        )
        self.base_distribution_sample_rate = base_distribution_sample_rate
        self.divergence = divergence
        self.base_distribution = base_distribution

        self.save_hyperparameters()

    def calculate_batch_divergence(self, batch):
        samples, _, weights = batch
        base_samples = self.base_distribution.draw(samples.shape[0] * self.base_distribution_sample_rate).to(self.device)
        p_summands = self.divergence.calculate_naive_p_summands(torch.exp(torch.clip(self.log_p_over_q_model(samples)[:, 0], -14, 14)))
        q_summands = self.divergence.calculate_naive_q_summands(torch.exp(torch.clip(self.log_p_over_q_model(base_samples)[:, 0], -14, 14)))
        divergence = (weights * p_summands).sum() / weights.sum() - q_summands.mean()
        return divergence

    def training_step(self, batch):
        loss = - self.calculate_batch_divergence(batch)
        self.log('train_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = - self.calculate_batch_divergence(batch)
        self.log('val_loss', loss, on_epoch=True, on_step=False, prog_bar=True)
        return loss

    def learned_log_prob(self, x):
        x = torch.as_tensor(x, device=self.device, dtype=torch.float)
        with torch.no_grad():
            return self.log_p_over_q_model(x)[:, 0] + self.base_distribution.log_prob(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.log_p_over_q_model.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=5,
                    factor=0.5,
                    monitor='val_loss',
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
