from abc import ABC, abstractmethod
from typing import Callable, List

from lightning import LightningModule
from lightning.pytorch.cli import ReduceLROnPlateau
from torch import Tensor
from torch.nn import Module
from torch.optim import Adam


class ConditionedMLFitterBase(LightningModule, ABC):
    def __init__(
        self,
        parameter_function_num_inputs: int,
        parameter_models: List[Module],
        lr_decay_factor: float = 0.5,
        initial_lr: float = 1e-3,
        lr_patience: int = 10,
    ):
        super().__init__()
        self.parameter_function_num_inputs = parameter_function_num_inputs
        self.parameter_models = parameter_models
        for i, model in enumerate(self.parameter_models):
            self.register_module(f'parameter_model_{i}', model)
        self.initial_lr = initial_lr
        self.lr_patience = lr_patience
        self.lr_decay_factor = lr_decay_factor

    @abstractmethod
    def conditioned_log_likelihood(self, samples, *parameters):
        """

        Parameters
        ----------
        samples
        parameters

        Returns
        -------

        """

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.initial_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.lr_patience,
                    factor=self.lr_decay_factor,
                    monitor='val_loss',
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def evaluate_log_likelihood(self, samples: Tensor):
        parameters = [parameter_model(samples[:, -self.num_condition_parameters:]) for parameter_model in self.parameter_functions]
        return self.conditioned_log_likelihood(samples[:, :-self.num_condition_parameters], *parameters)

    def evaluate_cross_entropy(self, batch):
        samples, _, weights = batch
        estimated_log_pdf_values = self.evaluate_log_likelihood(samples)
        return - (weights * estimated_log_pdf_values).sum() / weights.sum()

    def training_step(self, batch):
        cross_entropy = self.evaluate_cross_entropy(batch)
        self.log("train_loss", cross_entropy, on_step=False, on_epoch=True, prog_bar=True)
        return cross_entropy

    def validation_step(self, batch):
        cross_entropy = self.evaluate_cross_entropy(batch)
        self.log("val_loss", cross_entropy, on_step=False, on_epoch=True)
        return cross_entropy
