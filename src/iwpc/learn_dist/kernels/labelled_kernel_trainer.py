from lightning import LightningModule
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class LabelledKernelTrainer(LightningModule):
    """
    Basic information that trains a TrainableKernelBase given labelled samples by minimizing the negative log-likelihood
    """
    def __init__(self, kernel: TrainableKernelBase):
        super().__init__()
        self.kernel = kernel

    def calculate_loss(self, batch):
        cond, targets, _ = batch
        log_prob = self.kernel.log_prob(targets, cond)
        return - log_prob[log_prob.isfinite()].mean()

    def training_step(self, batch):
        loss = self.calculate_loss(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch):
        loss = self.calculate_loss(batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=10,
                    factor=0.1,
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }
