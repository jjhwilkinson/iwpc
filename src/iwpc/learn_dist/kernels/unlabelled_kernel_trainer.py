import torch
from lightning import LightningModule
from torch import optim
from torch.nn.functional import logsigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.divergences import DifferentiableFDivergence, KLDivergence
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class KernelKLDivergenceGradientLoss:
    def __call__(self, cond, kernel, log_p_over_q_model, weights=None):
        weights = torch.ones(cond.shape[0], dtype=torch.float32, device=cond.device) if weights is None else weights
        samples, log_prob = kernel.draw_with_log_prob(cond)
        with torch.no_grad():
            p_over_q = torch.exp(log_p_over_q_model(cond + samples))[:, 0]

        return -(weights * log_prob * p_over_q).mean()


class UnLabelledKernelTrainer(LightningModule):
    def __init__(
        self,
        kernel: TrainableKernelBase,
        log_p_over_q_model,
        divergence: DifferentiableFDivergence = KLDivergence(),
        kernel_is_diffs: bool = True,
        start_kernel_training_epoch: int = 10
    ):
        super().__init__()
        self.kernel = kernel
        self.divergence = divergence
        self.log_p_over_q_model = log_p_over_q_model
        self.loss = KernelKLDivergenceGradientLoss()
        self.kernel_is_diffs = kernel_is_diffs
        self.start_kernel_training_epoch = start_kernel_training_epoch
        self.automatic_optimization = False
        self.first_epoch = None

    def calculate_divergence(self, batch):
        cond, samples, weights = batch
        labels, cond = cond[:, 0], cond[:, 1:]
        mask = labels == 1
        # if self.kernel_is_diffs:
        #     samples[mask] += cond[mask] + self.kernel.draw(cond[mask])
        # else:
        #     samples[mask] = self.kernel.draw(cond[mask])
        # log_p_over_q = self.log_p_over_q_model(samples)

        q = cond[mask]
        return 1 + (
            logsigmoid(self.log_p_over_q_model(samples[~mask])).mean()
            + logsigmoid(-self.log_p_over_q_model(q + self.kernel.draw(q))).mean()
        ) / 2 / torch.log(torch.tensor(2.))

    def calculate_kernel_loss(self, batch):
        cond, samples, weights = batch
        labels, cond = cond[:, 0], cond[:, 1:]
        mask = labels == 1
        return self.loss(cond[mask], self.kernel, self.log_p_over_q_model, weights[mask])

    def training_step(self, batch, batch_idx):
        discriminator_optimizer, kernel_optimizer = self.optimizers()
        # if self.current_epoch > self.start_kernel_training_epoch and (self.current_epoch % 2 == 0):
        # if random() < 0.1:

        kernel_loss = self.calculate_kernel_loss(batch)
        if batch_idx == 0:
            kernel_optimizer.zero_grad()
        kernel_loss.backward()
        self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.first_epoch = self.current_epoch
        if batch_idx % 10 == 0:
            # print('step')
            kernel_optimizer.step()

        div = self.calculate_divergence(batch)
        self.log('train_divergence', div, on_step=True, on_epoch=True, prog_bar=True)
        discriminator_optimizer.zero_grad()
        (-div).backward()
        discriminator_optimizer.step()

    def validation_step(self, batch):
        div = self.calculate_divergence(batch)
        self.log('val_divergence', div, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        discriminator_optimizer = optim.Adam(self.log_p_over_q_model.parameters(), lr=1e-3)
        kernel_optimizer = optim.Adam(self.kernel.parameters(), lr=1e-4)
        return discriminator_optimizer, kernel_optimizer

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="max",
                    patience=10,
                    factor=0.1,
                ),
                "monitor": "val_divergence",
                "frequency": 1,
            },
        }
