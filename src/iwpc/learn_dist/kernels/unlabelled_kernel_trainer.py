import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning import LightningModule
from torch import optim
from torch.nn.functional import logsigmoid
from torch.optim.lr_scheduler import ReduceLROnPlateau

from iwpc.accumulators.binned_weighted_stat_accumulator import BinnedWeightedStatAccumulator
from iwpc.divergences import DifferentiableFDivergence, KLDivergence
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


def calculate_invmass_sq(vec1, vec2):
    dots = torch.clip(
        torch.sin(vec1[:, 1]) * torch.sin(vec2[:, 1]) * torch.cos(vec1[:, 2]) * torch.cos(vec2[:, 2])
        + torch.sin(vec1[:, 1]) * torch.sin(vec2[:, 1]) * torch.sin(vec1[:, 2]) * torch.sin(vec2[:, 2])
        + torch.cos(vec1[:, 1]) * torch.cos(vec2[:, 1])
    , -1, 1)
    invmass_sq = 2 / vec1[:, 0].abs() / vec2[:, 0].abs() / torch.sin(vec1[:, 1]) / torch.sin(vec2[:, 1]) * (1 - dots)
    return invmass_sq


def calc_eta_from_theta(theta):
    return -np.log(np.tan(theta / 2))


def is_in(x, a, b):
    return (x >= a) & (x <= b)


class KernelKLDivergenceGradientLoss:
    def __call__(self, cond, kernel, log_p_over_q_model, weights=None):
        weights = torch.ones(cond.shape[0], dtype=torch.float32, device=cond.device) if weights is None else weights
        samples, log_prob = kernel.draw_with_log_prob(cond)
        samples = cond + samples

        # mask = ((1 / samples[:, 0].abs()) > 15e3) & ((1 / samples[:, 3].abs()) > 15e3)
        # mask = mask & ((1 / samples[:, 0].abs()) < 80e3) & ((1 / samples[:, 3].abs()) < 80e3)

        with torch.no_grad():
            p_over_q = torch.exp(log_p_over_q_model(samples))[:, 0]

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

        self.p_acc = BinnedWeightedStatAccumulator([
            np.linspace(-2.5, 2.5, 101),
            np.linspace(-np.pi, np.pi, 101),
            # np.linspace(-1, 1, 101),
            # np.linspace(0e3, 150e3, 101),
        ])
        self.q_acc = BinnedWeightedStatAccumulator(self.p_acc.bins)
        self.p_acc2 = BinnedWeightedStatAccumulator([
            np.linspace(-2.5, 2.5, 101),
            np.linspace(0e3, 150e3, 101),
        ])
        self.q_acc2 = BinnedWeightedStatAccumulator(self.p_acc2.bins)

    def calculate_divergence(self, batch, log=False):
        cond, samples, weights = batch
        labels, cond = cond[:, 0], cond[:, 1:]
        mask = labels == 1
        # if self.kernel_is_diffs:
        #     samples[mask] += cond[mask] + self.kernel.draw(cond[mask])
        # else:
        #     samples[mask] = self.kernel.draw(cond[mask])
        # log_p_over_q = self.log_p_over_q_model(samples)

        p = samples[~mask]
        # p = p[((1 / p[:, 0].abs()) > 15e3) & ((1 / p[:, 3].abs()) > 15e3) & ((1 / p[:, 0].abs()) < 80e3) & ((1 / p[:, 3].abs()) < 80e3)]

        q = cond[mask]
        q = q + self.kernel.draw(q)
        # q = q[((1 / q[:, 0].abs()) > 15e3) & ((1 / q[:, 3].abs()) > 15e3) & ((1 / q[:, 0].abs()) < 80e3) & ((1 / q[:, 3].abs()) < 80e3)]

        # p = p[is_in(torch.sqrt(calculate_invmass_sq(p[:, :3], p[:, 3:])), 65e3, 110e3)]
        # q = q[is_in(torch.sqrt(calculate_invmass_sq(q[:, :3], q[:, 3:])), 65e3, 110e3)]
        # p = p[is_in(1 / p[:, 0].abs(), 20e3, 80e3)]
        # q = q[is_in(1 / q[:, 0].abs(), 20e3, 80e3)]
        # p = p[is_in(1 / p[:, 3].abs(), 20e3, 80e3)]
        # q = q[is_in(1 / q[:, 3].abs(), 20e3, 80e3)]
        #
        # p = p[~is_in(calc_eta_from_theta(p[:, 1]), -0.2, 0.2)]
        # p = p[~is_in(calc_eta_from_theta(p[:, 4]), -0.2, 0.2)]
        # q = q[~is_in(calc_eta_from_theta(q[:, 1]), -0.2, 0.2)]
        # q = q[~is_in(calc_eta_from_theta(q[:, 4]), -0.2, 0.2)]
        # p = p[is_in(calc_eta_from_theta(p[:, 1]), -2, 2)]
        # p = p[is_in(calc_eta_from_theta(p[:, 4]), -2, 2)]
        # q = q[is_in(calc_eta_from_theta(q[:, 1]), -2, 2)]
        # q = q[is_in(calc_eta_from_theta(q[:, 4]), -2, 2)]

        if log:
            self.p_acc.update([calc_eta_from_theta(p[:, 1]), p[:, 2]], logsigmoid(self.log_p_over_q_model(p)).cpu().detach().numpy()[:, 0])
            self.q_acc.update([calc_eta_from_theta(q[:, 1]), q[:, 2]], logsigmoid(-self.log_p_over_q_model(q)).cpu().detach().numpy()[:, 0])
            self.p_acc2.update([calc_eta_from_theta(p[:, 1]), 1/p[:, 0].abs()], logsigmoid(self.log_p_over_q_model(p)).cpu().detach().numpy()[:, 0])
            self.q_acc2.update([calc_eta_from_theta(q[:, 1]), 1/q[:, 0].abs()], logsigmoid(-self.log_p_over_q_model(q)).cpu().detach().numpy()[:, 0])

        return 1 + (
            logsigmoid(self.log_p_over_q_model(p)).mean()
            + logsigmoid(-self.log_p_over_q_model(q)).mean()
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

        if self.current_epoch % 3 == -1:
            kernel_loss = self.calculate_kernel_loss(batch)
            if batch_idx == 0:
                kernel_optimizer.zero_grad()
            # kernel_loss.backward()
            self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.first_epoch = self.current_epoch
            if batch_idx == 167:
            #     print('step')
                # for i in range()
                kernel_optimizer.step()
        else:
            div = self.calculate_divergence(batch)
            self.log('train_divergence', div, on_step=True, on_epoch=True, prog_bar=True)
            discriminator_optimizer.zero_grad()
            (-div).backward()
            discriminator_optimizer.step()

    def validation_step(self, batch):
        div = self.calculate_divergence(batch, log=True)
        # kernel_loss = self.calculate_kernel_loss(batch)

        self.log('val_divergence', div, on_step=False, on_epoch=True, prog_bar=True)
        # self.log('val_kernel_loss', kernel_loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        fig = plt.figure()
        plt.imshow(
            np.log(2) + 0.5*(self.p_acc.weighted_mean_hist + self.q_acc.weighted_mean_hist).T,
            aspect='auto',
            origin='lower',
            cmap='bwr',
            vmin=-0.01,
            vmax=0.01,
            extent=(-2.5, 2.5, -np.pi, np.pi),
        )
        plt.colorbar()
        self.logger.experiment.add_figure(
            'Binned Div eta-phi',
            fig,
            global_step=self.current_epoch
        )
        self.p_acc.reset()
        self.q_acc.reset()

        fig = plt.figure()
        plt.imshow(
            np.log(2) + 0.5*(self.p_acc2.weighted_mean_hist + self.q_acc2.weighted_mean_hist).T,
            aspect='auto',
            origin='lower',
            cmap='bwr',
            vmin=-0.01,
            vmax=0.01,
            extent=(-2.5, 2.5, 0, 150e3),
        )
        plt.colorbar()
        self.logger.experiment.add_figure(
            'Binned Div eta-pt',
            fig,
            global_step=self.current_epoch
        )
        self.p_acc2.reset()
        self.q_acc2.reset()

    def configure_optimizers(self):
        discriminator_optimizer = optim.Adam(self.log_p_over_q_model.parameters(), lr=1e-3)
        kernel_optimizer = optim.Adam(self.kernel.parameters(), lr=1e-3)
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
