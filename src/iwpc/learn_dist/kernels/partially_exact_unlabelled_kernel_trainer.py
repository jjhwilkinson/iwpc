from typing import Tuple, Optional

import numpy as np
import torch
from lightning import LightningModule
from matplotlib import pyplot as plt
from torch import optim, Tensor
from torch.nn import Module
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric

from calibration_chapter.unidirectional_cross_calibration.train_invmass import calculate_invmass_sq
from iwpc.divergences import JensenShannonDivergence
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernel, FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.learn_dist.kernels.unlabelled_kernel_trainer import KernelLRAdjustor


class PartiallyExactKernelKLDivergenceGradientLoss:
    """
    Given a data distribution p, and a model q formed by convolving a base distribution with a kernel, the expected
    gradient of this loss w.r.t. kernel model parameters is equal to the gradient of the KL-divergence of the observed
    data within the model. So minimizing this loss is equivalent to minimizing the KL-divergence of the data with
    respect to the kernel model parameters.
    """
    def __init__(self, kernel_resample_rate: int = 10):
        """
        Parameters
        ----------
        kernel_resample_rate
            When taking the expectation with respect to the kernel's current distribution we are free to choose how many
            samples to draw from the kernel. Increasing kernel_resample_rate likely decreases statistical noise in the
            gradient calculation, but this hasn't been careful explored
        """
        self.kernel_resample_rate = kernel_resample_rate
        self.jsd = JensenShannonDivergence()

    def __call__(
        self,
        base_samples: Tensor,
        exact_kernel: FiniteKernelInterface,
        sampled_kernel: TrainableKernelBase,
        log_p_over_q_model: Module,
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        base_samples
            Samples from the q base distribution
        kernel
            The TrainableKernelBase used to produce q
        log_p_over_q_model
            A model that provides an estimate of log(p(x) / q(x)) usually obtained by training a classifier
        weights
            An optional array of sample weights

        Returns
        -------
        Tensor
            The scalar loss
        """
        if weights is None:
            weights = torch.ones(base_samples.shape[0], dtype=torch.float32, device=base_samples.device)

        loss = torch.tensor(0., requires_grad=True, device=base_samples.device)
        for outcome, outcome_log_prob in exact_kernel.outcomes_with_log_prob_iter(base_samples):
            repeated_outcome = outcome[None, :].repeat((base_samples.shape[0], 1))
            cond = torch.concat([
                repeated_outcome,
                base_samples,
            ], dim=1)

            for i in range(self.kernel_resample_rate):
                samples, log_prob = sampled_kernel.draw_with_log_prob(cond)
                with torch.no_grad():
                    log_p_over_q = log_p_over_q_model(samples)[:, 0]
                    log_p_over_q = log_p_over_q - exact_kernel.log_prob(
                        outcome.repeat((samples.shape[0], 1)), samples[:, [1, 2, 3, 5, 6, 7]]
                    )

                loss = loss + (weights * outcome_log_prob.detach().exp() * self.jsd._f_dash_given_log_torch(-log_p_over_q) * (outcome_log_prob + log_prob)).mean()
                # loss = - (weights * outcome_log_prob.detach().exp() * p_over_q.detach() * (outcome_log_prob + log_prob)).mean()

        return loss / self.kernel_resample_rate


class PartiallyExactUnLabelledKernelTrainer(LightningModule):
    """
    """
    def __init__(
        self,
        exact_kernel: FiniteKernelInterface,
        sampled_kernel: TrainableKernelBase,
        log_p_over_q_model,
        min_train_divergence: float = 1.0,
        divergence_saturation_patience: int = 10,
        divergence_saturation_decay: float = 0.5,
        drop_cooldown: int = 5,
        discriminator_lr: float = 1e-3,
        kernel_lr: float = 1e-4,
        start_kernel_train_epoch: int = 1,
        kernel_resample_rate: int = 1,
        q_base = None,
        p = None,
    ):
        """
        A LightningModule that

        Parameters
        ----------
        kernel
            The TrainableKernelBase to train
        log_p_over_q_model
            A classifier model used to continuously train to learn the probability ratio of a given sample originating
            from p or from q
        min_train_divergence
            The initial value of min_train_divergence, below which the kernel is not trained
        divergence_saturation_patience
            The number of epochs to wait for the predicted divergence to exceed min_train_divergence before the value
            is considered saturated and dropped
        divergence_saturation_decay
            The factor by which min_train_divergence is decreased when the predicted divergence saturates below
            min_train_divergence
        drop_cooldown
            When min_train_divergence is dropped, the value cannot be dropped again for this many epochs
        discriminator_lr
            The learning rate of the discriminator
        kernel_lr
            The initial learning rate of the kernel
        """
        super().__init__()
        self.exact_kernel = exact_kernel
        self.sampled_kernel = sampled_kernel
        self.log_p_over_q_model = log_p_over_q_model
        self.loss = PartiallyExactKernelKLDivergenceGradientLoss(kernel_resample_rate=kernel_resample_rate)
        self.automatic_optimization = False
        self.register_buffer('log_two', torch.log(torch.tensor(2.)))
        self.discriminator_lr = discriminator_lr
        self.kernel_lr = kernel_lr

        self.train_divergence = MeanMetric()
        self.train_divergence_record = []
        self.min_train_divergence = min_train_divergence
        self.divergence_saturation_patience = divergence_saturation_patience
        self.divergence_saturation_decay = divergence_saturation_decay
        self.last_drop_epoch = 0
        self.drop_cooldown = drop_cooldown
        self.start_kernel_train_epoch = start_kernel_train_epoch

        if q_base is not None:
            self.q_base = torch.tensor(q_base, dtype=torch.float, device=self.device)
            self.p = torch.tensor(p, dtype=torch.float, device=self.device)

    def should_drop_min_train_divergence(self) -> bool:
        """
        Decide whether the divergence predicted by the discriminator has saturated below min_train_divergence by
        checking whether it has exceeded min_train_divergence by two standard deviations of the mean within the last
        divergence_saturation_patience epochs

        Returns
        -------
        bool
            Whether min_train_divergence should be reduced
        """
        if len(self.train_divergence_record) < (self.divergence_saturation_patience + 1):
            return False
        if (self.current_epoch - self.last_drop_epoch) < self.drop_cooldown:
            return False
        if (self.current_epoch - self.start_kernel_train_epoch) < 0:
            return False

        windowed_max = max(self.train_divergence_record[-self.divergence_saturation_patience - 1:])
        train_divergence_std = np.std(self.train_divergence_record[-self.divergence_saturation_patience - 1:])
        std = train_divergence_std / np.sqrt(self.divergence_saturation_patience + 1)

        return ((windowed_max - self.min_train_divergence) / std) < 2

    def is_kernel_training(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the kernel is currently training or not based on whether the last train_divergence value is greater
            than min_train_divergence
        """
        return (
            # (len(self.train_divergence_record) > 0)
            # and (self.train_divergence_record[-1] > self.min_train_divergence)
            (self.current_epoch >= self.start_kernel_train_epoch)
            # and (np.random.random() < 0.2)
        )

    def calculate_cross_entropy(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """
        Calculates the binary cross entropy loss of the predictions made by self.log_p_over_q_model classifying between
        p and q

        Parameters
        ----------
        batch
            The base_samples, data_samples, labels, and weights in the batch. Label 0 corresponds to actual data (p) and
            label 1 to model samples (q). Base samples are samples form the base distribution, not used when label==0.
            data_samples correspond to the reconstructed value, not used when label==1 (may change in future for
            cross-calibration)

        Returns
        -------
        Tensor
            The binary cross entropy loss of self.log_p_over_q_model
        """
        base_samples, data_samples, labels, weights = batch
        mask = labels == 1
        p = data_samples[~mask]
        # exact_samples = self.exact_kernel.draw(base_samples[mask])
        # q = self.sampled_kernel.draw(torch.cat([exact_samples, base_samples[mask]], dim=1))
        base_samples = base_samples[mask]


        p_loss = - logsigmoid(self.log_p_over_q_model(p)[:, 0] - self.exact_kernel.log_prob(p[:, [0, 4]], p[:, [1, 2, 3, 5, 6, 7]])).mean()
        q_loss = torch.tensor(0.)
        for outcome, outcome_log_prob in self.exact_kernel.outcomes_with_log_prob_iter(base_samples):
            repeated_outcome = outcome[None, :].repeat((base_samples.shape[0], 1))
            cond = torch.concat([
                repeated_outcome,
                base_samples,
            ], dim=1)
            q = self.sampled_kernel.draw(cond)
            log_p_over_q = self.log_p_over_q_model(q)[:, 0] - self.exact_kernel.log_prob(outcome.repeat((q.shape[0], 1)), q[:, [1, 2, 3, 5, 6, 7]])
            q_loss = q_loss - (outcome_log_prob.detach().exp() * logsigmoid(-log_p_over_q)).mean()

        return (p_loss + q_loss) / 2

    def calculate_kernel_loss(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Calculates the kernel loss given the learned values of self.log_p_over_q_model

        Parameters
        ----------
        batch
            The base_samples, data_samples, labels, and weights in the batch. Label 0 corresponds to actual data (p) and
            label 1 to model samples (q). Base samples are samples form the base distribution, not used when label==0.
            data_samples correspond to the reconstructed value, not used when label==1 (may change in future for
            cross-calibration)

        Returns
        -------
        Tensor
            The loss of the kernel
        """
        cond, samples, labels, weights = batch
        mask = labels == 1
        return self.loss(
            cond[mask],
            self.exact_kernel,
            self.sampled_kernel,
            self.log_p_over_q_model,
            weights[mask],
        )

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], batch_idx) -> None:
        """
        Optimizes log_p_over_q_model and the parameters in self.kernel to maximise the probability of the p samples
        in q. Logs the current learned divergence between p and q

        Parameters
        ----------
        batch
            The base_samples, data_samples, labels, and weights in the batch. Label 0 corresponds to actual data (p) and
            label 1 to model samples (q). Base samples are samples form the base distribution, not used when label==0.
            data_samples correspond to the reconstructed value, not used when label==1 (may change in future for
            cross-calibration)
        """
        discriminator_optimizer, kernel_optimizer = self.optimizers()

        if self.is_kernel_training():
            kernel_optimizer.zero_grad()
            kernel_loss = self.calculate_kernel_loss(batch)
            kernel_loss.backward()
            self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=False)
            kernel_optimizer.step()

        bce = self.calculate_cross_entropy(batch)
        train_divergence = 1 - bce / self.log_two
        self.log('train_divergence', train_divergence, on_step=True, on_epoch=True, prog_bar=True)
        self.log('epoch_train_divergence', self.train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        self.log('is_kernel_training', int(self.is_kernel_training()), on_step=True, on_epoch=False, prog_bar=True)
        self.log('min_train_divergence', self.min_train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        # if not self.is_kernel_training():
        discriminator_optimizer.zero_grad()
        bce.backward()
        discriminator_optimizer.step()

        self.train_divergence(train_divergence)

        # if batch_idx % 10 == 0:
        # self.make_plot(batch_idx)

    def make_plot(self, batch_idx):
        with torch.no_grad():
            q = self.sampled_kernel.draw(torch.concat([self.exact_kernel.draw(self.q_base), self.q_base], dim=1))
            log_p_over_q = self.log_p_over_q_model(q)[:, 0]
        q_mass = torch.sqrt(calculate_invmass_sq(q[:, [2, 1, 3]], q[:, [6, 5, 7]]))
        p_mass = torch.sqrt(calculate_invmass_sq(self.p[:, [2, 1, 3]], self.p[:, [6, 5, 7]]))
        base_mass = torch.sqrt(calculate_invmass_sq(self.q_base[:, [1, 0, 2]], self.q_base[:, [4, 3, 5]]))

        plt.figure(figsize=(6, 5), layout='constrained')
        vals, bins, _ = plt.hist(p_mass, range=(61e3, 121e3), bins=100, histtype='step', label='Benchmark', linewidth='2', color='red')
        plt.hist(base_mass, bins=bins, histtype='step', label='Truth')
        plt.hist(q_mass, bins=bins, histtype='step', label='Learned')
        plt.hist(q_mass, bins=bins, weights=torch.exp(log_p_over_q), histtype='step', label='Learned Reweighted', color='k')
        plt.xlabel('mll')
        plt.legend()
        plt.savefig(f'/Users/jeremywilkinson/PycharmProjects/Thesis/minKL/benchmark/anim/{self.current_epoch}_{batch_idx}.png')
        plt.close()

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
        """
        Calculates the validation learned divergence between p and q
        """
        bce = self.calculate_cross_entropy(batch)
        self.log('val_divergence', 1 - bce / self.log_two, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        """
        Returns
        -------
        Tuple[Optimizer, Optimizer]
            The classifier's and kernel's optimizer
        """
        discriminator_optimizer = optim.Adam(self.log_p_over_q_model.parameters(), lr=self.discriminator_lr)
        kernel_optimizer = optim.Adam([*self.exact_kernel.parameters(), *self.sampled_kernel.parameters()], lr=self.kernel_lr)

        return [
            {'optimizer': discriminator_optimizer},
            {'optimizer': kernel_optimizer, 'lr_scheduler': KernelLRAdjustor(kernel_optimizer, -1, 10)},
        ]

    def on_validation_epoch_end(self) -> None:
        """
        Updates the KernelLRAdjustor with the current train divergence and drops the min_train_divergence if required
        """
        kernel_sched = self.lr_schedulers()
        kernel_sched.step(self.train_divergence.compute().cpu().numpy())

        self.train_divergence_record.append(self.train_divergence.compute().cpu().numpy())
        if self.should_drop_min_train_divergence():
            self.min_train_divergence = self.min_train_divergence * self.divergence_saturation_decay
            self.last_drop_epoch = self.current_epoch
