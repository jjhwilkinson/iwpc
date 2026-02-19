from typing import Tuple, Optional

import numpy as np
import torch
from lightning import LightningModule
from torch import optim, Tensor
from torch.nn import Module
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torchmetrics import MeanMetric

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class KernelLRAdjustor(LRScheduler):
    """
    Custom LR scheduler that performs a simple hypothesis test of whether the recent size of fluctuations in the
    divergence predicted by the discriminating model are comparable in size to the actual predicted divergence. If so
    this is interpreted as the kernels changing to quickly for the discriminating model to keep up, and so the kernel
    optimizer learning rate is dropped

    The hypothesis test performs a least-squares linear fit to the recent losses. If the standard deviation of the
    fluctuations around the best fit model are greater than a certain multiple of the constant of the fit, then the
    fluctuations are considered too large and the kernel's LR is dropped

    When the LR is dropped, the LR cannot be dropped again for window_size number of epochs
    """
    def __init__(
        self,
        kernel_optimizer: Optimizer,
        window_size: int,
        noise_multiple: float = 3.,
        decay_factor: float = 0.25,
        warmup: int = 10,
        last_epoch: int = -1,
    ):
        """
        Parameters
        ----------
        kernel_optimizer
            The optimizer responsible for training the kernel
        window_size
            The window, in number of epochs, over which the discriminator divergence hypothesis test is performed
        noise_multiple
            The threshold number of standard deviations of the noise below which the kernel's LR is dropped
        decay_factor
            The factor by which the kernel's LR is dropped
        warmup
            The number of epochs to wait before the LR scheduler begins monitoring the predicted divergence. The first
            drop opportunity is at warmup + window_size epochs
        last_epoch
            Check LRScheduler docstring
        """
        self.window_size = window_size
        self.divergence_history = []
        self.noise_multiple = noise_multiple
        self.decay_factor = decay_factor
        self.warmup = warmup

        self._last_lr = [kernel_optimizer.param_groups[0]["lr"]]
        super().__init__(kernel_optimizer, last_epoch)

    def step(self, divergence: Tensor | None = None) -> None:
        """
        Performs a step of the scheduler, recording the next discriminator-predicted divergence depending on the current
        epoch

        Parameters
        ----------
        divergence
            The current predicted divergence between the model and actual data by a discriminator classifier
        """
        self.last_epoch += 1
        if divergence is None or self.last_epoch < self.warmup:
            return None
        self.divergence_history.append(float(divergence))
        if len(self.divergence_history) > self.window_size:
            self.divergence_history = self.divergence_history[-self.window_size:]
        return super().step()

    def should_drop_lr(self) -> bool:
        """
        Performs the hypothesis test described in the class docstring

        Returns
        -------
        bool
            Whether the LR should be dropped
        """
        if len(self.divergence_history) < self.window_size:
            return False

        cov = np.cov(range(self.window_size), self.divergence_history)
        m = cov[0, 1] / cov[0, 0]
        c = np.mean(self.divergence_history) - 0.5 * (self.window_size + 1) * m
        divergence_std = np.std(self.divergence_history)

        return c / divergence_std < self.noise_multiple

    def get_lr(self) -> list[float | Tensor]:
        """
        Calculates the next value of the learning rate depending on the outcome of the hypothesis test

        Returns
        -------
        list[float | Tensor]
            The new LR group
        """
        if self.should_drop_lr():
            self.divergence_history.clear()
            return [self.get_last_lr()[0] * self.decay_factor]
        return self.get_last_lr()


class KernelKLDivergenceGradientLoss:
    """
    Given a data distribution p, and a model q formed by convolving a base distribution with a kernel, the expected
    gradient of this loss w.r.t. kernel model parameters is equal to the gradient of the KL-divergence of the observed
    data within the model. So minimizing this loss is equivalent to minimizing the KL-divergence of the data with
    respect to the kernel model parameters.
    """
    def __init__(self, kernel_resample_rate: int = 1):
        """
        Parameters
        ----------
        kernel_resample_rate
            When taking the expectation with respect to the kernel's current distribution we are free to choose how many
            samples to draw from the kernel. Increasing kernel_resample_rate likely decreases statistical noise in the
            gradient calculation, but this hasn't been careful explored
        """
        self.kernel_resample_rate = kernel_resample_rate

    def __call__(
        self,
        base_samples: Tensor,
        kernel: TrainableKernelBase,
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

        loss = 0
        for i in range(self.kernel_resample_rate):
            samples, log_prob = kernel.draw_with_log_prob(base_samples)
            with torch.no_grad():
                p_over_q = torch.exp(log_p_over_q_model(samples))[:, 0]

            loss += -(weights * log_prob * p_over_q).mean()

        return loss / self.kernel_resample_rate


class UnLabelledKernelTrainer(LightningModule):
    """
    LightningModule implementation of the kernel training procedure that minimizes the KL-divergence between two
    distributions as described in Jeremy's thesis. Trains a TrainableKernelBase to maximise the probability of data
    samples within a model generated by convolving a base-distribution with a TrainableKernelBase. Only samples from the
    base-distribution are required, the probability distribution itself is not required

    A number of tricks are employed to increase training stability. Firstly, we would like to only train the kernel when
    the classifier-predicted divergence is relatively saturated. To this end, we prevent the kernel changing too quickly
    by only training the kernel if the current predicted divergence is greater than min_train_divergence. If the value
    of the predicted divergence is determined to have saturated below min_train_divergence (as decided by
    should_drop_train_divergence), then min_train_divergence is reduced by a factor divergence_saturation_decay.

    Secondly, if the kernel's LR is too high, then the distribution of the samples from q change too quickly for the
    discriminator to keep up. This often manifests as the loss of the discriminator fluctuating rapidly from
    epoch-to-epoch. An instance of KernelLRAdjustor is used to monitor the size of fluctuations in the discriminator's
    predicted divergence, and reduces the kernel's LR if required.
    """
    def __init__(
        self,
        kernel: TrainableKernelBase,
        log_p_over_q_model,
        min_train_divergence: float = 1.0,
        divergence_saturation_patience: int = 10,
        divergence_saturation_decay: float = 0.5,
        drop_cooldown: int = 5,
        discriminator_lr: float = 1e-3,
        kernel_lr: float = 1e-4,
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
        self.kernel = kernel
        self.log_p_over_q_model = log_p_over_q_model
        self.loss = KernelKLDivergenceGradientLoss()
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

        windowed_max = max(self.train_divergence_record[-self.divergence_saturation_patience - 1:])
        train_divergence_std = np.std(self.train_divergence_record[-self.divergence_saturation_patience - 1:])
        std = train_divergence_std / np.sqrt(self.divergence_saturation_patience + 1)

        return ((windowed_max - self.min_train_divergence) / std) > 2

    def is_kernel_training(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the kernel is currently training or not based on whether the last train_divergence value is greater
            than min_train_divergence
        """
        return (
            (len(self.train_divergence_record) > 0)
            and (self.train_divergence_record[-1] > self.min_train_divergence)
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
        q = self.kernel.draw(base_samples[mask])

        return -(
            logsigmoid(self.log_p_over_q_model(p)).mean()
            + logsigmoid(-self.log_p_over_q_model(q)).mean()
        ) / 2

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
        return self.loss(cond[mask], self.kernel, self.log_p_over_q_model, weights[mask])

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> None:
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
            kernel_loss = self.calculate_kernel_loss(batch)
            kernel_optimizer.zero_grad()
            kernel_loss.backward()
            self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=False)
            kernel_optimizer.step()

        bce = self.calculate_cross_entropy(batch)
        train_divergence = 1 - bce / self.log_two
        self.log('train_divergence', train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        self.log('epoch_train_divergence', self.train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        self.log('is_kernel_training', int(self.is_kernel_training()), on_step=False, on_epoch=True, prog_bar=True)
        self.log('min_train_divergence', self.min_train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        discriminator_optimizer.zero_grad()
        bce.backward()
        discriminator_optimizer.step()

        self.train_divergence(train_divergence)

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
        kernel_optimizer = optim.Adam(self.kernel.parameters(), lr=self.kernel_lr)

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
