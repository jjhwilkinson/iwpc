from typing import Tuple, Iterator

import torch
from lightning import LightningModule
from torch import Tensor, optim
from torch.nn import Module
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer
from torchmetrics import MeanMetric

from iwpc.divergences import DifferentiableFDivergence
from iwpc.learn_dist.kernels.finite_kernel import FiniteKernelInterface
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class FDivergenceMinimizingKernelTrainer(LightningModule):
    def __init__(
        self,
        sampled_kernel: TrainableKernelBase,
        log_p_over_q_model: Module,
        divergence: DifferentiableFDivergence,
        exact_kernel: FiniteKernelInterface | None = None,
        discriminator_opt_lr: float = 1e-3,
        kernel_opt_lr: float = 1e-4,
        start_kernel_train_epoch: int = 1,
        start_discriminator_train_epoch: int = 0,
        kernel_resample_rate: int = 1,
        zero_out_init_q_samples: bool = False,
        accumulate_kernel_batches: int = -1,
    ):
        super().__init__()

        self.sampled_kernel = sampled_kernel
        self.log_p_over_q_model = log_p_over_q_model
        self.divergence = divergence
        self.exact_kernel = exact_kernel
        self.discriminator_opt_lr = discriminator_opt_lr
        self.kernel_opt_lr = kernel_opt_lr
        self.start_kernel_train_epoch = start_kernel_train_epoch
        self.start_discriminator_train_epoch = start_discriminator_train_epoch
        self.kernel_resample_rate = kernel_resample_rate
        self.zero_out_init_q_samples = zero_out_init_q_samples
        self.accumulate_kernel_batches = accumulate_kernel_batches
        self.num_accumulated_kernel_batches = 0

        self.automatic_optimization = False
        self.register_buffer('log_two', torch.log(torch.tensor(2.)))
        self.train_divergence = MeanMetric()

    def is_kernel_training(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the kernel is currently training or not based on whether the last train_divergence value is greater
            than min_train_divergence
        """
        return (
            (self.current_epoch >= self.start_kernel_train_epoch)
            # and (np.random.random() < 0.2)
        )

    def is_discriminator_training(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the kernel is currently training or not based on whether the last train_divergence value is greater
            than min_train_divergence
        """
        return (
            (self.current_epoch >= self.start_discriminator_train_epoch)
            # and (np.random.random() < 0.2)
        )

    def calculate_log_p_over_q(self, samples) -> torch.Tensor:
        return self.log_p_over_q_model(samples)[:, 0]

    def exact_outcomes_with_log_prob_iter(self, q_base_samples) -> Iterator[tuple[Tensor, Tensor]]:
        return (
            self.exact_kernel.outcomes_with_log_prob_iter(q_base_samples) if self.exact_kernel is not None
            else [
                torch.zeros((q_base_samples.shape[0], 0), dtype=torch.float32, device=self.device),
                torch.zeros(q_base_samples.shape[0], dtype=torch.float32, device=self.device)
            ]
        )

    def sampled_kernel_cond_iter(self, q_base_samples) -> Iterator[tuple[Tensor, Tensor]]:
        if self.exact_kernel is None:
            yield q_base_samples, torch.zeros(q_base_samples.shape[0], dtype=torch.float32, device=self.device)
        else:
            for exact_outcome, exact_outcome_log_prob in self.exact_kernel.outcomes_with_log_prob_iter(q_base_samples):
                repeated_outcome = exact_outcome.repeat((q_base_samples.shape[0], 1))
                yield torch.concat(
                    [
                        repeated_outcome,
                        q_base_samples,
                    ], dim=1
                ), exact_outcome_log_prob

    def calculate_cross_entropy(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
        """
        Calculates the binary cross entropy loss of the predictions made by self.log_p_over_q_model classifying between
        p and q

        Parameters
        ----------
        batch
            The q_base_samples, p_samples, labels, and weights in the batch. Label 0 corresponds to actual data (p) and
            label 1 to model samples (q). Base samples are samples form the base distribution, not used when label==0.
            p_samples correspond to the reconstructed value, not used when label==1 (may change in future for
            cross-calibration)

        Returns
        -------
        Tensor
            The binary cross entropy loss of self.log_p_over_q_model
        """
        base_samples, samples, labels, weights = batch

        q_mask = labels == 1
        p_samples = samples[~q_mask]
        q_base_samples = base_samples[q_mask]
        q_init_samples = samples[q_mask]
        if self.zero_out_init_q_samples:
            q_init_samples = torch.zeros_like(q_init_samples)

        p_loss = - (weights[~q_mask] * logsigmoid(self.calculate_log_p_over_q(p_samples))).mean()

        q_loss = torch.tensor(0.)
        q_weights = weights[q_mask]
        for sampled_kernel_cond, exact_outcome_log_prob in self.sampled_kernel_cond_iter(q_base_samples):
            q = q_init_samples + self.sampled_kernel.draw(sampled_kernel_cond)
            log_p_over_q = self.calculate_log_p_over_q(q)
            q_loss = q_loss - (q_weights * exact_outcome_log_prob.detach().exp() * logsigmoid(-log_p_over_q)).mean()

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
        base_samples, samples, labels, weights = batch
        q_mask = labels == 1
        q_weights = weights[q_mask]
        q_base_samples = base_samples[q_mask]
        q_init_samples = samples[q_mask]
        if self.zero_out_init_q_samples:
            q_init_samples = torch.zeros_like(q_init_samples)

        loss = torch.tensor(0., requires_grad=True, device=base_samples.device)
        for sampled_kernel_cond, exact_outcome_log_prob in self.sampled_kernel_cond_iter(q_base_samples):
            for i in range(self.kernel_resample_rate):
                samples, log_prob = self.sampled_kernel.draw_with_log_prob(sampled_kernel_cond)
                q_samples = q_init_samples + samples
                with torch.no_grad():
                    log_p_over_q = self.calculate_log_p_over_q(q_samples)

                loss = loss + (q_weights * exact_outcome_log_prob.detach().exp() * self.divergence.f_dash_given_log(-log_p_over_q) * (exact_outcome_log_prob + log_prob)).mean()

        return loss / self.kernel_resample_rate

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
            kernel_loss = self.calculate_kernel_loss(batch)
            kernel_loss.backward()
            self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=False)
            self.num_accumulated_kernel_batches += 1
            self.log(
                "kernel params grad sum",
                sum(0 if p.grad is None else p.grad.sum() for p in kernel_optimizer.optimizer.param_groups[0]['params']),
                on_step=True, on_epoch=False, prog_bar=False
            )
            if self.num_accumulated_kernel_batches > self.accumulate_kernel_batches:
                kernel_optimizer.step()
                kernel_optimizer.zero_grad()
                self.num_accumulated_kernel_batches = 0

        bce = self.calculate_cross_entropy(batch)
        train_divergence = 1 - bce / self.log_two
        self.log('train_divergence', train_divergence, on_step=True, on_epoch=True, prog_bar=True)
        self.log('epoch_train_divergence', self.train_divergence, on_step=False, on_epoch=True, prog_bar=True)
        self.log('is_kernel_training', int(self.is_kernel_training()), on_step=True, on_epoch=False, prog_bar=True)
        self.train_divergence(train_divergence)
        if self.is_discriminator_training():
            discriminator_optimizer.zero_grad()
            bce.backward()
            discriminator_optimizer.step()

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
        discriminator_optimizer = optim.Adam(self.log_p_over_q_model.parameters(), lr=self.discriminator_opt_lr)
        kernel_params = [*self.sampled_kernel.parameters()]
        if self.exact_kernel is not None:
            kernel_params.extend(self.exact_kernel.parameters())
        kernel_optimizer = optim.Adam(kernel_params, lr=self.kernel_opt_lr)

        return [
            {'optimizer': discriminator_optimizer},
            {'optimizer': kernel_optimizer},
        ]
