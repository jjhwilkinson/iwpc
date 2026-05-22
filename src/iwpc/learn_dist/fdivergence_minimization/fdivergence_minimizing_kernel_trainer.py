import math
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
        target_cut_pass_prob: float = None,
    ):
        """
        Parameters
        ----------
        sampled_kernel
            The trainable kernel that produces q via convolution with the base distribution. Optimised to minimise the
            f-divergence between p and q
        log_p_over_q_model
            A torch Module that maps a sample x to a scalar estimate of log(p(x) / q(x)). Trained as a binary classifier
            between p and q samples
        divergence
            The DifferentiableFDivergence to minimise. Both the discriminator's BCE loss and the kernel's surrogate loss
            are derived from it
        exact_kernel
            Optional FiniteKernelInterface representing a discrete component of q whose outcomes can be enumerated
            exactly rather than sampled. If provided, expectations over its outcomes are taken via summation
        discriminator_opt_lr
            Learning rate for the discriminator (log_p_over_q_model) Adam optimizer
        kernel_opt_lr
            Learning rate for the kernel Adam optimizer
        start_kernel_train_epoch
            Epoch from which the kernel begins training. Earlier epochs only train the discriminator
        start_discriminator_train_epoch
            Epoch from which the discriminator begins training
        kernel_resample_rate
            Number of fresh kernel draws per batch when computing the kernel loss. Higher values reduce gradient
            variance at proportional compute cost
        zero_out_init_q_samples
            If True, the q-side input samples are zeroed before being added to the kernel draw. Used when the kernel
            output is to be interpreted as the full sample rather than a residual on top of an initial guess
        accumulate_kernel_batches
            Number of batches over which to accumulate kernel gradients before stepping the kernel optimiser. -1 steps
            every batch
        """
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
        self.target_cut_pass_prob = target_cut_pass_prob

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
        """
        Evaluates the discriminator on the given samples and returns the scalar log(p / q) estimate per sample

        Parameters
        ----------
        samples
            A tensor of samples of shape (N, sample_dimension)

        Returns
        -------
        torch.Tensor
            A tensor of shape (N,) of log(p / q) estimates
        """
        return self.log_p_over_q_model(samples)[:, 0]

    def exact_outcomes_with_log_prob_iter(self, q_base_samples) -> Iterator[tuple[Tensor, Tensor]]:
        """
        Enumerates the outcomes of the exact kernel and the corresponding log probability for each base sample. If no
        exact kernel is configured, returns a single trivial (zero-width outcome, zero log-prob) pair so callers can
        treat both code paths uniformly

        Parameters
        ----------
        q_base_samples
            Base samples used as conditioning information for the exact kernel, shape (N, base_dim)

        Returns
        -------
        Iterator[tuple[Tensor, Tensor]]
            Iterator yielding (outcome, log_prob) pairs over the exact kernel's discrete outcomes
        """
        return (
            self.exact_kernel.outcomes_with_log_prob_iter(q_base_samples) if self.exact_kernel is not None
            else [
                torch.zeros((q_base_samples.shape[0], 0), dtype=torch.float32, device=self.device),
                torch.zeros(q_base_samples.shape[0], dtype=torch.float32, device=self.device)
            ]
        )

    def sampled_kernel_cond_iter(self, q_base_samples) -> Iterator[tuple[Tensor, Tensor]]:
        """
        Yields the conditioning vector that should be fed into the sampled kernel for each exact-kernel outcome,
        together with the log probability of that outcome. When no exact kernel is configured, yields the base samples
        themselves with a zero log-prob so the caller iterates exactly once

        Parameters
        ----------
        q_base_samples
            Base samples used as conditioning information for both kernels, shape (N, base_dim)

        Yields
        ------
        tuple[Tensor, Tensor]
            (conditioning tensor of shape (N, exact_outcome_dim + base_dim), exact-outcome log probability of shape (N,))
        """
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

    def full_sample_iter_and_cut_pass_log_prob(
        self,
        q_base_samples: Tensor,
        q_init_samples: Tensor,
    ) -> Tuple[Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]], Tensor]:
        """
        For each discrete outcome of the exact (cut) kernel, yields a corresponding draw of the full q sample together
        with the log-probabilities needed for the cross-entropy and kernel losses. Also returns the per-row
        log-probability that a draw from the exact kernel's base distribution would have passed the cut, used to
        reweight terms so that expectations under the cut distribution can be recovered from samples drawn under the
        un-cut base distribution

        Parameters
        ----------
        q_base_samples
            Base-distribution samples for the q-side of the batch, of shape (N, base_dim). Used as conditioning
            information for both the exact and sampled kernels
        q_init_samples
            Initial q samples to which the kernel output is added, of shape (N, sample_dim). Zeroed if
            self.zero_out_init_q_samples is True

        Returns
        -------
        Tuple[Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]], Tensor]
            1. An iterator over the exact kernel's allowed outcomes yielding tuples of
                (q, sample_log_prob, exact_outcome, exact_log_prob) where:
                - q is the full q sample of shape (N, sample_dim)
                - sample_log_prob is the joint log-probability of the sampled kernel draw and the exact outcome,
                  of shape (N,)
                - exact_outcome is the single discrete outcome of the exact kernel for this iteration, of shape
                  (exact_outcome_dim,)
                - exact_log_prob is the per-row log-probability of the exact outcome conditional on passing the cut,
                  of shape (N,)
            2. cut_pass_log_prob: the per-row log-probability that a sample from the exact kernel's base distribution
                would pass the cut, of shape (N,)
        """
        if self.zero_out_init_q_samples:
            q_init_samples = torch.zeros_like(q_init_samples)

        exact_outcome_log_prob_iter, cut_pass_log_prob = self.exact_kernel.outcome_with_log_prob_iter_and_cut_pass_log_prob(q_base_samples)
        def full_sample_iter() -> Iterator[Tuple[Tensor, Tensor, Tensor, Tensor]]:
            for exact_outcome, exact_log_prob in exact_outcome_log_prob_iter:
                repeated_exact_outcome = exact_outcome.repeat((q_base_samples.shape[0], 1))
                sampled_kernel_cond = torch.concat([repeated_exact_outcome, q_base_samples], dim=1)
                sampled_kernel_samples, sampled_log_prob = self.sampled_kernel.draw_with_log_prob(sampled_kernel_cond)
                q = q_init_samples + torch.concat([sampled_kernel_samples, repeated_exact_outcome], dim=1)
                sample_log_prob = sampled_log_prob + exact_log_prob
                yield q, sample_log_prob, exact_outcome, exact_log_prob

        return full_sample_iter(), cut_pass_log_prob

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
        q_weights = weights[q_mask]
        p_weights = weights[~q_mask]

        weighted_log_sigmoids = torch.tensor(0., device=self.device)
        full_sample_iter, cut_pass_log_prob = self.full_sample_iter_and_cut_pass_log_prob(base_samples[q_mask], samples[q_mask])
        for q, sample_log_prob, exact_outcome, exact_log_prob in full_sample_iter:
            log_p_over_q = self.calculate_log_p_over_q(q)
            weighted_log_sigmoids = weighted_log_sigmoids + exact_log_prob.detach().exp() * logsigmoid(-log_p_over_q)

        cut_pass_probs = cut_pass_log_prob.exp().detach()
        q_loss = - (q_weights * cut_pass_probs * weighted_log_sigmoids).mean() / (q_weights * cut_pass_probs).mean()
        p_loss = - (p_weights * logsigmoid(self.calculate_log_p_over_q(samples[~q_mask]))).mean()
        return (p_loss + q_loss) / 2

    def calculate_kernel_loss(self, batch: Tuple[Tensor, Tensor, Tensor, Tensor], stage) -> Tensor:
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

        full_sample_iter, cut_pass_log_prob = self.full_sample_iter_and_cut_pass_log_prob(base_samples[q_mask], samples[q_mask])
        loss = torch.tensor(0., device=base_samples.device, requires_grad=True)
        log_summand = cut_pass_log_prob + torch.log(q_weights.abs())
        neg_inf = torch.full_like(log_summand, float('-inf'))
        log_pos = torch.where(q_weights > 0, log_summand, neg_inf).logsumexp(0)
        log_neg = torch.where(q_weights < 0, log_summand, neg_inf).logsumexp(0)
        log_average_cut_pass_prob = (
            log_pos + torch.log1p(-torch.exp(log_neg - log_pos))
        ) - math.log(q_weights.shape[0])
        for q, sample_log_prob, exact_outcome, exact_log_prob in full_sample_iter:
            with torch.no_grad():
                log_p_over_q = self.calculate_log_p_over_q(q)
            total_q_weight = q_weights * (exact_log_prob.detach() + cut_pass_log_prob.detach() - log_average_cut_pass_prob.detach()).exp()
            loss = loss + (total_q_weight * self.divergence.f_dash_given_log(-log_p_over_q) * (sample_log_prob + cut_pass_log_prob - log_average_cut_pass_prob)).mean()

        normalized_log_poisson_term = - (
            self.target_cut_pass_prob * log_average_cut_pass_prob - log_average_cut_pass_prob.exp()
            - (self.target_cut_pass_prob * torch.log(self.target_cut_pass_prob) - self.target_cut_pass_prob)
        )
        loss = loss + normalized_log_poisson_term

        self.log(f"{stage}_kernel_loss", loss, on_step=True, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_average_cut_pass_prob", log_average_cut_pass_prob.exp(), on_step=False, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_normalized_log_poisson_term", normalized_log_poisson_term, on_step=False, on_epoch=True, prog_bar=False)
        return loss

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
            kernel_loss = self.calculate_kernel_loss(batch, 'train')
            kernel_loss.backward()
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
        Calculates the validation learned divergence between p and q via the discriminator's BCE loss and logs it as
        `val_divergence`

        Parameters
        ----------
        batch
            (base_samples, data_samples, labels, weights). Same convention as training_step
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