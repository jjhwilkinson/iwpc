from itertools import chain
from typing import List, Tuple, Optional

import torch
from lightning import LightningModule
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import logsigmoid
from torch.optim import Optimizer, Adam

from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase, ConcatenatedKernel


class MultiKernelKLDivergenceGradientLoss:
    """
    Given a data distribution p and a model q, the gradient of this loss w.r.t. model parameters is equal to the
    gradient of the negative log-probability of the observed data within the model. So minimizing this loss is
    equivalent to maximising the probability of the data with respect to the model parameters.

    The loss is only valid if the model q is assumed to take the form of a fixed base distribution convolved with a
    trainable kernel
    """
    def __call__(
        self,
        cond: Tensor,
        combined_kernel: ConcatenatedKernel,
        log_p_over_q_models: List[Module],
        weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Parameters
        ----------
        cond
            Samples from the q base distribution
        kernel
            The TrainableKernelBase used to produce q
        log_p_over_q_model
            A model that provides an estimate of log(p(x) / q(x)) likely obtained by training a classifier
        weights
            An optional array of sample weights

        Returns
        -------
        Tensor
            The scalar loss
        """
        weights = torch.ones(cond.shape[0], dtype=torch.float32, device=cond.device) if weights is None else weights
        samples, all_log_probs = combined_kernel.draw_with_separate_log_prob(cond)

        with torch.no_grad():
            all_p_over_q = [torch.exp(log_p_over_q_model(samples)[:, 0]) for log_p_over_q_model in log_p_over_q_models]

        return - (weights * (sum(log_prob * p_over_q for log_prob, p_over_q in zip(all_log_probs, all_p_over_q)))).mean()


class UnlabelledMultiKernelTrainer(LightningModule):
    def __init__(
        self,
        kernels: List[TrainableKernelBase],
        log_p_over_q_models: List[Module],
        start_kernel_training_epoch: int = 10,
    ):
        super().__init__()
        self.kernels = kernels
        self.combined_kernel = ConcatenatedKernel(kernels, concatenate_cond=True)
        self.log_p_over_q_models = log_p_over_q_models
        self.start_kernel_training_epoch = start_kernel_training_epoch
        self.automatic_optimization = False
        self.register_buffer('log_two', torch.log(torch.tensor(2.)))
        self.loss = MultiKernelKLDivergenceGradientLoss()

        for i, kernel in enumerate(kernels):
            self.register_module(f'kernel_{i}', kernel)
        for i, model in enumerate(log_p_over_q_models):
            self.register_module(f'log_p_over_q_model_{i}', model)

    def calculate_cross_entropies(self, batch: Tuple[Tensor, Tensor, Tensor], stage) -> List[Tensor]:
        """
        Calculates the binary cross entropy loss of the predictions made by self.log_p_over_q_model classifying between
        p and q

        Parameters
        ----------
        batch
            The conditioning information, samples, and weights in the batch

        Returns
        -------
        Tensor
            The binary cross entropy loss of self.log_p_over_q_model
        """
        cond, samples, weights = batch
        labels, cond = cond[:, 0], cond[:, 1:]
        mask = labels == 1

        p = samples[~mask]
        q = self.combined_kernel.draw(cond[mask])

        bce_losses = []
        for i, log_p_over_q_model in enumerate(self.log_p_over_q_models):
            p_llr = log_p_over_q_model(p)
            q_llr = log_p_over_q_model(q)
            if stage == 'val':
                self.validation_cross_entropy_hook(i, p, q, p_llr, q_llr)

            bce_losses.append(-(logsigmoid(p_llr).mean() + logsigmoid(-q_llr).mean()) / 2)

        return bce_losses

    def calculate_kernel_loss(self, batch: Tuple[Tensor, Tensor, Tensor]) -> Tensor:
        """
        Calculates the kernel loss given the learned values of self.log_p_over_q_model

        Parameters
        ----------
        batch

        Returns
        -------
        Tensor
            The loss of the kernel
        """
        cond, samples, weights = batch
        labels, cond = cond[:, 0], cond[:, 1:]
        mask = labels == 1
        return self.loss(cond[mask], self.combined_kernel, self.log_p_over_q_models, weights[mask])

    def training_step(self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx) -> None:
        """
        Optimizes log_p_over_q_model and the parameters in self.kernel to maximise the probability of the p samples
        in q. Logs the current learned divergence between p and q
        """
        discriminator_optimizer, kernel_optimizer = self.optimizers()

        if self.current_epoch >= self.start_kernel_training_epoch:
            kernel_loss = self.calculate_kernel_loss(batch)
            self.log('train_kernel_loss', kernel_loss, on_step=True, on_epoch=True, prog_bar=False)
            kernel_loss.backward()
            if batch_idx % 10 == 9:
                kernel_optimizer.step()
                kernel_optimizer.zero_grad()

        bce_losses = self.calculate_cross_entropies(batch, 'train')
        for i, bce in enumerate(bce_losses):
            self.log(f'train_divergence_{i}', 1 - bce / self.log_two, on_step=True, on_epoch=True, prog_bar=False)
        discriminator_optimizer.zero_grad()
        self.log(f'train_divergence_mean', sum(1 - bce / self.log_two for bce in bce_losses) / len(bce_losses), on_step=True, on_epoch=True, prog_bar=True)
        sum(bce_losses).backward()
        discriminator_optimizer.step()

    def validation_step(self, batch: Tuple[Tensor, Tensor, Tensor]):
        """
        Calculates the validation learned divergence between p and q
        """
        bce_losses = self.calculate_cross_entropies(batch, 'val')
        for i, bce in enumerate(bce_losses):
            self.log(f'val_divergence_{i}', 1 - bce / self.log_two, on_epoch=True, prog_bar=False)
        self.log(f'val_divergence_mean', sum(1 - bce / self.log_two for bce in bce_losses) / len(bce_losses), on_epoch=True, prog_bar=True)

    def validation_cross_entropy_hook(self, kernel_no, p, q, p_llrs, q_llrs):
        pass

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        """
        Returns
        -------
        Tuple[Optimizer, Optimizer]
            The classifier's and kernel's optimizer
        """
        discriminator_optimizer = Adam(list(chain(*[log_p_over_q_model.parameters() for log_p_over_q_model in self.log_p_over_q_models])), lr=1e-3)
        kernel_optimizer = Adam(self.combined_kernel.parameters(), lr=1e-4)
        return discriminator_optimizer, kernel_optimizer
