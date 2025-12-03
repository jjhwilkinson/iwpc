from typing import Optional

import numpy as np
import torch
from torch.nn import Module

from iwpc.encodings.matrix_encoding import MatrixEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MultivariateGaussianKernel(TrainableKernelBase):
    """
    A multidimensional Normal kernel with trainable mean and std deviations
    """
    def __init__(
        self,
        cond: int | torch.Tensor,
        sample_dim: int | torch.Tensor,
        max_chi: Optional[float] = None,
        mean_model: Optional[Module] = None,
        log_diag_model: Optional[Module] = None,
        log_rot_model: Optional[Module] = None,
    ):
        """
        Parameters
        ----------
        cond
            The conditioning space encoding or dimension
        sample_dim
            The sample space encoding or dimension
        max_chi
            The maximum chi-squared to consider in the negative log-prob when fitting for numerical stability.
        mean_model
            Optional model that constructs the mean of the distribution for the given conditioning information
        log_diag_model
            Optional model that constructs the log diagonal matrix of the distribution for the given conditioning information.
        log_rot_model
            Optional model that constructs the log rotational matrix of the distribution for the given conditioning information.

        """
        super().__init__(sample_dim, cond)
        self.cond = cond
        self.sample_dim = sample_dim
        self.mean_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if mean_model is None else mean_model
        self.log_diag_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if log_diag_model is None else log_diag_model
        self.log_rot_model = basic_model_factory(TrivialEncoding(cond), MatrixEncoding(sample_dim)) if log_rot_model is None else log_rot_model
        self.max_chi = max_chi


    def _draw(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information for each sample

        Returns
        -------
        Tensor
            A sample from the gaussian kernel for each row of conditioning information
        """
        mean = self.mean_model(cond)
        cov = self.construct_cov(self.cond)
        L = torch.cholesky(cov)
        noise = np.random.normal(0, 1, size=(cond.shape[0], cond.shape[1]))
        correlated_noise = np.einsum('bjk,bk->bj', L, noise)
        return correlated_noise + mean

    def log_prob(self,
        samples: torch.Tensor,
        cond: torch.Tensor,
        return_chi_sqs: bool = False
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        samples
            The sampling information for each sample
        cond
            The conditioning information for each sample
        return_chi_sqs
            Boolean to return the chi-square of the distribution

        Returns
        -------
        Tensor
            Log probability of samples given the conditioning information
        """
        M = self.log_rot_model(cond)
        mean = self.mean_model(cond)
        log_diags = self.log_diag_model(cond)
        diffs = samples - mean
        normed_diffs = torch.einsum('bij,bj->bi', torch.matrix_exp(M - M.transpose(1, 2)), diffs)
        normed_diffs = torch.exp(- 0.5 * log_diags) * normed_diffs
        chi_sqs = torch.sum(normed_diffs ** 2, dim=-1)
        log_prob = - 0.5 * (chi_sqs + log_diags.sum(dim=-1) + self.sample_dimension*np.log(2 * np.pi))
        return (log_prob, chi_sqs) if return_chi_sqs else log_prob

    def calculate_loss(self, batch: tuple) -> torch.Tensor:
        """
        Calculate the loss of the given batch

        Parameters
        ----------
        batch : tuple
            Training batch

        Returns
        -------
        Tensor
            A tensor containing ``-mean(log_prob)`` over finite entries.
        """
        cond, targets, _ = batch
        log_prob, chi_sqs = self.log_prob(targets, cond, return_chi_sqs = True)
        if self.max_chi is not None:
            mask = chi_sqs < self.max_chi ** 2
            log_prob = log_prob[mask]
        return - log_prob[log_prob.isfinite()].mean()

    def construct_cov(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information for each sample

        Returns
        -------
        Tensor
            Covariance matrix of the distribution for the given conditioning information.
        """
        M = self.log_rot_model(cond)
        log_diags = self.log_diag_model(cond)
        rot = torch.matrix_exp(M - M.transpose(1, 2))
        diags = torch.exp(log_diags)
        cov = torch.einsum('bij,bj,bjk->bik', rot.transpose(1, 2), diags, rot)
        return cov