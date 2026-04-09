from typing import Optional, Iterable, Tuple

import numpy as np
import torch
from numpy import ndarray
from scipy.linalg import logm
from torch import Tensor
from torch.nn import Module

from iwpc.encodings.antisymmetric_matrix_encoding import AntiSymmetricMatrixEncoding
from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory


def construct_init_parameters(cov: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """
    Given a covariance matrix, calculates the initial values for the parameterization within the
    MultivariateGaussianKernel needed to seed the kernel with the given covariance matrix.

    Parameters
    ----------
    cov
        A square covariance matrix

    Returns
    -------
    tuple[ndarray, ndarray, ndarray]
        1. The logarithm of the eigenvalues of the correlation matrix
        2. The matrix logarithm of the rotation that diagonalizes the correlation matrix
        3. The logarithm of the standard deviations of the covariance matrix
    """
    corr = cov / np.sqrt(np.diag(cov))[:, None] / np.sqrt(np.diag(cov))[None, :]
    corr_eigvals, corr_rotation = np.linalg.eigh(corr)

    log_eigvals = np.log(corr_eigvals)
    log_rot = logm(corr_rotation)
    log_stds = np.log(np.sqrt(np.diag(cov)))

    return log_eigvals, log_rot, log_stds


class MultivariateGaussianKernel(TrainableKernelBase):
    """
    Multivariate normal kernel with mean and covariance conditioned on `cond`. The covariance matrix is parameterized
    using the decomposition of a positive definite matrix into the form:

    std-dev * correlation * std-dev

    Where std-dev is a row vector of the standard deviations of the covariance matrix, and correlation is the covariance
    matrix's correlation matrix. The correlation matrix is in turn parameterized using a positive definite matrix,
    referred to as the 'un-normalized correlation matrix' through

    correlation = (
        unnormalized_correlation_matrix
        / sqrt(diag(unnormalized correlation matrix))[:, None]
        / sqrt(diag(unnormalized correlation matrix))[None, :]
    )

    Finally, S is parameterized using its eigenvalue decomposition,

    unnormalized_correlation_matrix = rot.T * D * rot

    where D is the set of positive eigenvalues of unnormalized_correlation_matrix and rot is the rotation that diagonalizes
    the unnormalized_correlation_matrix.

    Positivity of the eigenvalues is enforced using a simple exponential encoding when needed. The reason for this
    long-winded parameterization is to allow for better numerical convergence and the addition of symmetries into the
    parameter networks
    """

    def __init__(
        self,
        cond: int | torch.Tensor,
        sample_dim: int | torch.Tensor,
        max_chi: Optional[float] = None,
        mean_model: Optional[Module] = None,
        log_diag_model: Optional[Module] = None,
        log_rot_model: Optional[Module] = None,
        log_std_model: Optional[Module] = None,
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
        log_std_model
            Optional model that constructs the log standard deviation of the distribution for the given conditioning information.
        """
        super().__init__(sample_dim, cond)
        self.cond = cond
        self.sample_dim = sample_dim
        self.mean_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if mean_model is None else mean_model
        self.log_diag_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if log_diag_model is None else log_diag_model
        self.log_rot_model = basic_model_factory(TrivialEncoding(cond), AntiSymmetricMatrixEncoding(sample_dim)) if log_rot_model is None else log_rot_model
        self.log_std_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if log_std_model is None else log_std_model
        self.max_chi = max_chi


    def construct_variables(self, cond: torch.Tensor):
        """
        Parameters
        ----------
        cond : torch.Tensor
            The conditioning information for each sample.

        Returns
        -------
        mean : torch.Tensor
            The computed mean tensor.

        log_std : torch.Tensor
            Logarithm of the standard deviation tensor.

        unnorm_corr_eigvals : torch.Tensor
            Unnormalized eigenvalues of the correlation matrix.

        unnorm_corr_rot : torch.Tensor
            Unnormalized rotation matrix for the correlation matrix.

        unnorm_corr_diag : torch.Tensor
            Diagonal values of the unnormalized correlation matrix, representing its standard deviations.

        std : torch.Tensor
            Standard deviation tensor, computed as the exponential of the log standard deviations.

        log_unnorm_corr_eigvals : torch.Tensor
            Logarithm of the unnormalized eigenvalues of the correlation matrix.
        """
        log_unnorm_corr_rot = self.log_rot_model(cond)
        mean = self.mean_model(cond)
        log_std = self.log_std_model(cond)
        log_unnorm_corr_eigvals = self.log_diag_model(cond)

        unnorm_corr_eigvals = torch.exp(log_unnorm_corr_eigvals)
        unnorm_corr_rot = torch.matrix_exp(log_unnorm_corr_rot)
        unnorm_corr = torch.einsum('bji,bj,bjk->bik', unnorm_corr_rot, unnorm_corr_eigvals, unnorm_corr_rot)
        unnorm_corr_diag = torch.sqrt(torch.diagonal(unnorm_corr, dim1=1, dim2=2))
        std = torch.exp(log_std)


        return mean, log_std, unnorm_corr_eigvals, unnorm_corr_rot, unnorm_corr_diag, std, log_unnorm_corr_eigvals


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
        mean, log_std, unnorm_corr_eigvals, unnorm_corr_rot, unnorm_corr_diag, std, _ = self.construct_variables(cond)

        root_cov = torch.einsum(
            'bi,bij,bj,bjk->bik',
            std / unnorm_corr_diag,
            unnorm_corr_rot.transpose(-1, -2),
            torch.sqrt(unnorm_corr_eigvals),
            unnorm_corr_rot,
        )
        noise = torch.randn_like(mean)
        correlated_noise = torch.einsum('bjk,bk->bj', root_cov, noise)
        return correlated_noise + mean


    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            The conditioning information for each sample

        Returns
        -------
        Tuple[Tensor, Tensor]
            A sample from the gaussian kernel for each row of conditioning information and its corresponding log
            probability efficiently implemented
        """
        mean, log_std, unnorm_corr_eigvals, unnorm_corr_rot, unnorm_corr_diag, std, log_unnorm_corr_eigvals = self.construct_variables(cond)

        root_cov = torch.einsum(
            'bi,bij,bj,bjk->bik',
            std / unnorm_corr_diag,
            unnorm_corr_rot.transpose(-1, -2),
            torch.sqrt(unnorm_corr_eigvals),
            unnorm_corr_rot,
        )
        noise = torch.randn_like(mean)
        correlated_noise = torch.einsum('bjk,bk->bj', root_cov, noise)
        samples = correlated_noise + mean

        diffs = samples - mean
        normed_diffs = diffs / std
        normed_diffs = torch.exp(-0.5 * log_unnorm_corr_eigvals) * torch.einsum(
            'bij,bj->bi',
            unnorm_corr_rot,
            normed_diffs * unnorm_corr_diag,
        )
        chi_sqs = torch.sum(normed_diffs ** 2, dim=-1)
        log_probs = - 0.5 * (
            chi_sqs + 2 * log_std.sum(dim=-1)
            - 2 * torch.log(unnorm_corr_diag).sum(dim=-1)
            + log_unnorm_corr_eigvals.sum(dim=-1)
            + self.sample_dimension * np.log(2 * np.pi)
        )

        return samples, log_probs


    @classmethod
    def initialise(
        cls,
        data: ndarray,
        cond: int | Encoding,
        **kwargs,
    ) -> "MultivariateGaussianKernel":
        """
        Compute initialisation parameters for the smearing kernel.

        Parameters
        ----------
        data : ndarray or Iterable[ndarray]
            The data to compute the initialisation parameters for
        cond: int | Encoding
            The dimension or encoding of the conditioning space

        Returns
        -------
        MultivariateGaussianKernel
            An initialized instance of the `MultivariateGaussianKernel` class with models
            and parameters derived from the input data or user-provided models.
        """
        cov = np.cov(data.T)
        mean = np.mean(data, axis=0)

        return cls.initialise_cov(cov, mean, cond, **kwargs)

    @classmethod
    def initialise_cov(
        cls,
        cov: ndarray,
        mean: ndarray,
        cond: int | Encoding,
        mean_model: Optional[Module] = None,
        log_diag_model: Optional[Module] = None,
        log_rot_model: Optional[Module] = None,
        log_std_model: Optional[Module] = None,
        **kwargs,
    ) -> "MultivariateGaussianKernel":

        """
        Initializes a multivariate Gaussian kernel with optional user-defined models for
        mean, log diagonal, log rotation, and log standard deviation.

        Parameters
        ----------
        cov : ndarray
            Covariance matrix. Its shape should be (sample_dim, sample_dim)
        mean: ndarray | None
            Mean vector. Its shape should be (sample_dim,)
        cond: int | Encoding
            The dimension or encoding of the conditioning information
        mean_model
            Optional model that constructs the mean of the distribution for the given conditioning information
        log_diag_model
            Optional model that constructs the log diagonal matrix of the distribution for the given conditioning information.
        log_rot_model
            Optional model that constructs the log rotational matrix of the distribution for the given conditioning
            information. Must use an AntiSymmetricMatrixEncoding
        log_std_model
            Optional model that constructs the log standard deviation of the distribution for the given conditioning information.

        Returns
        -------
        MultivariateGaussianKernel
            An initialized instance of the `MultivariateGaussianKernel` class with models
            and parameters derived from the input data or user-provided models.
        """
        sample_dim = cov.shape[0]
        log_corr_eigvals, log_corr_rot, log_std = construct_init_parameters(cov)

        mean_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(scale=np.exp(log_std), shift=mean)],
        ) if mean_model is None else mean_model

        log_diag_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_corr_eigvals)],
        ) if log_diag_model is None else log_diag_model

        log_rot_model = basic_model_factory(
            TrivialEncoding(cond),
            AntiSymmetricMatrixEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_corr_rot)],
        ) if log_rot_model is None else log_rot_model

        log_std_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_std)],
        ) if log_std_model is None else log_std_model

        return cls(
            cond=cond,
            sample_dim=sample_dim,
            mean_model=mean_model,
            log_diag_model=log_diag_model,
            log_rot_model=log_rot_model,
            log_std_model=log_std_model,
            **kwargs,
        )

    def log_prob(self,
        samples: torch.Tensor,
        cond: torch.Tensor,
        return_chi_sqs: bool = False,
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
        mean, log_std, unnorm_corr_eigvals, unnorm_corr_rot, unnorm_corr_diag, std, log_unnorm_corr_eigvals = self.construct_variables(cond)


        diffs = samples - mean
        normed_diffs = diffs / torch.exp(log_std)
        normed_diffs = torch.exp(-0.5 * log_unnorm_corr_eigvals) * torch.einsum(
            'bij,bj->bi',
            unnorm_corr_rot,
            normed_diffs * unnorm_corr_diag,
        )
        chi_sqs = torch.sum(normed_diffs ** 2, dim=-1)
        log_prob = - 0.5 * (
            chi_sqs + 2 * log_std.sum(dim=-1)
            - 2 * torch.log(unnorm_corr_diag).sum(dim=-1)
            + log_unnorm_corr_eigvals.sum(dim=-1)
            + self.sample_dimension * np.log(2 * np.pi)
        )
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
            A tensor containing -mean(log_prob) over finite entries.
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
        log_unnorm_corr_rot = self.log_rot_model(cond)
        log_unnorm_corr_eigvals = self.log_diag_model(cond)
        log_std = self.log_std_model(cond)

        unnorm_corr_rot = torch.matrix_exp(log_unnorm_corr_rot)
        cov_tilda = torch.einsum(
            'bji,bj,bjk->bik',
            unnorm_corr_rot,
            log_unnorm_corr_eigvals.exp(),
            unnorm_corr_rot
        )
        root_unnorm_corr_diag = torch.sqrt(torch.diagonal(cov_tilda, dim1=1, dim2=2))

        std = torch.exp(log_std)
        return torch.einsum('bi,bij,bj->bij',
                            std / root_unnorm_corr_diag,
                            cov_tilda,
                            std / root_unnorm_corr_diag
                            )
