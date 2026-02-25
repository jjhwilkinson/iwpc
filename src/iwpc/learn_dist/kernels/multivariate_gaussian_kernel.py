from typing import Optional, Iterable

import numpy as np
import torch
from numpy import ndarray
from scipy.linalg import logm
from torch.nn import Module

from iwpc.encodings.antisymmetric_matrix_encoding import AntisymMatrixEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.layers import ConstantScaleLayer
from iwpc.models.utils import basic_model_factory



def initial_guess(
    cov: Optional[ndarray] = None,
    data: Optional[ndarray | Iterable[ndarray]] = None,
) -> tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Compute initialisation parameters for the smearing kernel.

    Parameters
    ----------
    data : PandasDirDataModule
        The data to compute the initialisation parameters for.
    cov: ndarray | None
        The covariance matrix to compute the initialisation parameters for.

    Returns
    -------
    mean_scale : numpy.ndarray
        The square roots of the diagonal entries of the covariance matrix, i.e. the
        per-parameter standard deviations, used to scale the mean model outputs.

    log_diag_shift : numpy.ndarray
        The logarithm of the eigenvalues of the correlation matrix, used as an initial
        shift for the log-diagonal covariance model.

    log_rot_shift : numpy.ndarray
        Half of the matrix logarithm of the eigenvector matrix of the correlation
        (flattened to 1D), used as an initial shift for the rotation model.

    log_std_shift : numpy.ndarray
        The logarithm of the standard deviations (square roots of the covariance
        diagonal), used as an initial shift for the log-standard-deviation model.
    """
    cov = np.cov(data.T) if data is not None else cov
    corr = cov / np.sqrt(np.diag(cov))[:, None] / np.sqrt(np.diag(cov))[None, :]
    corr_diagonal, corr_rotation = np.linalg.eigh(corr)

    mean_scale = np.sqrt(np.diag(cov))
    log_diag_shift = np.log(corr_diagonal)
    log_rot_shift = logm(corr_rotation).reshape(-1)
    log_std_shift = np.log(np.sqrt(np.diag(cov)))

    return mean_scale, log_diag_shift, log_rot_shift, log_std_shift


class MultivariateGaussianKernel(TrainableKernelBase):
    """
    Multivariate normal kernel with mean and covariance conditioned on `cond`. The
    mean is predicted by `mean_model(cond)`. The covariance is parameterised as
    “std-dev × correlation × std-dev”.

    
    An explanation of the covariance matrix's parameterisation is as follows:

    The std-dev is produced by `log_std_model(cond)` and exponentiated so it stays 
    positive. The correlation matrix is built in a way that guarantees it is 
    well-defined; we start from a positive diagonal spectrum, given by 
    `exp(log_diag_model(cond))`.
    
    We then rotate it using an orthogonal mixing matrix derived from 
    `log_rot_model(cond)`, via the antisymmetric part and a matrix exponential, 
    to introduce correlations while preserving positive definiteness.
    
    We then re-enforce normalisation so the diagonal entries are 1 by dividing by 
    the square root of the diagonal entries, giving a pure correlation matrix. 
    Finally, the std-dev scales are applied on both axes, to give the final 
    covariance matrix.

    We use this parameterisation because it lets the networks output unconstrained 
    values while still producing a valid covariance; symmetric and positive definite. 
    It also separates standard deviations from correlations, which is typically easier 
    to learn and tends to be more numerically stable than learning an arbitrary full 
    matrix directly.
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
        self.log_rot_model = basic_model_factory(TrivialEncoding(cond), AntisymMatrixEncoding(sample_dim)) if log_rot_model is None else log_rot_model
        self.log_std_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim)) if log_std_model is None else log_std_model
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

    @classmethod
    def initialise(
        cls,
        data: ndarray | Iterable[ndarray],
        cond: int | torch.Tensor,
        **kwargs,
    ) -> "MultivariateGaussianKernel":
        """
        Compute initialisation parameters for the smearing kernel.

        Parameters
        ----------
        data : ndarray or Iterable[ndarray]
            The data to compute the initialisation parameters for. Can be a single
            array or an iterable of chunks (e.g. a list of arrays, a generator of
            batches from a DataLoader) to avoid loading everything into memory at once.
        cond: int | torch.Tensor
            The conditioning information for each sample

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
        cond: int | torch.Tensor,
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
            Covariance matrix. Its shape should be (sample_dim, sample_dim).
        mean: ndarray | None
            Mean vector. Its shape should be (sample_dim,).
        cond: int | torch.Tensor
            The conditioning information for each sample.

        Returns
        -------
        MultivariateGaussianKernel
            An initialized instance of the `MultivariateGaussianKernel` class with models
            and parameters derived from the input data or user-provided models.
        """
        sample_dim = cov.shape[0]
        mean_scale, log_diag_shift, log_rot_shift, log_std_shift = initial_guess(cov)

        mean_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(scale=mean_scale, shift=mean)],
        ) if mean_model is None else mean_model

        log_diag_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_diag_shift)],
        ) if log_diag_model is None else log_diag_model

        log_rot_model = basic_model_factory(
            TrivialEncoding(cond),
            AntisymMatrixEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_rot_shift)],
        ) if log_rot_model is None else log_rot_model

        log_std_model = basic_model_factory(
            TrivialEncoding(cond),
            TrivialEncoding(sample_dim),
            final_layers=[ConstantScaleLayer(shift=log_std_shift)],
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
        M = self.log_rot_model(cond)
        mean = self.mean_model(cond)
        log_std = self.log_std_model(cond)
        log_diags = self.log_diag_model(cond)

        rot = torch.matrix_exp(M)
        diags = torch.exp(log_diags)
        cov_tilde = torch.einsum('bij,bj,bjk->bik', rot.transpose(1, 2), diags, rot)
        cov_tilde_diag = torch.sqrt(torch.diagonal(cov_tilde, dim1=1, dim2=2))

        diffs = samples - mean
        normed_diffs = diffs / torch.exp(log_std)
        normed_diffs = torch.exp(-0.5 * log_diags) * torch.einsum(
            'bij,bj->bi',
            torch.matrix_exp(M),
            normed_diffs * cov_tilde_diag,
        )
        chi_sqs = torch.sum(normed_diffs ** 2, dim=-1)
        log_prob = - 0.5 * (
            chi_sqs + 2 * log_std.sum(dim=-1)
            - 2 * torch.log(cov_tilde_diag).sum(dim=-1)
            + log_diags.sum(dim=-1)
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
            A tensor containing ``-mean(log_prob)`` over finite entries.
        """
        cond, targets, _ = batch
        log_prob, chi_sqs = self.log_prob(targets, cond, return_chi_sqs = True)
        if self.max_chi is not None:
            mask = chi_sqs < self.max_chi ** 2
            log_prob = log_prob[mask]
        return  - log_prob[log_prob.isfinite()].mean()

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
        log_std = self.log_std_model(cond)

        rot = torch.matrix_exp(M)
        diags = torch.exp(log_diags)
        cov_tilda = torch.einsum('bij,bj,bjk->bik', rot.transpose(1, 2), diags, rot)
        S = torch.sqrt(torch.diagonal(cov_tilda, dim1=1, dim2=2))  # (B,d)

        std = torch.exp(log_std)
        cov = cov_tilda * (std*S).unsqueeze(-2) * (std*S).unsqueeze(-1)
        return cov