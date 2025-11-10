import numpy as np
import torch
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.encodings.matrix_encoding import MatrixEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.learn_dist.kernels.labelled_kernel_trainer import LabelledKernelTrainer
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MultivariateNormalKernel(TrainableKernelBase):
    """
    A multidimentional Normal kernel with trainable mean and std deviations
    """
    def __init__(
        self,
        cond,
        sample_dim
    ):
        """
        Parameters
        ----------
        cond
            The conditioning space encoding or dimension
        mean_model
            Optional model that constructs the mean of the distribution for the given conditioning information
        log_diag_model
            Optional model that constructs the log diagonal matrix of the distribution for the given conditioning information.
        log_rot_model
            Optional model that constructs the log rotational matrix of the distribution for the given conditioning information.
        """
        super().__init__(sample_dim, cond)
        self.mean_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim))
        self.log_diag_model = basic_model_factory(TrivialEncoding(cond), TrivialEncoding(sample_dim))
        self.log_rot_model = basic_model_factory(TrivialEncoding(cond), MatrixEncoding(sample_dim))

    def _draw(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Returns
        -------
        Tensor
            A sample from the gaussian kernel for each row of conditioning information
        """
        raise NotImplementedError()

    def log_prob(self, samples: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
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
        chi_sqs_M = torch.sum(normed_diffs ** 2, dim=-1)
        log_prob = - 0.5 * (chi_sqs_M + log_diags.sum(dim=-1) + self.sample_dimension*np.log(2 * np.pi))
        return log_prob

    def construct_cov(self, cond: torch.Tensor):
        """
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