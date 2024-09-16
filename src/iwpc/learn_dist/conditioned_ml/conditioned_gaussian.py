import numpy as np
import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.cli import ReduceLROnPlateau
from matplotlib import pyplot as plt
from pandas import DataFrame
from scipy import stats
from torch import Tensor
from torch.optim import Adam

from iwpc.data_modules.pandas_data_module import PandasDataModule
from iwpc.encodings.exponential_encoding import ExponentialEncoding
from iwpc.encodings.trivial_encoding import TrivialEncoding
from iwpc.models.utils import basic_model_factory


class ConditionedMultivariateGaussianFitter(LightningModule):
    def __init__(
        self,
        sample_dimension,
        num_condition_parameters,
        mean_model = None,
        diagonal_model = None,
        off_diagonal_model = None,
    ):
        super().__init__()
        self.sample_dimension = sample_dimension
        self.num_condition_parameters = num_condition_parameters
        self.mean_model = mean_model or basic_model_factory(
            TrivialEncoding(sample_dimension),
            TrivialEncoding(sample_dimension),
        )
        self.diagonal_model = diagonal_model or basic_model_factory(
            TrivialEncoding(sample_dimension),
            ExponentialEncoding(sample_dimension),
        )
        self.off_diagonal_model = off_diagonal_model or basic_model_factory(
            TrivialEncoding(self.off_diagonal_size),
            TrivialEncoding(self.off_diagonal_size),
        )
        self.register_buffer("global_N", torch.tensor(0.))
        self.register_buffer("global_sums", torch.zeros(sample_dimension))
        self.register_buffer("global_sq_sums", torch.zeros(sample_dimension))

        self.register_buffer('two_pi', torch.tensor(2.0 * np.pi))
        self.initial_lr = 1e-3
        self.lr_patience = 15
        self.lr_decay_factor = 0.5

    @property
    def off_diagonal_size(self):
        return self.sample_dimension * (self.sample_dimension - 1) // 2

    @property
    def global_means(self):
        return self.global_sums / self.global_N

    @property
    def global_standard_deviations(self):
        return torch.sqrt(self.global_sq_sums / self.global_N - self.global_means**2)

    def conditioned_log_likelihood(self, samples, means, diagonal, off_diagonal):
        delta = samples - means
        det = torch.prod(diagonal, dim=-1)
        exponent_dot = 0.
        for i in range(self.sample_dimension):
            diagonal_bit = diagonal[:, i] * delta[:, i]
            off_diagonal_start_idx = (i * (2*self.sample_dimension - i - 1)) // 2
            off_diagonal_end_idx = off_diagonal_start_idx + self.sample_dimension - i - 1
            off_diagonal_bit = torch.sum(delta[:, (i+1):] * off_diagonal[:, off_diagonal_start_idx: off_diagonal_end_idx], dim=-1)
            exponent_dot += (diagonal_bit + off_diagonal_bit)**2

        return - 0.5 * self.sample_dimension * torch.log(self.two_pi) + torch.log(det) - 0.5 * torch.clip(exponent_dot, 0, 20)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.initial_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    patience=self.lr_patience,
                    factor=self.lr_decay_factor,
                    monitor='val_loss',
                ),
                "monitor": "val_loss",
                "frequency": 1,
            },
        }

    def evaluate_log_likelihood(self, targets, conditioning_inputs):
        return self.conditioned_log_likelihood(
            (targets - self.global_means[None, :]) / self.global_standard_deviations[None, :],
            self.mean_model(conditioning_inputs),
            self.diagonal_model(conditioning_inputs),
            self.off_diagonal_model(conditioning_inputs) if self.off_diagonal_size > 0 else torch.zeros(tuple(targets.shape)[:-1] + (0,), device=samples.device),
        )

    def evaluate_cross_entropy(self, batch):
        samples, targets, weights = batch
        estimated_log_pdf_values = self.evaluate_log_likelihood(targets, samples)
        return - (weights * estimated_log_pdf_values).sum() / weights.sum()

    def training_step(self, batch):
        if self.current_epoch == 0:
            samples, targets, weights = batch
            self.global_N += targets.shape[0]
            self.global_sums += targets.sum(dim=0)
            self.global_sq_sums += (targets**2).sum(dim=0)
            return

        cross_entropy = self.evaluate_cross_entropy(batch)
        self.log("train_loss", cross_entropy, on_step=True, on_epoch=False, prog_bar=True)
        return cross_entropy

    def validation_step(self, batch):
        if self.current_epoch == 0:
            self.log('val_loss', torch.inf, on_step=False, on_epoch=True, prog_bar=True)
            return
        cross_entropy = self.evaluate_cross_entropy(batch)
        self.log("val_loss", cross_entropy, on_step=False, on_epoch=True, prog_bar=True)
        return cross_entropy

    def covariance(self, samples):
        diagonals = self.diagonal_model(samples)
        off_diagonal = self.off_diagonal_model(samples) if self.off_diagonal_size > 0 else torch.zeros(tuple(samples.shape)[:-1] + (0,), device=samples.device)

        rows = []
        for i in range(self.sample_dimension):
            rows.append(torch.concatenate([
                off_diagonal[:, i * (i - 1) // 2: i * (i + 1) // 2],
                diagonals[:, i:i+1],
                torch.zeros((diagonals.shape[0], self.sample_dimension - (i+1)), device=diagonals.device)
            ], dim=-1))
        A = torch.stack(rows, dim=-2)
        return torch.linalg.inv(torch.matmul(A, torch.transpose(A, -2, -1))) / self.global_standard_deviations[None, :, None]  / self.global_standard_deviations[None, None, :]


def generate_samples_1D(
    size,
    conditioning_parameter_distribution,
    mean_function,
    sigma_function,
):
    """
    Generates a bunch of samples by drawing a 'parameter', lambda_, from conditioning_parameter_distribution, then
    drawing the actual sample from a gaussian with parameters given by mean_function(lambda_) and
    sigma_function(lambda_)

    Returns
    -------
    DataFrame
    """
    lambda_ = conditioning_parameter_distribution.rvs(size)
    means = mean_function(lambda_)
    sigmas = sigma_function(lambda_)
    samples = np.random.normal(loc=means, scale=sigmas, size=size)
    data = DataFrame({
        'sample': samples,
        'conditioning_parameter': lambda_,
        'true_mean': means,
        'true_sigma': sigmas,
    })

    return data


def generate_samples(
    size,
    conditioning_parameter_distribution,
    mean,
    cov,
):
    """
    Generates a bunch of samples by drawing a 'parameter', lambda_, from conditioning_parameter_distribution, then
    drawing the actual sample from a gaussian with parameters given by mean_function(lambda_) and
    sigma_function(lambda_)

    Returns
    -------
    DataFrame
    """
    lambda_ = conditioning_parameter_distribution.rvs(size)
    samples = np.random.multivariate_normal(mean, cov, size=size)
    data = DataFrame({
        **{f'sample_{i}': samples[:, i] for i in range(mean.shape[0])},
        'conditioning_parameter': lambda_,
    })

    return data


if __name__ == "__main__":
    num_samples = 1000000
    conditioning_parameter_distribution = stats.uniform(loc=0, scale=1)
    mean = np.asarray([0, 0, 0])
    cov = np.asarray([[1, 0.5, 0.1], [0.5, 2, -0.2], [0.1, -0.2, 3]])
    samples = generate_samples(
        num_samples,
        conditioning_parameter_distribution,
        mean,
        cov,
    )
    datamodule = PandasDataModule(
        samples,
        feature_cols=['conditioning_parameter'],
        target_cols=[f"sample_{i}" for i in range(mean.shape[0])],
        dataloader_kwargs={'batch_size': 2**15}
    )

    fitter = ConditionedMultivariateGaussianFitter(mean.shape[0], 1)
    trainer = Trainer(max_epochs=50)
    trainer.fit(
        fitter,
        datamodule=datamodule,
    )

    fitter.cpu()
    fitter.eval()
    regressed_mean_model, covariance_function = fitter.mean_model, fitter.covariance

    with torch.no_grad():
        print(regressed_mean_model(torch.tensor([[0]])))
        print(covariance_function(torch.tensor([[0]])))
