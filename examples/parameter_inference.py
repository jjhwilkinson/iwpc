import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from pandas import DataFrame
from scipy.stats import expon

from iwpc.data_modules.pandas_data_module import PandasDataModule, BinaryPandasDataModule
from iwpc.divergences import JensenShannonDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator
from iwpc.calculate_divergence import calculate_divergence


def generate_samples(num_samples, lambda_):
    return DataFrame({'decay_time': expon(scale=lambda_).rvs(size=num_samples)})


def run_inference_example(
    true_lambda,
    trial_lambdas,
    num_samples,
    divergence,
):
    """
    Simple example showing parameter estimation using divergence minimization using an exponential distribution
    Parameters
    ----------
    true_lambda
    trial_lambdas
    num_samples
    divergence

    Returns
    -------
    """
    true_samples = generate_samples(num_samples, true_lambda)

    trial_divs = np.zeros_like(trial_lambdas)
    trial_errs = np.zeros_like(trial_lambdas)
    for i, lambda_ in enumerate(trial_lambdas):
        trial_samples = generate_samples(num_samples, lambda_)
        dm = BinaryPandasDataModule(
            true_samples,
            trial_samples,
            feature_cols=['decay_time'],
            dataloader_kwargs={'batch_size': 256, 'num_workers': 0}
        )
        module = GenericNaiveVariationalFDivergenceEstimator(dm.num_features, divergence)

        result = calculate_divergence(
            module,
            dm,
            trainer_kwargs={
                'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
                'max_epochs': 20,
            },
            name=f'trial_lambda_{lambda_}'
        )
        trial_divs[i] = result.divergence
        trial_errs[i] = result.divergence_stderr

    plt.errorbar(trial_lambdas, trial_divs, yerr=trial_errs, fmt='o', markersize=3, capsize=5)
    plt.axvline(true_lambda, label=r'True $\lambda$')
    plt.axhline(0)
    plt.xlabel(r'Trial $\lambda$')
    plt.ylabel(f'{divergence.short_name} divergence to true samples')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    trial_lambdas = np.linspace(0.2, 3, 10)
    true_lambda = trial_lambdas[4]
    num_samples = 2000

    vals, bins, _ = plt.hist(
        generate_samples(num_samples, true_lambda),
        bins=100,
        range=(0, 2 * trial_lambdas.max()),
        density=True,
        color='k',
        label=rf'True $\lambda={true_lambda:.2f}$'
    )
    for lambda_ in trial_lambdas:
        plt.hist(
            generate_samples(num_samples, lambda_),
            bins=bins,
            density=True,
            label=rf'$\lambda={lambda_:.2f}$',
            histtype='step'
        )
    plt.legend()
    plt.xlabel('Decay time')
    plt.ylabel('Count')
    plt.show()

    run_inference_example(
        true_lambda,
        trial_lambdas,
        num_samples,
        JensenShannonDivergence(),
    )
