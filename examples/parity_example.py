"""
This script reproduces the plots used in the original divergences paper https://arxiv.org/abs/2405.06397
"""

from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from lightning import seed_everything
from numpy._typing import NDArray
from scipy import stats

from iwpc.accumulators.binned_Df_accumulator import BinnedDfAccumulator
from iwpc.calculate_divergence import DivergenceResult, calculate_divergence
from iwpc.data_modules.numpy_data_module import BinaryNumpyDataModule
from iwpc.divergences import KLDivergence, DifferentiableFDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator
from iwpc.scalars.scalar_function import ScalarFunction
from iwpc.utils import format_quantity_with_uncertainty


def calc_parity(vecs: NDArray) -> NDArray:
    """
    Calculates the 'parity' value of an array of vectors, $\vec{x} \cdot (\vec{y} \times \vec{z})$
    Parameters
    ----------
    vecs
        An NDArray of shape (N, 9) where N is the number of samples and 9 corresponds to the 9 components of the three
        3-vectors

    Returns
    -------
    NDArray
        Of shape (N,) containing the parity value of each sample
    """
    vecs = np.asarray(vecs)
    if vecs.shape[-1] == 10:
        return vecs[:, -1]
    if vecs.shape[-1] == 9:
        vecs = vecs.reshape((-1, 3, 3))
    return (np.cross(vecs[:, 0, :], vecs[:, 1, :]) * vecs[:, 2, :]).sum(axis=-1)


def generate_samples(num_trials: int, asymmetry: float) -> NDArray:
    """
    Generates a list of num_trials samples of three 3-vectors with a degree of parity asymmetry specified by the
    'asymmetry' parameter

    Parameters
    ----------
    num_trials
        The number of samples to generate
    asymmetry
        A number between 0 and 1

    Returns
    -------
    NDArray
        An array of shape (num_trials, 9) containing the samples
    """
    vecs = np.random.normal(size=(num_trials, 9))
    parity = calc_parity(vecs)
    flip = np.random.random(size=vecs.shape[0]) < asymmetry / (1 + np.exp(-1. * parity))
    vecs[flip] = - vecs[flip]
    return vecs.astype(np.float32)


def calculate_mag_x_binned_Dfs(mag_x_bins: NDArray, result: DivergenceResult) -> BinnedDfAccumulator:
    """
    Initialises a BinnedDfAccumulator to estimate the divergence of the distributions within each bin of |x|

    Parameters
    ----------
    mag_x_bins
        The bin edges for the |x| bins
    result
        A DivergenceResult instance

    Returns
    -------
    BinnedDfAccumulator
    """
    scalar = ScalarFunction(
        lambda arr: np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2),
        label="|x|",
        latex_label=r"|\vec{x}|",
        bins=mag_x_bins,
    )
    df_accumulator = BinnedDfAccumulator(
        [scalar],
        divergence=result.best_module.divergence,
        estimate_marginalised_from_learned=False,
    )
    for update_fn, dataloader in [
        (df_accumulator.update_train, result.data_module.train_dataloader()),
        (df_accumulator.update_val, result.data_module.val_dataloader()),
    ]:
        for samples, labels, weights in dataloader:
            scalar_values = [scalar(samples.numpy())]
            with torch.no_grad():
                p_over_q = np.exp(result.best_module.model(samples)[:, 0]).numpy()
            update_fn(scalar_values, labels.numpy(), weights.numpy(), p_over_q)

    return df_accumulator


def run_parity_inference_example(
    true_asymmetry: float,
    trial_asymmetries: Iterable[float],
    num_samples: int,
    divergence: DifferentiableFDivergence,
) -> None:
    """
    A simple parity-based example of parameter inference using the divergence framework used to generate the plots in
    https://arxiv.org/abs/2405.06397

    Parameters
    ----------
    true_asymmetry
        The true asymmetry to use for the reference distribution
    trial_asymmetries
        The trial asymmetry to test
    num_samples
        The number of samples to use in the test
    divergence
        Which divergence to calculate
    """
    true_samples = generate_samples(num_samples, true_asymmetry)

    trial_divs = np.zeros_like(trial_asymmetries)
    trial_errs = np.zeros_like(trial_asymmetries)
    for i, trial_asymmetry in enumerate(trial_asymmetries):
        trial_samples = generate_samples(num_samples, trial_asymmetry)
        module = GenericNaiveVariationalFDivergenceEstimator(
            9,
            divergence,
            initial_learning_rate=1e-3,
            lr_decay_factor=None,
            model_factory_kwargs={"hidden_layer_sizes": (128, 64, 64, 64, 64)}
        )
        data_module = BinaryNumpyDataModule(
            true_samples,
            trial_samples,
            dataloader_kwargs={"batch_size": 1024},
        )
        result = calculate_divergence(
            module,
            data_module,
            patience=10,
            trainer_kwargs={'max_epochs': 50},
            name=f"trial_asymmetry_{trial_asymmetry}",
        )

        mag_x_bins = np.linspace(0, 4, 15)
        df_accumulator = calculate_mag_x_binned_Dfs(mag_x_bins, result)
        plt.subplots(1, 2, figsize=(9, 3))
        plt.subplot(121)
        plt.xlabel(r'$|\vec{x}|$')
        plt.ylabel(r'Count')
        mag_x_bin_centers = (mag_x_bins[1:] + mag_x_bins[:-1]) / 2
        true_mag_x = np.sqrt(true_samples[:, 0] ** 2 + true_samples[:, 1] ** 2 + true_samples[:, 2] ** 2)
        trial_mag_x = np.sqrt(trial_samples[:, 0] ** 2 + trial_samples[:, 1] ** 2 + trial_samples[:, 2] ** 2)
        true_mag_x_hist, _ = np.histogram(true_mag_x, bins=mag_x_bins)
        plt.errorbar(
            mag_x_bin_centers,
            true_mag_x_hist,
            yerr=np.sqrt(true_mag_x_hist),
            markersize=0,
            capsize=3,
            drawstyle='steps-mid',
            label='Data (p)',
        )
        trial_mag_x_hist, _ = np.histogram(trial_mag_x, bins=mag_x_bins)
        plt.errorbar(
            mag_x_bin_centers,
            trial_mag_x_hist,
            yerr=np.sqrt(trial_mag_x_hist),
            markersize=0,
            capsize=3,
            drawstyle='steps-mid',
            label='Model (q)',
        )
        plt.legend()
        plt.subplot(122)
        plt.xlabel(r'$|\vec{x}|$')
        plt.ylabel(r'$\hat{D}_{KL}(p(\cdot \mid |\vec{x}|), q(\cdot \mid |\vec{x}|))$')
        plt.errorbar(
            mag_x_bin_centers,
            df_accumulator.perp_df_hist,
            df_accumulator.perp_df_err_hist,
            fmt='o',
            markersize=3,
            capsize=3,
        )
        plt.tight_layout()
        plt.subplots_adjust(left=0.085, right=0.987, top=0.977, bottom=0.16, wspace=0.23)

        dm, best_module, trial_divs[i], trial_errs[i] = result.data_module, result.best_module, result.divergence, result.divergence_stderr

        new_true_samples = generate_samples(num_samples // 2, true_asymmetry)
        new_trial_samples = generate_samples(num_samples // 2, trial_asymmetry)
        true_parity = calc_parity(new_true_samples)
        trial_parity = calc_parity(new_trial_samples)
        with torch.no_grad():
            y_hat = best_module.model(torch.as_tensor(new_trial_samples)).numpy()[:, 0]
        weights = np.exp(y_hat)
        plt.figure(rf'$\alpha={trial_asymmetry}$', figsize=(10, 3))
        plt.subplot(121)
        _, bins, _ = plt.hist(
            true_parity, bins=100, range=(-10, 10), alpha=0.6, density=True, label='Data'
        )
        plt.hist(
            trial_parity, bins=bins, range=(-10, 10), alpha=0.6, density=True, label='Model'
        )
        plt.xlabel(r'$\vec{x}\cdot(\vec{y}\times\vec{z})$')
        plt.ylabel('Distribution')
        plt.legend()
        plt.subplot(122)
        true_vals, bins = np.histogram(true_parity, bins=100, range=(-10, 10))
        trial_parity_hist, _ = np.histogram(trial_parity, bins=bins, range=(-10, 10))
        trial_reweighted_parity_hist, _ = np.histogram(trial_parity, bins=bins, range=(-10, 10), weights=weights)

        plt.stairs(
            true_vals,
            edges=bins,
            fill=True,
            alpha=0.6,
            label='Data'
        )
        plt.stairs(
            trial_reweighted_parity_hist,
            edges=bins,
            fill=True,
            alpha=0.6,
            label='Re-weighted Model'
        )
        plt.xlabel(r'$\vec{x}\cdot(\vec{y}\times\vec{z})$')
        plt.legend()
        plt.subplots_adjust(left=0.065, right=0.99, top=0.977, bottom=0.16, wspace=0.15)

    for trial_asymmetry, div, err in zip(trial_asymmetries, trial_divs, trial_errs):
        print(f"trial_asymmetry={trial_asymmetry} div={format_quantity_with_uncertainty(div, err, with_sig=True)}")

    plt.subplots(1, 2, gridspec_kw={'width_ratios': [2, 1]}, figsize=(9, 3))
    plt.subplot(121)
    plt.errorbar(trial_asymmetries, trial_divs, yerr=trial_errs, fmt='o', markersize=3, capsize=5)
    plt.axhline(0, c='k', linestyle='--')
    plt.xlabel(r'$\alpha$')
    plt.ylabel(f'{divergence.short_name} divergence lower bound')
    plt.subplot(122)
    plt.errorbar(trial_asymmetries, trial_divs, yerr=trial_errs, fmt='o', markersize=3, capsize=5)
    plt.axhline(0, c='k', linestyle='--')
    plt.xlim(0.6, 0.9)
    plt.ylim(-0.0001, 0.0004)
    plt.xlabel(r'$\alpha$')
    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.987, top=0.977, bottom=0.16, wspace=0.2)
    plt.show()


if __name__ == '__main__':
    seed_everything(12442)
    trial_asymmetries = np.linspace(0, 1.0, 9)
    true_asymmetry = 0.75
    num_samples = 100000
    print("True asymmetries:", true_asymmetry)
    print("Trial asymmetries:", trial_asymmetries)

    symmetric_samples = generate_samples(num_samples, 0.0)
    asymmetric_samples = generate_samples(num_samples, true_asymmetry)

    print("Creating pair plot")
    plt.figure(figsize=(9, 9))
    vec_names = ['x', 'y', 'z']
    for i in range(9):
        for j in range(9):
            ax = plt.subplot(9, 9, 9*i + j + 1)
            x = np.linspace(-5, 5, 100)
            if i == j:
                sns.histplot(asymmetric_samples[:, i], ax=ax, stat='density', bins=30, binrange=(-4, 4))
                ax.plot(x, stats.norm(loc=0, scale=1).pdf(x), c='r', linewidth=2)
            else:
                sns.histplot(x=asymmetric_samples[:, i], y=asymmetric_samples[:, j], stat='density', ax=ax, bins=30, binrange=(-4, 4))

            if i == 8:
                ax.set_xlabel(f'${vec_names[j // 3]}_{j % 3}$')
            else:
                ax.get_xaxis().set_visible(False)
            if j == 0:
                ax.set_ylabel(f'${vec_names[i // 3]}_{i % 3}$')
            else:
                ax.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.07, bottom=0.06, right=0.99, top=0.99, wspace=0.07, hspace=0.07)

    run_parity_inference_example(
        true_asymmetry,
        trial_asymmetries,
        num_samples,
        KLDivergence(),
    )
