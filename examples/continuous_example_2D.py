import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import DataFrame
from scipy import stats
from scipy.integrate import quad

from iwpc.data_modules.pandas_data_module import BinaryPandasDataModule
from iwpc.divergences import JensenShannonDivergence, KLDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator
from iwpc.calculate_divergence import calculate_divergence


def make_continuous_samples(num_samples, eps):
    """
    Samples (num_samples * eps) samples from a uniform distribution between (-pi,pi) and (num_samples * (1 - eps))
    from a cosine distribution. Has the effective pdf of (1 + eps*cos(x)) / 2 / pi. Returns these angles and an array of
    2D units vectors pointing in the direction of each angle with magnitude sampled from a normal distribution with
    mean 1.0 and standard deviation 0.1
    """
    num_background = int(num_samples * (1 - eps))
    num_signal = num_samples - num_background
    background_samples = np.random.uniform(-np.pi, np.pi, size=num_background)
    signal_samples = stats.cosine().rvs(num_signal)
    angles = np.concatenate([background_samples, signal_samples])
    np.random.shuffle(angles)
    r = np.random.normal(1.0, 0.1, size=num_signal+num_background)
    df = DataFrame({
        'angles': angles,
        'x': r * np.cos(angles),
        'y': r * np.sin(angles),
    })

    return df


def dist_pdf(x, eps):
    """
    The pdf of the sampling function in make_continuous_samples for a given eps
    """
    return (1 + eps * np.cos(x)) / 2 / np.pi


def KL(pdf1, pdf2, a=-np.pi, b=np.pi):
    """
    Calculates the KL divergence between pdf1 and pdf2 using numerical integration
    """
    return quad(lambda x: pdf1(x) * np.log(pdf1(x) / pdf2(x)), a, b)[0]


def numerically_integrate_divergence(divergence, pdf1, pdf2, a=-np.pi, b=np.pi):
    return quad(lambda x: pdf2(x) * divergence.f(pdf1(x) / pdf2(x)), a, b)[0]


def run_continuous_example(
    sample_sizes,
    divergence,
    eps1,
    eps2=0.,
    verbose_plot=False
):
    """
    Demonstrates the convergence of the calculate_divergence function as the number of samples increases. Samples are
    2D unit vectors constructed from samples from an angular distribution with asymmetry parameter eps1 and the another
    with asymmetry parameter eps2

    Parameters
    ----------
    sample_sizes
        An array of (presumably increasing) integers giving a sequence of sample sizes on which calculate_divergence
        will be called
    divergence_name
        The name of the divergence to calculate. Can be 'TV', 'KL', 'JSD'
    eps1
        The asymmetry parameter of the first distribution
    eps2
        The asymmetry parameter of the second distribution
    verbose_plot
        Whether to plot the learned function for each sample_size after convergence
    """
    numerical_divergence = numerically_integrate_divergence(
        divergence,
        lambda x: dist_pdf(x, eps1),
        lambda x: dist_pdf(x, eps2),
    )

    Dfs = np.zeros(sample_sizes.shape[0])
    Dferrs = np.zeros_like(Dfs)
    for i, num_samples in enumerate(sample_sizes):
        vecs1 = make_continuous_samples(num_samples, eps1)
        vecs2 = make_continuous_samples(num_samples, eps2)
        dm = BinaryPandasDataModule(
            vecs1,
            vecs2,
            feature_cols=['x', 'y'],
            dataloader_kwargs={'batch_size': 256, 'num_workers': 0}
        )
        module = GenericNaiveVariationalFDivergenceEstimator(dm.num_features, divergence)

        result = calculate_divergence(
            module,
            dm,
            trainer_kwargs={
                'accelerator': 'mps' if torch.backends.mps.is_available() else 'cpu',
                'max_epochs': 5,
            },
            name=f"sample_size_{num_samples}"
        )

        Dfs[i] = result.divergence
        Dferrs[i] = result.divergence_stderr

        if verbose_plot:
            plt.hist(vecs1.angles[:num_samples], bins=100, alpha=0.5, density=True)
            plt.hist(vecs2.angles[:num_samples], bins=100, alpha=0.5, density=True)
            bins = np.linspace(-np.pi, np.pi, 100)
            with torch.no_grad():
                plt.plot(bins, result.best_module.model(torch.as_tensor(np.asarray([np.cos(bins), np.sin(bins)]).T.astype(np.float32))).numpy())
            plt.show()

    sigs = Dfs / Dferrs
    print(f"{sample_sizes=}")
    print(f"{Dfs=}")
    print(f"{Dferrs=}")
    print(f"{sigs=}")

    plt.errorbar(sample_sizes, Dfs, yerr=Dferrs, fmt='o', markersize=3, capsize=5)
    plt.axhline(numerical_divergence, label=f'True {divergence.short_name}')
    plt.ylim(0, plt.ylim()[1])
    plt.xlabel('Sample size')
    plt.ylabel(f'{divergence.name} divergence lower bound')
    plt.legend()

    plt.figure()
    plt.scatter(sample_sizes, sigs)
    plt.xlabel('Sample size')
    plt.ylabel(f'{divergence.short_name} divergence significance')
    plt.show()


if __name__ == '__main__':
    """
    See run_continuous_example docstring
    """
    # seed = 24417
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    eps1 = 0.2
    eps2 = 0.0

    sample_sizes = np.linspace(1000, 50000, 3, dtype=int)
    run_continuous_example(sample_sizes, KLDivergence(), eps1=eps1, eps2=eps2)
    run_continuous_example(sample_sizes, JensenShannonDivergence(), eps1=eps1, eps2=eps2)
