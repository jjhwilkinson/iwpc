from typing import List, Callable, Union, Optional, Tuple, Iterable

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy._typing import NDArray
from pandas import DataFrame
from tqdm import tqdm

from .Df_accumulator import LabeledBinaryNaiveDfAccumulator
from .binned_stat_accumulator import BinnedStatAccumulator
from .binned_weighted_stat_accumulator import BinnedWeightedStatAccumulator
from .histogram_accumulator import HistogramAccumulator
from .utils import construct_bin_number_regular_bins
from ..data_modules.pandas_directory_data_module import PandasDirDataModule
from ..divergences import DifferentiableFDivergence
from ..scalars.scalar import Scalar
from ..stat_utils import propagate_uncertainty_through_ratio, normalised_weight_sum_uncertainty
from ..utils import bin_centers, format_quantity_with_uncertainty


def construct_p_over_q(df, p_over_q_cols):
    p_over_q = np.ones(df.shape[0])
    for col in p_over_q_cols:
        p_over_q = p_over_q * df[col].values
    return np.clip(p_over_q, 1e-6, 1e6)


class BinnedDfAccumulator:
    """
    Accumulator which tracks a number of useful metrics for understand the differences in the distribution over some set
    of variables under two distributions. In particular, this accumulator produces a plot which is particularly useful
    when trying to understand the differences found between two distributions by a neural network. Setup of an instance
    performs a pass over the training data accumulating values via '_update_train' and a pass over the validation
    data and calling '_update_val'. The plot method provides a set of standard plots showing how the divergence varies
    as a function of the provided scalars
    """
    def __init__(
        self,
        scalars: Union[Scalar, List[Scalar]],
        divergence: DifferentiableFDivergence,
        p_name: str = 'p',
        q_name: str = 'q',
        filter: Callable[[DataFrame], DataFrame] = None,
        estimate_marginalised_from_learned: bool = True,
    ):
        """
        Parameters
        ----------
        scalars
            A list of Scalars of length S. The accumulator tracks an S-dimensional histogram in each distribution as
            well as an estimate of the divergence within each bin amongst other useful features
        divergence
            A DifferentiableFDivergence instance
        p_name
            The name of the first distribution
        q_name
            The name of the second distribution
        filter
            An optional filter function. If provided, the accumulator is only evaluated on the samples returned by the
            filter function
        estimate_marginalised_from_learned
            In the process of calculating various quantities, the accumulator requires an estimate of the marginalised
            distribution of each distribution in the S scalars. If estimate_marginalised_from_learned=False these
            distributions are estimated from the binned histograms in the S scalars for each distribution. If
            estimate_marginalised_from_learned=True, the marginalised distributions are derived from the likelihood
            ratio between the distributions learnt by the network. estimate_marginalised_from_learned=True is
            recommended unless you know what you're doing
        """
        super().__init__()
        self.scalars = [scalars] if isinstance(scalars, Scalar) else scalars
        self.divergence = divergence
        self.p_name = p_name
        self.q_name = q_name
        self.filter = filter
        self.estimate_marginalised_from_learned = estimate_marginalised_from_learned
        self.bins = [scalar.bins for scalar in self.scalars]

        self.train_p_hist = HistogramAccumulator(self.bins)
        self.train_q_hist = HistogramAccumulator(self.bins)
        self.val_p_hist = HistogramAccumulator(self.bins)
        self.val_q_hist = HistogramAccumulator(self.bins)
        self.perp_p_accumulator = BinnedWeightedStatAccumulator(self.bins)
        self.perp_q_accumulator = BinnedWeightedStatAccumulator(self.bins)
        self.train_learned_p = HistogramAccumulator(self.bins)
        self.train_learned_q = HistogramAccumulator(self.bins)
        self.train_learned_dists = BinnedStatAccumulator(3, self.bins)
        self.val_learned_dists = BinnedStatAccumulator(3, self.bins)

        self.marginalised_df_accumulator = LabeledBinaryNaiveDfAccumulator(
            divergence,
            p_name=f"marginalised {self.p_name}",
            q_name=f"marginalised {self.q_name}"
        )
        self.global_df_accumulator = LabeledBinaryNaiveDfAccumulator(
            divergence,
            p_name=self.p_name,
            q_name=self.q_name
        )

    def _update_train(
        self,
        samples: Union[NDArray, List[NDArray]],
        labels: NDArray,
        weights: NDArray,
        p_over_q: NDArray,
    ) -> None:
        """
        Updates various internal states with training data that may be used in the construction of marginalised
        likelihood functions without biasing the validation result

        Parameters
        ----------
        samples
            The values provided by each scalar
        labels
            The labels of each sample. 0 for p and 1 for q
        weights
            The weights of each sample
        p_over_q
            The predicted likelihood ratio between p and q for this sample
        """
        if isinstance(samples, list):
            samples = np.asarray(samples).T
        is_p = labels == 0
        is_q = ~is_p
        self.train_p_hist.update(
            samples[is_p],
            weights[is_p],
        )
        self.train_q_hist.update(
            samples[is_q],
            weights[is_q],
        )
        self.train_learned_dists.update(
            samples,
            [weights, weights * 2 * (p_over_q / (1 + p_over_q)), weights * 2 * (1 / (1 + p_over_q))],
        )
        self.train_learned_p.update(
            samples,
            weights * 2 * (p_over_q / (1 + p_over_q)),
        )
        self.train_learned_q.update(
            samples,
            weights * 2 * (1 / (1 + p_over_q)),
        )

    # @profile
    def _update_val(
        self,
        samples: Union[NDArray, List[NDArray]],
        labels: NDArray,
        weights: NDArray,
        p_over_q: NDArray,
    ):
        """
        Updates various internal states with validation data that allow for the estimation of the divergence between p
        and q in the provided scalars, and within each bin orthogonal to the provided scalars

        Parameters
        ----------
        samples
            The values provided by each scalar
        labels
            The labels of each sample. 0 for p and 1 for q
        weights
            The weights of each sample
        p_over_q
            The predicted likelihood ratio between p and q for this sample
        """
        if isinstance(samples, list):
            samples = np.asarray(samples).T
        if samples.ndim == 1:
            samples = samples[:, np.newaxis]

        binnumber = construct_bin_number_regular_bins(samples, self.bins).astype(int)
        mask = np.ones(samples.shape[0], dtype=bool)

        for i, bins in enumerate(self.bins):
            mask = mask & (0 <= binnumber[:, i]) & (binnumber[:, i] < (bins.shape[0] - 1))

        is_p = (labels == 0) & mask
        is_q = (labels == 1) & mask

        if self.estimate_marginalised_from_learned:
            marginalised_p = self.train_learned_p.normalised_weight_sum_hist.__getitem__(tuple(binnumber.T[:, mask]))
            marginalised_q = self.train_learned_q.normalised_weight_sum_hist.__getitem__(tuple(binnumber.T[:, mask]))
        else:
            marginalised_p = self.train_p_hist.normalised_weight_sum_hist.__getitem__(tuple(binnumber.T[:, mask]))
            marginalised_q = self.train_q_hist.normalised_weight_sum_hist.__getitem__(tuple(binnumber.T[:, mask]))
        marginalised_p_over_q = marginalised_p / marginalised_q
        marginalised_p_over_q[(marginalised_p == 0) & (marginalised_q == 0)] = 1.
        cond_p_over_q = p_over_q[mask] / marginalised_p_over_q

        learned_p_values = self.divergence.calculate_naive_p_values(cond_p_over_q[is_p[mask]])
        learned_q_values = self.divergence.calculate_naive_q_values(cond_p_over_q[is_q[mask]])
        p_result = self.perp_p_accumulator.update(
            samples[is_p],
            learned_p_values,
            weights[is_p],
        )
        q_result = self.perp_q_accumulator.update(
            samples[is_q],
            learned_q_values,
            weights[is_q],
        )
        self.val_p_hist.update(
            samples[is_p],
            weights[is_p],
            prev_binned_statistic_result=p_result,
        )
        self.val_q_hist.update(
            samples[is_q],
            weights[is_q],
            prev_binned_statistic_result=q_result,
        )
        self.val_learned_dists.update(
            samples,
            [weights, weights * 2 * (p_over_q / (1 + p_over_q)), weights * 2 * (1 / (1 + p_over_q))],
        )

        self.marginalised_df_accumulator.update(
            marginalised_p_over_q,
            labels[mask],
            weights[mask],
        )
        self.global_df_accumulator.update(
            p_over_q,
            labels,
            weights
        )

    @property
    def perp_df_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An S-dimensional array containing an estimate for the lower bound of the divergence of p and q conditioned
            on each bin across the remaining degrees of freedom. See perp_df_err_hist for uncertainties
        """
        return self.perp_p_accumulator.weighted_mean_hist - self.perp_q_accumulator.weighted_mean_hist

    @property
    def perp_df_err_hist(self) -> NDArray:
        """
        Returns
        -------
        NDArray
            An S-dimensional array containing an estimate of the standard error on the lower bound estimators provided
            by self.perp_df_hist
        """
        return np.sqrt(self.perp_p_accumulator.weighted_stderr_hist ** 2 + self.perp_q_accumulator.weighted_stderr_hist ** 2)

    @property
    def weighted_df_avg(self) -> float:
        """
        Returns
        -------
            self.perp_df_hist provides an estimate of the divergence between the distributions of p and q conditioned on
            a particular bin across the remaining degrees of in the distributions. This function returns a weighted
            average of these lower bounds with weights taken as the inverse of the variance given by
            self.perp_df_err_hist**2
        """
        means, errs = self.perp_df_hist, self.perp_df_err_hist
        means, errs = means.flatten(), errs.flatten()
        mask = errs > 0
        means, errs = means[mask], errs[mask]
        weights = 1 / errs**2
        weighted_mean = (weights * means).sum() / weights.sum()
        return weighted_mean

    @property
    def weighted_df_avg_err(self) -> float:
        """
        Returns
        -------
            self.perp_df_hist provides an estimate of the divergence between the distributions of p and q conditioned on
            a particular bin across the remaining degrees of in the distributions. This function returns a weighted
            average of these lower bounds with weights taken as the inverse of the variance given by
            self.perp_df_err_hist**2
        """
        means, errs = self.perp_df_hist, self.perp_df_err_hist
        means, errs = means.flatten(), errs.flatten()
        mask = errs > 0
        means, errs = means[mask], errs[mask]
        weights = 1 / errs**2
        weighted_mean = (weights * means).sum() / weights.sum()
        return weighted_mean

    @property
    def variability_chi_sq_dof(self) -> float:
        """
        Returns
        -------
            Quick and dirty function which calculates the chi-square per dof of the marginalised divergences given by
            self.perp_df_hist with their average value given by self.weighted_df_avg
        """
        means, errs = self.perp_df_hist, self.perp_df_err_hist
        means, errs = means.flatten(), errs.flatten()
        mask = errs > 0
        means, errs = means[mask], errs[mask]
        chi_sq_per_dof = (((means - self.weighted_df_avg) / errs)**2).sum() / len(means)

        return chi_sq_per_dof

    def plot(
        self,
        title: Optional[str] = None,
        vmin: Optional[float] = 0,
        vmax: Optional[float] = 2.0,
        max_Df: Optional[float] = None,
        log_dists: bool = False,
    ) -> Tuple[Figure, Tuple[Axes, Axes, Axes, Axes]]:
        """
        Creates a 4-panel plot summarising the trends of the divergence as a function of the provided scalars for
        significantly more detail, see the package readme examples. Note that plt.show() must be called manually

        Parameters
        ----------
        title
            An optional title for the figure
        vmin
            An optional parameter which sets the minimum value for the colour scale of 2D ratio plots
        vmax
            An optional parameter which sets the maximum value for the colour scale of 2D ratio plots
        max_Df
            An optional parameter which sets the maximum value for the colour scale of 2D marginalised divergence plots
        log_dists
            Whether to plot distributions on a log scale

        Returns
        -------
        Tuple[Figure, Tuple[Axes, Axes, Axes, Axes]]
            The figure and axes of the plot
        """
        if len(self.bins) > 2:
            raise NotImplementedError("Plotting not implemented for more than 2 scalars")

        perp_df, perp_df_errs = self.perp_df_hist, self.perp_df_err_hist
        train_p_sum, train_p_errs = self.train_p_hist.weight_sum_hist, self.train_p_hist.weight_sum_stderr_hist
        train_q_sum, train_q_errs = self.train_q_hist.weight_sum_hist, self.train_q_hist.weight_sum_stderr_hist
        val_p_sum, val_p_errs = self.val_p_hist.weight_sum_hist, self.val_p_hist.weight_sum_stderr_hist
        val_q_sum, val_q_errs = self.val_q_hist.weight_sum_hist, self.val_q_hist.weight_sum_stderr_hist

        marginalised_df_str = format_quantity_with_uncertainty(
            self.marginalised_df_accumulator.accumulated_df,
            self.marginalised_df_accumulator.accumulated_df_stderr,
        )
        global_df_str = format_quantity_with_uncertainty(
            self.global_df_accumulator.accumulated_df,
            self.global_df_accumulator.accumulated_df_stderr,
        )
        title = f"{title}\n" if title else ''
        title += f"Global {self.divergence.short_name}: {global_df_str}"

        if len(self.scalars) == 1:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 9))
            if title:
                plt.text(0.5, 0.95, title, transform=fig.transFigure, horizontalalignment='center')
            scalar = self.scalars[0]
            centers = bin_centers(scalar.bins)

            for ax in (ax0, ax1, ax2, ax3):
                ax.set_xlabel(scalar.latex_label)

            ax0.set_ylabel('Weight Sum')
            ax1.set_ylabel(f'{self.divergence.short_name}')
            ax2.set_ylabel('Weight Sum')
            ax3.set_ylabel(f'{self.p_name} / {self.q_name}')

            ax0.errorbar(
                centers,
                val_p_sum,
                yerr=val_p_errs,
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f"val {self.p_name}",
            )
            ax0.errorbar(
                centers,
                val_q_sum,
                yerr=val_q_errs,
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f"val {self.q_name}",
            )
            plt.text(
                0.125,
                0.89,
                f"Marginalised {self.divergence.short_name}: {marginalised_df_str}",
                transform=fig.transFigure,
            )
            ax0.legend()

            ax1.errorbar(
                centers,
                perp_df,
                yerr=perp_df_errs,
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
            )
            ax1.axhline(self.weighted_df_avg, label=f'Avg: {self.weighted_df_avg:.2E}')
            ax1.legend()
            ax1.set_ylim(0, max_Df)

            ax2.errorbar(
                centers,
                self.val_learned_dists.sum_hist[1] / self.val_learned_dists.sum_hist[1].sum(),
                yerr=normalised_weight_sum_uncertainty(
                    self.val_learned_dists.sum_hist[1],
                    np.sqrt(self.val_learned_dists.sq_sum_hist[1, 1])
                ),
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f'learned {self.p_name} hist',
            )
            ax2.errorbar(
                centers,
                self.val_learned_dists.sum_hist[2] / self.val_learned_dists.sum_hist[2].sum(),
                yerr=normalised_weight_sum_uncertainty(
                    self.val_learned_dists.sum_hist[2],
                    np.sqrt(self.val_learned_dists.sq_sum_hist[2, 2])
                ),
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f'learned {self.q_name} hist',
            )
            ax2.legend()

            ax3.errorbar(
                centers,
                val_p_sum / val_q_sum,
                yerr=propagate_uncertainty_through_ratio(
                    val_p_sum,
                    val_q_sum,
                    np.asarray([[val_p_errs**2, np.zeros_like(val_p_errs)], [np.zeros_like(val_q_errs), val_q_errs**2]])
                ),
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f'val {self.p_name} / {self.q_name}'
            )
            ax3.errorbar(
                centers,
                self.val_learned_dists.sum_hist[1] / self.val_learned_dists.sum_hist[2] / (self.val_learned_dists.sum_hist[1].sum() / self.val_learned_dists.sum_hist[2].sum()),
                yerr=propagate_uncertainty_through_ratio(
                    self.val_learned_dists.sum_hist[1],
                    self.val_learned_dists.sum_hist[2],
                    self.val_learned_dists.sq_sum_hist[1:, 1:]
                ),
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f'val learned {self.p_name} / {self.q_name}'
            )
            ax3.errorbar(
                centers,
                train_p_sum / train_q_sum,
                yerr=propagate_uncertainty_through_ratio(
                    train_p_sum,
                    train_q_sum,
                    np.asarray([[train_p_errs**2, np.zeros_like(train_p_errs)], [np.zeros_like(train_q_errs), train_q_errs**2]])
                ),
                markersize=0,
                capsize=3,
                drawstyle='steps-mid',
                label=f'train {self.p_name} / {self.q_name}'
            )
            plt.axhline(1.0, color='k', linestyle='--')
            ax3.legend()

            return fig, (ax0, ax1, ax2, ax3)
        elif len(self.bins) == 2:
            fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2, figsize=(15, 9))
            if title:
                plt.text(0.5, 0.95, title, transform=fig.transFigure, horizontalalignment='center')
            scalar1, scalar2 = self.scalars

            for ax in (ax0, ax1, ax2, ax3):
                ax.set_xlabel(scalar1.latex_label)
                ax.set_ylabel(scalar2.latex_label)

            im = ax0.imshow(
                (val_p_sum / val_q_sum).T,
                origin='lower',
                extent=(scalar1.bins[0], scalar1.bins[-1], scalar2.bins[0], scalar2.bins[-1]),
                cmap='bwr',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
            )
            plt.text(
                0.125,
                0.89,
                f"Marginalised {self.divergence.short_name}: {marginalised_df_str}",
                transform=fig.transFigure,
            )
            fig.colorbar(im, orientation='vertical', label=f'validation {self.p_name} / {self.q_name}')
            im = ax1.imshow(
                perp_df.T,
                origin='lower',
                extent=(scalar1.bins[0], scalar1.bins[-1], scalar2.bins[0], scalar2.bins[-1]),
                aspect='auto',
                interpolation='nearest',
                vmin=0,
                vmax=max_Df,
            )
            plt.text(
                0.55,
                0.89,
                f"Average conditional {self.divergence.short_name}: {self.weighted_df_avg:.2E}",
                transform=fig.transFigure,
            )
            fig.colorbar(im, orientation='vertical', label=f'Conditioned {self.divergence.short_name} lower bound')

            im = ax2.imshow(
                (self.val_learned_dists.sum_hist[1] / self.val_learned_dists.sum_hist[2]).T,
                origin='lower',
                extent=(scalar1.bins[0], scalar1.bins[-1], scalar2.bins[0], scalar2.bins[-1]),
                cmap='bwr',
                aspect='auto',
                vmin=vmin,
                vmax=vmax,
                interpolation='nearest',
            )
            fig.colorbar(im, orientation='vertical', label=f'Learned {self.p_name} / {self.q_name}')

            im = ax3.imshow(
                np.log(val_p_sum.T) if log_dists else train_p_sum.T,
                origin='lower',
                extent=(scalar1.bins[0], scalar1.bins[-1], scalar2.bins[0], scalar2.bins[-1]),
                aspect='auto',
                interpolation='nearest',
            )
            fig.colorbar(
                im,
                orientation='vertical',
                label=f'validation {self.p_name} log weight sum' if log_dists else f"validation {self.p_name} weight sum"
            )

            return fig, (ax0, ax1, ax2, ax3)

    def _prep_data(self, df: DataFrame) -> DataFrame:
        """
        Applies the filter and any other transformations to an input dataframe before being passed to the scalars

        Parameters
        ----------
        df
            A dataframe

        Returns
        -------
        DataFrame
            A dataframe of the filtered data
        """
        if self.filter:
            df = self.filter(df)
        return df

    def evaluate(
        self,
        datamodule: PandasDirDataModule,
        p_over_q_cols: Iterable[str],
    ) -> None:
        """
        Initialises the accumulator using the contents of a PandasDirDataModule. Iterates over the files in the
        datamodule, evaluating the scalars on the dataframes, and calling the relevant update functions

        Parameters
        ----------
        datamodule
            A PandasDirDataModule instance
        p_over_q_cols
            The names of columns which should be multiplied together to use as an estimate for the likelihood ratio
            $\frac{p(x)}{q(x)}$
        """
        for update_fn, include_train, include_val, num_files in [
            (self._update_train, True, False, len(datamodule.train_files)),
            (self._update_val, False, True, len(datamodule.validation_files)),
        ]:
            for file, df in tqdm(
                datamodule.file_iter(include_train_files=include_train, include_validation_files=include_val),
                total=num_files,
            ):
                df = self._prep_data(df)
                scalar_values = [scalar(df) for scalar in self.scalars]
                labels = df.label.values.astype(bool)
                weights = df[datamodule.weight_col].values if datamodule.weight_col else np.ones_like(labels, dtype=float)
                p_over_q = construct_p_over_q(df, p_over_q_cols)
                update_fn(scalar_values, labels, weights, p_over_q)
