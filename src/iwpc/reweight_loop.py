from dataclasses import dataclass
from typing import Dict, Callable, Tuple, List

import numpy as np
from lightning import Trainer
from numpy._typing import NDArray
from pandas import DataFrame
from torch.utils.data import DataLoader

from iwpc.calculate_divergence import calculate_divergence, DivergenceResult
from iwpc.datasets.pandas_dataset import PandasDataset
from iwpc.divergences import DifferentiableFDivergence
from iwpc.modules.fdivergence_base import FDivergenceEstimator
from .accumulators.Df_accumulator import LabeledBinaryNaiveDfAccumulator
from .data_modules.pandas_directory_data_module import PandasDirDataModule


def add_p_over_q_transformation(
    df: DataFrame,
    module: FDivergenceEstimator,
    data_module: PandasDirDataModule,
    p_over_q_col: str,
):
    """
    Accepts a Pandas DataFrame, evaluates the module on its contents, and adds the exponential of the results as a new
    column to the DataFrame as p_over_q_col. When well-trained, this quantity is interpretable as the likelihood ratio
    of the two category distributions, $\frac{p(x)}{q(x)}$.

    Parameters
    ----------
    df
        A DataFrame
    module
        A trained FDivergenceEstimator instance
    data_module
        A datamodule containing the column names needed for evaluation
    p_over_q_col
        The name of the new column to insert exp(module(df))

    Returns
    -------
    DataFrame
        The original dataframe with the new column containing the exponential of the output of the module
    """
    ds = PandasDataset(
        df,
        data_module.feature_cols,
        data_module.target_cols,
        data_module.weight_col,
    )
    trainer = Trainer(enable_checkpointing=False, logger=False)
    log_p_over_q = trainer.predict(model=module, dataloaders=DataLoader(ds, data_module.batch_size))
    p_over_q = np.exp(np.concat(log_p_over_q))
    df[p_over_q_col] = p_over_q
    return df


def reweight_down_from_p_over_q(df: DataFrame, label_col: str, p_over_q_col: str) -> NDArray:
    """
    Calculates a factor by which the samples in the dataframe should be reweighted based on the estimate of
    $\frac{p(x)}{q(x)}$ in p_over_q_col. Every sample from distribution p (label 0) is assigned a reweighting by a
    factor $min(\frac{q(x)}{p(x)}, 1.)$ and samples from q are given a reweight factor $min(\frac{p(x)}{q(x)}, 1.)$.
    If the module used to produce p_over_q_col is perfectly trained this will reweight both distributions to
    $m(x)=min(p(x), q(x))$ so no differences exist between the reweighted distributions

    Parameters
    ----------
    df
        A DataFrame containing label_col, and p_over_q_col
    label_col
        The name of the column containing the class labels. Should containing value 0 for distribution p and 1 for
        distribution q
    p_over_q_col
        The name of a column containing an estimate of $\frac{p(x)}{q(x)}$

    Returns
    -------
    NDArray
        An NDArray containing the reweight factors of each sample
    """
    reweight_value = df[p_over_q_col].values.copy()
    is_p = df[label_col] == 0
    reweight_value[is_p] = 1 / reweight_value[is_p]
    return np.clip(reweight_value, 0, 1.0)


def calculate_total_divergence(
    data_module: PandasDirDataModule,
    divergence: DifferentiableFDivergence,
) -> LabeledBinaryNaiveDfAccumulator:
    """
    Calculates the total divergence using the product of the reweighting columns produced in the reweight loop procedure

    Parameters
    ----------
    data_module
        A PandasDirDataModule with target columns containing first, the label of each sample, and then all the
        p_over_q_{i} columns generated in the reweight loop
    divergence
        A DifferentiableFDivergence instance

    Returns
    -------
    LabeledBinaryNaiveDfAccumulator
    """
    df_accumulator = LabeledBinaryNaiveDfAccumulator(divergence)

    for (_, y, weights) in data_module.val_dataloader():
        labels = y[:, 0].numpy()
        p_over_q = np.prod(y[:, 1:].numpy(), axis=1)
        df_accumulator.update(p_over_q, labels, weights)

    return df_accumulator


@dataclass
class ReweightLoopResult:
    calculate_divergence_results: List[DivergenceResult]
    p_over_q_cols: List[str]
    final_divergence_accumulator: LabeledBinaryNaiveDfAccumulator
    final_data_module: PandasDirDataModule


def run_reweight_loop(
    module_factory: Callable[[float], FDivergenceEstimator],
    data_module: PandasDirDataModule,
    num_iter: int,
    tag: str,
    min_sig: float = 3.,
    resume: bool = False,
    initial_lr: float = 1e-2,
    lr_decay_factor: float = 0.2,
    output_weight_col: str = 'normalised_weight',
    calculate_divergence_kwargs: Dict = None,
) -> ReweightLoopResult:
    """
    Runs a reweighting loop to estimate the divergence between two distributions. In each iteration, the divergence
    is estimated using calculate_divergence. If the obtained significance of the divergence is greater than min_sig,
    the learned likelihood ratio is used to reweight the distributions to remove the learnt features. The reweighted
    distribution is then used as the input for the next iteration, freeing up the networks to find other smaller
    features in the data. In each iteration, the initial learning rate of the model is reduced by a

    Parameters
    ----------
    module_factory
        A callable that accepts an initial learning rate and returns a FDivergenceEstimator instance initialised with
        the given learning rate
    data_module
        An instance of PandasDataDirModule
    num_iter
        The number of reweighting iterations to perform
    tag
        A short tag describing the model or any other details
    min_sig
        The minimum significance required to accept that the model has learnt a significant feature in the data which
        should be reweighted away before the next iteration
    resume
        Whether training should be resumed
    initial_lr
        The initial learning rate
    lr_decay_factor
        The factor by which the learning rate should be multiplied by between iteration
    output_weight_col
        The output weight column name. Defaults to data_module.weight_col
    calculate_divergence_kwargs
        Any other parameters to pass into calculate_divergence

    Returns
    -------
    ReweightLoopResult
        ReweightLoopResult object containing the result of each iteration, the name of the p_over_q_{i} columns added,
        the final divergence accumulator and the final reweighted data module
    """
    reweighted_path = data_module.dataset_dir.parent / f"{data_module.dataset_dir.name}_{tag}_reweighted"
    current_datamodule = data_module
    if resume:
        current_datamodule = data_module.copy(dataset_dir=reweighted_path)
    elif reweighted_path.exists():
        raise Exception(f"{reweighted_path} already exists. Please manually delete, or configure training to resume, and try again.")

    calculate_divergence_kwargs.setdefault('trainer_kwargs', {})
    calculate_divergence_kwargs['trainer_kwargs'].setdefault("max_epochs", -1)

    current_lr = initial_lr
    p_over_q_cols = []
    results = []
    for i in range(0, num_iter):
        module = module_factory(current_lr)

        try:
            result = calculate_divergence(
                module=module,
                data_module=current_datamodule,
                log_dir=data_module.dataset_dir,
                name=f"{tag}_reweighted",
                **(calculate_divergence_kwargs or {}),
            )
        except KeyboardInterrupt:
            break

        if result.sig > min_sig:
            num_p_over_q = sum('_reweighted' in tag for tag in current_datamodule.tags)
            p_over_q_col = f'p_over_q_{num_p_over_q}'
            p_over_q_cols.append(p_over_q_col)

            current_datamodule = current_datamodule.transform(
                lambda df: add_p_over_q_transformation(df, result.best_module, current_datamodule, p_over_q_col),
                out_dir=reweighted_path,
                desc="Adding p_over_q",
                force=True,
            ).reweight(
                f"{tag}_reweighted",
                lambda df: reweight_down_from_p_over_q(
                    df,
                    data_module.target_cols[0],
                    p_over_q_col
                ),
                reweighted_path,
                force=True,
                output_weight_col=output_weight_col or current_datamodule.weight_col,
            )

        results.append(result)
        current_lr *= lr_decay_factor

    final_data_module = current_datamodule.copy(
        weight_col=data_module.weight_col,
        target_cols=current_datamodule.target_cols + p_over_q_cols
    )

    return ReweightLoopResult(
        calculate_divergence_results=results,
        p_over_q_cols=p_over_q_cols,
        final_divergence_accumulator=calculate_total_divergence(final_data_module, module.divergence),
        final_data_module=final_data_module,
    )
