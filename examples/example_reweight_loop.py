import shutil
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from iwpc.accumulators.binned_Df_accumulator import BinnedDfAccumulator
from iwpc.data_modules.pandas_directory_data_module import PandasDirDataModule
from iwpc.divergences import JensenShannonDivergence
from iwpc.modules.naive import GenericNaiveVariationalFDivergenceEstimator
from iwpc.reweight_loop import run_reweight_loop
from iwpc.scalars.scalar import Scalar


if Path(Path(".") / "sample_dataset_test_reweighted").exists():
    shutil.rmtree(Path(".") / "sample_dataset_test_reweighted")


datamodule = PandasDirDataModule(
    Path(".") / "sample_dataset",
    ['x', 'y'],
    target_cols='label',
)
divergence = JensenShannonDivergence()
module_factory = lambda lr: GenericNaiveVariationalFDivergenceEstimator(
    datamodule.ndim,
    divergence,
    initial_learning_rate=lr,
    lr_patience=10
)

result = run_reweight_loop(
    module_factory,
    datamodule,
    1,
    'test',
    calculate_divergence_kwargs={'trainer_kwargs': {'max_epochs': 10}},
    initial_lr=1e-3,
)

angle_scalar = Scalar(lambda df: df['angles'], np.linspace(-np.pi, np.pi, 50), 'angle', latex_label=r'$\theta$')
radius_scalar = Scalar(lambda df: (df['x']**2 + df['y']**2)**0.5, np.linspace(0.5,1.5, 50), 'r')
final_data_module = PandasDirDataModule(
    Path(".") / "sample_dataset_test_reweighted",
    ['x', 'y'],
    target_cols='label',
)

for scalars in [
    [radius_scalar],
    [angle_scalar],
    [angle_scalar, radius_scalar],
]:
    binned_df_accumulator = BinnedDfAccumulator(
        scalars,
        JensenShannonDivergence(),
    )
    binned_df_accumulator.evaluate(
        final_data_module,
        result.p_over_q_cols,
    )
    axs = binned_df_accumulator.plot()
    print(binned_df_accumulator.marginalised_df_accumulator)
    print(binned_df_accumulator.global_df_accumulator)

plt.show()
