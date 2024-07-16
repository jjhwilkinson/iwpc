import datetime
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict

import lightning as L
from lightning import LightningDataModule
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from .modules.fdivergence_base import FDivergenceEstimator
from .types import PathLike


@dataclass
class DivergenceResult:
    """
    Contains information about the result of a divergence calculation including the trained model and the resulting
    divergence value, uncertainty, and more
    """
    divergence: float
    divergence_stderr: float
    data_module: Optional[LightningDataModule] = None
    best_module: Optional[FDivergenceEstimator] = None
    best_model_checkpoint_path: Optional[Path] = None
    trainer: Optional[L.Trainer] = None

    @property
    def sig(self) -> float:
        """
        Returns
        -------
        Significance of the divergence calculated
        """
        return self.divergence / self.divergence_stderr


def calculate_divergence(
    module: FDivergenceEstimator,
    data_module: LightningDataModule,
    patience: int = 20,
    resume_training_from: Path = None,
    log_dir: PathLike = Path(os.getcwd()),
    name: str = None,
    trainer_kwargs: Optional[Dict] = None,
) -> DivergenceResult:
    """
    Estimates a lower bound of an f-divergence between two distributions. Typically, the FDivergenceEstimator instance
    expects data_module to provide batches comprised of a tuple containing (data_features, target_features, weights)
    with a single target feature that is 0 for one distribution and 1 for the other. Training is performed using an
    instance of Trainer from the lightning package. This includes many useful parameters such as min_epochs and
    max_epochs, which may be configured using trainer_kwargs. An EarlyStopping callback decides when training ends with
    a configurable patience. You can check progress using `tensorboard --logdir {log_dir}` and terminate the training
    early using a keyboard interrupt. See https://arxiv.org/abs/2405.06397 for more details

    Parameters
    ----------
    module
        A FDivergenceEstimator instance
    data_module
        A datamodule compatible with the provided module
    patience
        How many epochs the validation divergence may not increase before training is terminated
    resume_training_from
        A path to a checkpoint to resume interrupted training
    log_dir
        The directory into which logs should be saved
    name
        A name for the run *highly* recommended. The date and time of the run is automatically appended to the name
    trainer_kwargs
        Any additional arguments to pass to the Trainer constructor. min_epochs and max_epochs are common choices

    Returns
    -------
    DivergenceResult
    """
    trainer_kwargs = trainer_kwargs or {}
    datetime_str = datetime.datetime.now().strftime('%Y%m%d%H%M')
    name = f"{name}-{datetime_str}" if name else datetime_str

    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor="val_Df",
        mode="max"
    )
    tb_logger = TensorBoardLogger(save_dir=log_dir, version=name)
    trainer = L.Trainer(
        callbacks=[
            checkpoint_callback,
            EarlyStopping(monitor="val_Df", mode="max", patience=patience),
            LearningRateMonitor(logging_interval='epoch'),
        ],
        default_root_dir=log_dir,
        logger=tb_logger,
        **trainer_kwargs
    )
    trainer.fit(model=module, datamodule=data_module, ckpt_path=resume_training_from)
    best_module = type(module).load_from_checkpoint(checkpoint_callback.best_model_path)

    results = trainer.validate(
        model=best_module,
        datamodule=data_module,
        verbose=True,
    )[0]

    return DivergenceResult(
        data_module=data_module,
        best_module=best_module,
        best_model_checkpoint_path=Path(checkpoint_callback.best_model_path),
        trainer=trainer,
        divergence=results['val_Df'],
        divergence_stderr=results['val_Df_err'],
    )
