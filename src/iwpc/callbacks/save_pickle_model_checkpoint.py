from pathlib import Path
from typing import Optional

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning_fabric.utilities import rank_zero_only
from torch.nn.parallel import DistributedDataParallel

from ..types import PathLike


class SavePickleModelCheckpoint(ModelCheckpoint):
    """
    Extension of the ModelCheckpoint class which also saves a pickled version of the best model to a given directory for
    easy access
    """
    def __init__(self, model_save_dir: Optional[PathLike] = None, name: str = 'best_model', **kwargs):
        """
        Parameters
        ----------
        model_save_dir
            Directory into which the best model pickle should be saved. If not specified, the checkpoint will behave
            identically to ModelCheckpoint
        name
            Name of the file to save the best model to without the pkl extension
        kwargs
            Any other arguments passed through to the ModelCheckpoint constructor
        """
        super().__init__(**kwargs)
        if model_save_dir is not None:
            model_save_dir = Path(model_save_dir)
        self.model_save_dir = model_save_dir
        self.name = name

    @property
    def best_model_pickle_path(self) -> Optional[Path]:
        """
        Returns
        -------
        Optional[Path]
            The path of the best model pickle file. Returns None is model_save_dir was not provided
        """
        return None if self.model_save_dir is None else self.model_save_dir / f"{self.name}.pt"

    @rank_zero_only
    def _save_model(self, trainer: Trainer) -> None:
        """
        Saves the trainer's current model version to self.best_model_pickle_path
        """
        if isinstance(trainer.model, DistributedDataParallel):
            model = trainer.model.module.model
        else:
            model = trainer.model.model
        if self.model_save_dir:
            torch.save(model, self.best_model_pickle_path)

    def _save_checkpoint(self, trainer: Trainer, filepath: str) -> None:
        """
        Overrides and calls ModelCheckpoint to save the trainer's current model version to self.best_model_pickle_path
        """
        self._save_model(trainer)
        super()._save_checkpoint(trainer, filepath)
