import copy
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional, Union, List, Callable, Tuple

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from pandas import DataFrame
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric
from tqdm import tqdm

from ..datasets.pandas_file_list_dataset import PandasFileListDataset
from ..types import PathLike, TensorOrNDArray
from ..utils import read_yaml, temp_directory, dump_yaml

logger = logging.getLogger(__name__)


class PandasDirDataModule(LightningDataModule):
    """
    A generic LightningDataModule implementation that accepts a directory of pickled pandas dataframes. The directory
    structure must contain a number of pickle files named 'file_{i}.pkl' numbered from 0 through N-1 and a metadata
    file named ds_info.yml which must contain at least a 'file_sizes' entry providing an ordered list of the number of
    samples in each file. E.g.

    file_sizes:
        - 10669194
        - 10669194

    The size of each file should be chosen such that the number of dataloader workers (train and validation) times the
    total in-memory size of the dataframe fits comfortably into memory on the device used for training.

    The total list of files is split up into a list of training and validation files according to the 'split' parameter.
    The first ceil(N * split) files are allocated for training and the remaining files are used for validation.
    As such one must be careful that the files partitioned in this way results in a sensible and unbiased
    train-validation split.

    The ds_info dictionary may also contain a lot of additional information useful for tracking the history of a dataset
    and the manipulations applied. As such the state of the dictionary may be important. It is recommended that all
    modifications to the dataset are performed through the transform method defined below and that a tag is provided
    with each modification. The full list of tags recording the state of the data is available through 'self.tags'.
    """
    def __init__(
        self,
        dataset_dir: PathLike,
        feature_cols: List[str],
        target_cols: Optional[Union[str, List[str]]] = None,
        weight_col: Optional[str] = None,
        split: float = 0.5,
        batch_size: int = 2**15,
        limit_files: Optional[int] = None,
        num_workers: int = 0,
    ):
        """
        Parameters
        ----------
        dataset_dir
            A path to a dataset directory structured as described in the class docstring
        feature_cols
            A list of the names of feature columns to provide when iterated over
        target_cols
            Optional. A list of names of columns to provide as targets when iterated over
        weight_col
            Optional. The name of a weight column to provide when iterated over
        split
            The train-validation split. The first ceil(N * split) files are allocated for training and the remaining
            files are used for validation
        batch_size
            The batch size to passed DataLoader instances
        limit_files
            Limit the number of files used to allow rapid testing
        num_workers
            The number of workers to provide the DataLoader instances. os.cpu_count() is recommended
        """
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.feature_cols = feature_cols
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.weight_col = weight_col
        self.split = split
        self.batch_size = batch_size
        self.limit_files = limit_files
        self.num_workers = num_workers

        self.ds_info = read_yaml(self.dataset_dir / 'ds_info.yml')
        self.all_files = [self.dataset_dir / f"file_{i}.pkl" for i in range(len(self.ds_info['file_sizes']))]
        if limit_files:
            self.all_files = self.all_files[:limit_files]
        num_train_files = int(np.ceil(len(self.all_files) * split))
        self.train_files = self.all_files[:num_train_files]
        self.validation_files = self.all_files[num_train_files:]


    @property
    def train_ds(self) -> PandasFileListDataset:
        """
        Constructs a PandasFileListDataset for the training files
        """
        return PandasFileListDataset(
            self.train_files,
            self.feature_cols,
            self.target_cols,
            self.weight_col,
            file_sizes=self.file_sizes[:len(self.train_files)],
            shuffle_in_file=True,
        )

    @property
    def val_ds(self) -> PandasFileListDataset:
        """
        Constructs a PandasFileListDataset for the validation files
        """
        return PandasFileListDataset(
            self.validation_files,
            self.feature_cols,
            self.target_cols,
            self.weight_col,
            file_sizes=self.file_sizes[len(self.train_files):],
            shuffle_in_file=False,
        )

    @property
    def ndim(self) -> int:
        """
        The number of dimensions/features in the data
        """
        return len(self.feature_cols)

    @property
    def file_sizes(self) -> List[int]:
        """
        List of the number of samples in each file
        """
        if self.limit_files:
            return self.ds_info['file_sizes'][:self.limit_files]
        return self.ds_info['file_sizes']

    @property
    def num_files(self) -> int:
        """
        Total number of train and validation files
        """
        return len(self.all_files)

    def train_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader which iterates over the samples in the training files
        """
        if self.num_workers == 1:
            raise Exception("Fix num_workers == 1, you wanted to set to os.cpu_count()")

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a DataLoader which iterates over the samples in the validation files
        """
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False
        )

    @property
    def tags(self) -> List[str]:
        """
        Returns
        -------
        List[str]
            The list of tags associated with the dataset
        """
        return self.ds_info.get('tags', [])

    def file_iter(
        self,
        include_train_files: bool = True,
        include_validation_files: bool = True,
    ) -> Tuple[Path, DataFrame]:
        """
        Yields the path and presiding dataframe for each file

        Parameters
        ----------
        include_train_files
        include_validation_files

        Yields
        ------
        Tuple[Path, DataFrame]
        """
        assert self.limit_files is None

        files = []
        if include_train_files:
            files += self.train_files
        if include_validation_files:
            files += self.validation_files

        for file in files:
            yield file, pd.read_pickle(file)

    def transform(
        self,
        transformation: Callable[[DataFrame], DataFrame],
        out_dir: Optional[PathLike],
        new_ds_info: dict = None,
        update_ds_info: dict = None,
        desc: str = 'Applying transformation',
        force: bool = False,
        tag: Union[str, List[str]] = None,
    ) -> "PandasDirDataModule":
        """
        Centralised function for manipulating the datasets. Ensures the state of ds_info is consistent. All
        modifications to a dataset should be performed through this function. Since all work is first completed in a
        temporary directory before being moved across to out_dir, data corruptions are unlikely if crashes occur

        Parameters
        ----------
        transformation
            A function that takes a dataframe and returns another dataframe with the desired modification
        out_dir
            The directory into which the new dataset should be saved. Can be None if the current dataset should be
            overwritten (must set force=True too)
        new_ds_info
            Manual override of the ds_info dictionary. The new file sizes and all tags will be inserted into this
            dictionary
        update_ds_info
            Information to update in the new ds_info dictionary. Information in the existing self.ds_info not
            overwritten in update_ds_info will be inherited and tags cannot be overwritten
        desc
            Description of the transformation for the loading bar
        force
            Overwrite any existing dataset with the path given in out_dir
        tag
            A tag (or list of tags) to add to the list of dataset tags to identify this transformation

        Returns
        -------
        PandasDirDataModule
            The new data module with the transformation applied
        """
        if out_dir is None:
            out_dir = self.dataset_dir
        out_dir = Path(out_dir)
        if out_dir.exists() and not force:
            raise Exception(f"{out_dir} already exists. Use `force' to overwrite.")

        with temp_directory(out_dir.parent) as tmpdir:
            new_file_sizes = []
            for file, df in tqdm(self.file_iter(), desc=desc, total=self.num_files):
                new_df = transformation(df.copy())
                new_file_sizes.append(new_df.shape[0])
                pd.to_pickle(new_df, tmpdir / file.name)

            if new_ds_info is None:
                new_ds_info = copy.deepcopy(self.ds_info)
            if update_ds_info:
                new_ds_info.update(update_ds_info)
            new_ds_info['file_sizes'] = new_file_sizes

            if tag is not None:
                if isinstance(tag, str):
                    tag = [tag]
                new_ds_info['tags'] = new_ds_info.get('tags', []) + tag
            dump_yaml(new_ds_info, tmpdir / 'ds_info.yml')

            if out_dir.exists():
                for path in out_dir.glob('*.pkl'):
                    path.unlink()
                (out_dir / "ds_info.yml").unlink()
            else:
                out_dir.mkdir()

            for file in tmpdir.glob("*.pkl"):
                shutil.move(file, out_dir / file.name)
            shutil.move(tmpdir / "ds_info.yml", out_dir / "ds_info.yml")

        return PandasDirDataModule(
            out_dir,
            self.feature_cols,
            self.target_cols,
            self.weight_col,
            split=self.split,
            batch_size=self.batch_size,
        )

    def reweight(
        self,
        tag: str,
        reweight_fn: Callable[[DataFrame], TensorOrNDArray],
        out_dir: PathLike,
        force: bool = False,
        label_col: Optional[str] = None,
        output_weight_col: Optional[str] = None,
    ) -> "PandasDirDataModule":
        """
        Reweight the samples in the dataset according to the values given by reweight_fn

        Parameters
        ----------
        tag
            A short tag describing the reweighting
        reweight_fn
            A function that takes in a dataframe and returns an array of reweighting values such that the new weight for
            each sample is given by the old weight times the reweight value
        out_dir
            The directory into which the reweighted dataset should be written
        force
            Whether to overwrite any existing dataset that exists with the path out_dir
        label_col
            An optional column name for class labels. Weights are normalised within each class. If not provided, the
            average weight is normalised to 1.0 across the whole dataset
        output_weight_col
            The name of the column into which the new weight should be written. Defaults to self.weight_col if not
            provided. If self.weight_col is None, output_weight_col must be specified

        Returns
        -------
        PandasDirDataModule
            The reweighted dataset
        """
        weight_col = self.weight_col or output_weight_col
        output_weight_col = output_weight_col or self.weight_col
        if output_weight_col is None:
            raise ValueError(
                "If the dataset being reweighted does not already have a weight column, "
                "you must specify output_weight_col"
            )

        def reweight_wrapper(df):
            if output_weight_col not in df:
                logger.debug("Dataset seems unweighted. inserting 1.0 weights")
                df[output_weight_col] = 1.0
            df[output_weight_col] = df[weight_col] * reweight_fn(df)
            return df

        self.transform(
            reweight_wrapper,
            out_dir,
            force=force,
            tag=tag,
        )
        new_dm = PandasDirDataModule(
            dataset_dir=out_dir,
            feature_cols=self.feature_cols,
            target_cols=self.target_cols,
            weight_col=output_weight_col,
            split=self.split,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
        new_dm.normalise_weights(label_col)
        return self.copy(dataset_dir=out_dir, weight_col=output_weight_col)

    def normalise_weights(self, label_col: Optional[str]) -> None:
        """
        Normalises the weights in the dataset so the average weights are normalised to 1.0. If label_col is provided,
        the average weight is normalised to 1.0 within each class

        Parameters
        ----------
        label_col
            Optional name of the column which labels the class a given sample is from
        """
        assert self.weight_col is not None

        weight_metrics = defaultdict(MeanMetric)
        for file, df in tqdm(self.file_iter(), desc="Calculating weight sums", total=self.num_files):
            if label_col is None:
                weight_metrics[0].update(df[self.weight_col].values)
            else:
                for label in np.unique(df[label_col]):
                    weight_metrics[label].update(df[self.weight_col][df[label_col].values == label].values)

        def normalise_wrapper(df):
            weights = df[self.weight_col].values

            if label_col is None:
                weights /= weight_metrics[0].compute()
            else:
                for label, metric in weight_metrics.items():
                    weights[df[label_col].values == label] /= metric.compute()

            df[self.weight_col] = weights
            return df

        self.transform(
            normalise_wrapper,
            self.dataset_dir,
            desc="Normalising weights",
            force=True,
        )

    def copy(self, **overrides) -> "PandasDirDataModule":
        """
        Creates a copy of the data module with the provided changes

        Parameters
        ----------
        overrides
            Any options to override in the copy. See the PandasDirDataModule constructor

        Returns
        -------
        PandasDirDataModule
            A copy of the dataset with the provided changes applied
        """
        arguments = {
            'dataset_dir': self.dataset_dir,
            'feature_cols': self.feature_cols,
            'target_cols': self.target_cols,
            'weight_col': self.weight_col,
            'split': self.split,
            'batch_size': self.batch_size,
            'limit_files': self.limit_files,
        }
        arguments.update(overrides)
        return PandasDirDataModule(**arguments)
