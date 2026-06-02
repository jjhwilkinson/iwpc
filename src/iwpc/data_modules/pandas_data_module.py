from typing import List, Optional

import numpy as np
import pandas as pd
from lightning import LightningDataModule
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from iwpc.datasets.pandas_dataset import PandasDataset, StructuredDataSpec


class PandasDataModule(LightningDataModule):
    """
    Datamodule that wraps a Pandas DataFrame, provides a train-validation split and defines dataloaders which provide
    batches containing the data in the columns referenced by ``feature_spec``
    """
    def __init__(
        self,
        df: DataFrame,
        feature_spec: StructuredDataSpec,
        weight_col: Optional[str] = None,
        validation_split: Optional[float] = 0.5,
        dataloader_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        df
            A Pandas DataFrame
        feature_spec
            A StructuredDataSpec describing the columns to load. See PandasDataset docstring for more details. The
            canonical nesting convention shared with ``PandasDirDataModule`` is ``[feature_cols, [target_col, ...]]``;
            a flat list of column names is also accepted when no target tensor is required
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use. Must be between 0 and 1 and represents the fraction of data used for
            training. Defaults to 0.5
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        super().__init__()
        self.feature_spec = feature_spec
        self.weight_col = weight_col

        self.all_data_ds = PandasDataset(
            df,
            feature_spec=feature_spec,
            weight_col=weight_col,
        )
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.validation_split = validation_split
        self.train_ds, self.val_ds = train_test_split(
            self.all_data_ds,
            train_size=self.validation_split,
            shuffle=True,
        )

    def train_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader
            A DataLoader instance initialised with the train portion of the original DataFrame
        """
        return DataLoader(
            self.train_ds,
            **self.dataloader_kwargs
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns
        -------
        DataLoader
            A DataLoader instance initialised with the validation portion of the original DataFrame
        """
        return DataLoader(
            self.val_ds,
            **self.dataloader_kwargs,
        )


class BinaryPandasDataModule(PandasDataModule):
    """
    A DataModule which wraps a pair of DataFrames containing the features associated with samples from two different
    classes. A ``__label`` column is automatically inserted (0 for ``p_df``, 1 for ``q_df``) and the corresponding
    ``feature_spec=[feature_cols, ['__label']]`` is constructed internally before delegating to ``PandasDataModule``.
    """
    def __init__(
        self,
        p_df: DataFrame,
        q_df: DataFrame,
        feature_cols: List[str],
        weight_col: Optional[str] = None,
        validation_split: Optional[float] = 0.5,
        dataloader_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        p_df
            A DataFrame containing features from one class (label 0)
        q_df
            A DataFrame containing features from a second class (label 1). Must have the same columns as p_df
        feature_cols
            A list of the features in the dataframes which should be provided by the various DataLoaders
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use. Must be between 0 and 1 and represents the fraction of data used for
            training. Defaults to 0.5
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        self.p_df = p_df
        self.q_df = q_df
        self.feature_cols = list(feature_cols)

        all_data_df = pd.concat([p_df, q_df], ignore_index=True)
        all_data_df['__label'] = np.concatenate(
            [np.zeros(self.p_df.shape[0]), np.ones(self.q_df.shape[0])]
        )

        super().__init__(
            all_data_df,
            feature_spec=[self.feature_cols, ['__label']],
            weight_col=weight_col,
            validation_split=validation_split,
            dataloader_kwargs=dataloader_kwargs,
        )

    @property
    def num_features(self) -> int:
        """
        Returns
        -------
        int
            The number of input feature columns
        """
        return len(self.feature_cols)
