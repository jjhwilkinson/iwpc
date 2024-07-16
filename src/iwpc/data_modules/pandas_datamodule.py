from typing import List, Optional, Union

import numpy as np
import pandas as pd
import pylab as pl
from lightning import LightningDataModule
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from iwpc.datasets.pandas_dataset import PandasDataset


class PandasDataModule(LightningDataModule):
    """
    Datamodule that wraps a Pandas DataFrame, provides a train-validation split and defines dataloaders which provide
    batches containing the data in the specified columns
    """
    def __init__(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_cols: Optional[Union[str, List[str]]] = None,
        weight_col: Optional[str] = None,
        validation_split: Optional[float] = 0.5,
        dataloader_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        df
            A Pandas DataFrame
        feature_cols
            A list of column names to be provided as features in batches
        target_cols
            A list of column names to be provided as targets in batches
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        super().__init__()
        self.all_data_ds = PandasDataset(
            df,
            feature_cols=feature_cols,
            target_cols=target_cols,
            weight_col=weight_col,
        )
        self.feature_cols = feature_cols
        self.target_cols = target_cols
        self.weight_col = weight_col
        self.dataloader_kwargs = dataloader_kwargs or {}
        self.validation_split = validation_split
        self.train_ds, self.val_ds = train_test_split(self.all_data_ds, test_size=self.validation_split)

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

    @property
    def num_features(self) -> int:
        """
        Returns
        -------
        int
            The number of input features in the data
        """
        return self.all_data_ds.num_features


class BinaryPandasDataModule(PandasDataModule):
    """
    A DataModule which wraps a pair of DataFrames containing the features associated with samples from two different
    classes
    """
    def __init__(
        self,
        df1: DataFrame,
        df2: DataFrame,
        feature_cols: List[str] = None,
        weight_col: Optional[str] = None,
        validation_split: Optional[float] = 0.5,
        dataloader_kwargs: Optional[dict] = None,
    ):
        """
        Parameters
        ----------
        df1
            A DataFrame containing features from one class
        df2
            A DataFrame containing features from a second class must have the same columns as df1
        feature_cols
            A list of the features in the dataframes which should be provided by the various DataLoaders
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        self.df1 = df1
        self.df2 = df2
        all_data_df = pd.concat([df1, df2], ignore_index=True)
        all_data_df['__label'] = np.concat([np.zeros(self.df1.shape[0]), np.ones(self.df1.shape[0])])

        super().__init__(
            all_data_df,
            feature_cols=feature_cols,
            target_cols='__label',
            weight_col=weight_col,
            validation_split=validation_split,
            dataloader_kwargs=dataloader_kwargs
        )
