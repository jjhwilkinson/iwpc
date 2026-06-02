from typing import Optional

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
            A StructuredDataSpec describing the columns to load. See PandasDataset docstring for more details. A
            previous iteration of PandasDataModule allowed the user to specify a list of feature columns and target
            columns. The new equivalent specification for the same result is feature_spec=[feature_cols, target_cols]
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use. Must be between 0 and 1 and represents the fraction of data used for
            training. Defaults to 0.5
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        super().__init__()
        self.all_data_ds = PandasDataset(
            df,
            feature_spec=feature_spec,
            weight_col=weight_col,
        )
        self.feature_spec = feature_spec
        self.weight_col = weight_col
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

    @property
    def num_features(self) -> int:
        """
        Returns the number of feature columns, taken from the first entry of ``feature_spec``. Assumes the canonical
        ``[feature_cols, target_cols]`` nesting convention; if ``feature_spec`` is a flat list of column names, the
        length of that list is returned instead.

        Returns
        -------
        int
            The number of input features in the data
        """
        first = self.feature_spec[0]
        if isinstance(first, str):
            return len(self.feature_spec)
        return len(first)


class BinaryPandasDataModule(PandasDataModule):
    """
    A DataModule which wraps a pair of DataFrames containing the features associated with samples from two different
    classes. A label column is automatically inserted (0 for p, 1 for q). The name of the label column is taken from
    the first entry of ``feature_spec[1]`` (i.e. the target nest), matching the convention used by
    ``PandasDirDataModule``.
    """
    def __init__(
        self,
        p_df: DataFrame,
        q_df: DataFrame,
        feature_spec: StructuredDataSpec,
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
        feature_spec
            A StructuredDataSpec describing the columns to load. Must be nested as ``[feature_cols, [label_col, ...]]``
            following the convention used by ``PandasDirDataModule``. The first entry of ``feature_spec[1]`` names the
            label column that this data module populates with 0/1. Any further columns in ``feature_spec[1]`` must
            already be present in ``p_df`` and ``q_df``. See PandasDataset docstring for more details
        weight_col
            The name of a column containing sample weights to be provided in batches
        validation_split
            The train-validation split to use. Must be between 0 and 1 and represents the fraction of data used for
            training. Defaults to 0.5
        dataloader_kwargs
            Any other arguments to be provided to DataLoader instances
        """
        if (
            not isinstance(feature_spec, list)
            or len(feature_spec) < 2
            or not isinstance(feature_spec[1], list)
            or len(feature_spec[1]) == 0
            or not isinstance(feature_spec[1][0], str)
        ):
            raise ValueError(
                "BinaryPandasDataModule requires feature_spec of the form [feature_cols, [label_col, ...]]; "
                f"got {feature_spec!r}"
            )

        label_col = feature_spec[1][0]
        self.p_df = p_df
        self.q_df = q_df
        all_data_df = pd.concat([p_df, q_df], ignore_index=True)
        all_data_df[label_col] = np.concatenate(
            [np.zeros(self.p_df.shape[0]), np.ones(self.q_df.shape[0])]
        )

        super().__init__(
            all_data_df,
            feature_spec=feature_spec,
            weight_col=weight_col,
            validation_split=validation_split,
            dataloader_kwargs=dataloader_kwargs
        )
