from typing import List, Optional, Union, Tuple

import numpy as np
from pandas import DataFrame
from torch.utils.data import Dataset


class PandasDataset(Dataset):
    """
    Dataset implementation that returns data from rows of a pandas DataFrame
    """
    def __init__(
        self,
        df: DataFrame,
        feature_cols: List[str],
        target_cols: Optional[Union[str, List[str]]] = None,
        weight_col: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        df
            A Pandas DataFrame containing all columns specified in feature_cols, target_cols, and weight_col
        feature_cols
            A list of the names of the feature columns to provide when iterated over
        target_cols
            Optional. A list of names of columns to provide as targets when iterated over
        weight_col
            Optional. The name of a weight column to provide when iterated over. All weights are set to 1 if not
            specified
        """
        self.feature_cols = feature_cols
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.weight_col = weight_col

        self.df = df
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[target_cols].values.astype(np.float32) if target_cols else np.zeros((self.df.shape[0], 0), dtype=np.float32)
        self.weights = self.df[weight_col].values.astype(np.float32) if weight_col else np.ones(self.df.shape[0], dtype=np.float32)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of samples in the dataframe
        """
        return self.df.shape[0]

    def __getitem__(self, idx: int) -> Union[np.ndarray, Tuple]:
        """
        Returns the sample data, targets and weight for the requested idx

        Parameters
        ----------
        idx

        Returns
        -------
        Union[np.ndarray, Tuple]
            The requested sample at the given idx. If self.target_cols is None, an array of shape (1, 0) is returned as
            the target. If self.weight_col is None, 1. is returned as the weight
        """
        return self.features[idx], self.targets[idx], self.weights[idx]

    @property
    def num_features(self) -> int:
        """
        Returns
        -------
        int
            The number of features in the data
        """
        return len(self.feature_cols)
