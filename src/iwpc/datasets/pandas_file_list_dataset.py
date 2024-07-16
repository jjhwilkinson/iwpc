import logging
from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from .pandas_dataset import PandasDataset
from ..types import PathLike

logger = logging.getLogger(__name__)


class PandasFileListDataset(Dataset):
    """
    Dataset implementation that dynamically loads samples from a list of pickled pandas dataframes. Only one file is
    loaded into memory at a time and subsequent requests for samples from the same file will not trigger any further IO
    requests until a new file is loaded. It is recommended that the individual pandas dataframes are equally sized and
    comfortably fit into memory. When using multiple dataloader workers (you should be!), it is important to ensure that
    num_workers * dataframe size fits into memory. It is also recommended that any wrapping DataLoader is not configured
    to shuffle, as this could result in a significant number of adjacent samples being requested from different files
    which will trigger a lot of slow IO requests loading dataframes into memory. If shuffle_in_file is set to True, then
    samples within a single file will be shuffled each time the file is reloaded. No shuffling across files is currently
    supported
    """
    def __init__(
        self,
        files: List[PathLike],
        feature_cols: List[str],
        target_cols: Optional[Union[str, List[str]]] = None,
        weight_col: Optional[str] = None,
        file_sizes: Optional[List[int]] = None,
        shuffle_in_file: bool = False,
    ):
        """
        Parameters
        ----------
        files
            A list of pickled pandas dataframes
        feature_cols
            A list of the names of the feature columns to provide when iterated over
        target_cols
            Optional. A list of names of columns to provide as targets when iterated over
        weight_col
            Optional. The name of a weight column to provide when iterated over
        file_sizes
            An optional list of the number of samples in each file. If not provided, then each file will be opened to
            check the file size (might be quite slow)
        shuffle_in_file
            If True, samples with a given dataframe will be shuffled each time a file is loaded
        """
        self.files = files
        self.file_sizes = file_sizes
        if self.file_sizes is None:
            self.file_sizes = [
                pd.read_pickle(file).shape[0]
                for file in tqdm(list(files), desc="Calculating file sizes")
            ]
        self.feature_cols = feature_cols
        self.target_cols = [target_cols] if isinstance(target_cols, str) else target_cols
        self.weight_col = weight_col
        self.shuffle_in_file = shuffle_in_file

        self._last_file_no = -1
        self._current_ds = None

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of samples in the dataset across all files
        """
        return np.sum(self.file_sizes)

    def load_file(self, file_idx: int) -> PandasDataset:
        """
        Loads the file with the given idx into various member attributes

        file_idx
            The index of the file within self.files to load
        """
        if file_idx == self._last_file_no:
            return self._current_ds

        logger.debug("Reading file:", self.files[file_idx])
        self._last_file_no = file_idx
        df = pd.read_pickle(self.files[file_idx])
        if self.shuffle_in_file:
            logger.debug("Shuffling dataframe entries")
            df = df.sample(frac=1).reset_index(drop=True)
        self._current_ds = PandasDataset(
            df,
            self.feature_cols,
            self.target_cols,
            self.weight_col,
        )
        return self._current_ds

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns the sample data, targets and weight for the requested idx. idx may be a number in
        [0, sum(self.file_sizes)]

        Parameters
        ----------
        idx

        Returns
        -------
        Tuple
            The requested sample at the given idx. A tuple is returned containing a subset of
            (sample data, targets, weight) in that order. the targets array can be empty (shape (1, 0)) if no target
            columns are provided and the weight is set to 1.0 if no weight column is specified
        """
        file_idx = 0
        in_file_idx = None
        for size in self.file_sizes:
            if idx < size:
                in_file_idx = idx
                break
            idx = idx - size
            file_idx += 1

        return self.load_file(file_idx)[in_file_idx]
