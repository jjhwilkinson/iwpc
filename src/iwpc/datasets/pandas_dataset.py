import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset


type StructuredData = list[Tensor | list["StructuredData"]]
type StructuredDataSpec = list[str | list["StructuredDataSpec"]]


def structure_data(df: DataFrame, feature_spec: StructuredDataSpec) -> StructuredData | Tensor:
    """
    Converts a Pandas DataFrame into an instance of StructuredData using the specified StructuredDataSpec

    Parameters
    ----------
    df
        A pandas DataFrame containing all columns specified in feature_spec
    feature_spec
        A StructuredDataSpec instance

    Returns
    -------
    StructuredData
        The data contained in df structured into a StructuredData instance
    """
    if isinstance(feature_spec, str) or all(isinstance(entry, str) for entry in feature_spec):
        return torch.as_tensor(df[feature_spec].values.astype(float).copy(), dtype=torch.float32)
    return [structure_data(df, entry) for entry in feature_spec]


def recursive_slice_structured_data(
    structured_data: StructuredData,
    idx: slice | int,
) -> Tensor | tuple:
    """
    Recursively indexes a StructuredData instance with the given slice object

    Parameters
    ----------
    structured_data
        An instance of StructuredData
    idx
        A slice object or an integer index

    Returns
    -------
    Tensor | tuple
    """
    if isinstance(structured_data, Tensor):
        return structured_data[idx]
    return tuple(recursive_slice_structured_data(entry, idx) for entry in structured_data)


class PandasDataset(Dataset):
    """
    Dataset implementation that returns data from rows of a pandas DataFrame in a structured format
    """
    def __init__(
        self,
        df: DataFrame,
        feature_spec: StructuredDataSpec,
        weight_col: str | None = None,
    ):
        """
        Parameters
        ----------
        df
            A Pandas DataFrame containing all columns specified in feature_cols, target_cols, and weight_col
        feature_spec
            A recursive specification of the columns to serve and the shape in which to provide them referred to as a
            StructuredDataSpec. Each entry must be either a string or another StructuredData instance. For example,
            ['col1', 'col2', 'col3'] or [['col1', 'col2'], 'col3']. The first version would result in drawn samples that
            are a tuple of a single Tensor of shape (3,). The second version would result in drawn samples that are a
            tuple containing two tensors of shape (2,) and (1,) respectively
        weight_col
            Optional. The name of a weight column to provide when iterated over. All weights are set to 1 if not
            specified
        """
        self.feature_spec = feature_spec
        self.weight_col = weight_col
        self.num_rows = df.shape[0]

        self.weights = (
            torch.as_tensor(df[weight_col].values, dtype=torch.float32) if weight_col
            else torch.ones(self.num_rows, dtype=torch.float32)
        )
        self.structured_data: StructuredData = structure_data(df, feature_spec)
        self.structured_data.append(self.weights)

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            The number of samples in the dataset
        """
        return self.num_rows

    def __getitem__(self, idx: int) -> tuple:
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
        return recursive_slice_structured_data(self.structured_data, idx)

    @property
    def num_features(self) -> int:
        """
        Returns
        -------
        int
            The number of features in the data
        """
        return len(self.feature_cols)


if __name__ == '__main__':
    ds = PandasDataset(
        pd.read_pickle("/Users/jeremywilkinson/research_data/MPA/MPA_SingleTruthProbe_v1/file_0.pkl"),
        [['probe_theta', 'probe_phi'], ['probe_matched_IDTracks', 'probe_matched_MSTracks'], 'matchobj_id_phi'],
    )
    print(ds[[0, 1, 2]])
