"""
Pluggable (de)serialization for the on-disk DataFrame shards used by PandasDirDataModule.

A dataset directory stores its DataFrames either as pickles (``file_i.pkl``, the historical
default) or in any other format selected via a DataFrameSerializer. Pickle is convenient but
non-portable: pandas pickles embed numpy's internal module paths, so a DataFrame pickled under
numpy 2 (classes under ``numpy._core``) fails to load under numpy 1.x with
``ModuleNotFoundError: No module named 'numpy._core.numeric'``. Parquet is numpy-version-neutral
and avoids the Python/pandas/numpy version coupling entirely.
"""
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from ..types import PathLike


@dataclass(frozen=True)
class DataFrameSerializer:
    """
    Describes how DataFrame shards are written to and read from disk.

    Attributes
    ----------
    name
        Short identifier persisted into ``ds_info.yml`` (and used as the registry key) so a
        dataset records the format it was written with and round-trips without the caller
        re-specifying it
    extension
        The file extension (including the leading dot) used for shard filenames, e.g. ``".pkl"``
    write_fn
        Callable writing a DataFrame to the given path
    read_fn
        Callable reading a DataFrame back from the given path
    """
    name: str
    extension: str
    write_fn: Callable[[pd.DataFrame, PathLike], None]
    read_fn: Callable[[PathLike], pd.DataFrame]

    def write(self, df: pd.DataFrame, path: PathLike) -> None:
        """
        Writes the DataFrame to the given path using this serializer's format

        Parameters
        ----------
        df
            The DataFrame to write
        path
            The destination path
        """
        self.write_fn(df, path)

    def read(self, path: PathLike) -> pd.DataFrame:
        """
        Reads a DataFrame back from the given path using this serializer's format

        Parameters
        ----------
        path
            The source path

        Returns
        -------
        DataFrame
            The deserialized DataFrame
        """
        return self.read_fn(path)


def _write_pickle(df: pd.DataFrame, path: PathLike) -> None:
    df.to_pickle(path)


def _write_parquet(df: pd.DataFrame, path: PathLike) -> None:
    df.to_parquet(path)


# The write/read callables are module-level (not lambdas) so the serializer — and any
# PandasFileListDataset holding a reference to serializer.read — pickles cleanly when DataLoader
# workers are spawned (num_workers > 0).
PICKLE_SERIALIZER = DataFrameSerializer(
    name="pickle",
    extension=".pkl",
    write_fn=_write_pickle,
    read_fn=pd.read_pickle,
)
PARQUET_SERIALIZER = DataFrameSerializer(
    name="parquet",
    extension=".parquet",
    write_fn=_write_parquet,
    read_fn=pd.read_parquet,
)

#: Built-in serializers keyed by name, selectable from a config by string
SERIALIZERS: dict[str, DataFrameSerializer] = {
    serializer.name: serializer for serializer in (PICKLE_SERIALIZER, PARQUET_SERIALIZER)
}


def resolve_serializer(
    serializer: "DataFrameSerializer | str | None",
    ds_info: dict | None = None,
) -> DataFrameSerializer:
    """
    Resolves the serializer to use, preferring an explicit choice and falling back to the format
    recorded in a dataset's ds_info and finally to pickle (backward compatible).

    Parameters
    ----------
    serializer
        Either a DataFrameSerializer instance, the name of a built-in serializer (a key of
        SERIALIZERS), or None to auto-detect. When None, the format recorded in ds_info is used if
        present, otherwise the pickle serializer is used so existing ``.pkl`` datasets keep working
    ds_info
        Optional ds_info dictionary whose ``serializer`` field (when present) names the format the
        dataset was written with

    Returns
    -------
    DataFrameSerializer
        The resolved serializer
    """
    if isinstance(serializer, DataFrameSerializer):
        return serializer
    if isinstance(serializer, str):
        if serializer not in SERIALIZERS:
            raise ValueError(
                f"Unknown serializer '{serializer}'. Known serializers: {sorted(SERIALIZERS)}. "
                f"Pass a DataFrameSerializer instance for a custom format."
            )
        return SERIALIZERS[serializer]
    if serializer is not None:
        raise TypeError(
            f"serializer must be a DataFrameSerializer, a string name, or None, got {type(serializer)}"
        )

    recorded = (ds_info or {}).get("serializer")
    if recorded is None:
        return PICKLE_SERIALIZER
    return resolve_serializer(recorded)
