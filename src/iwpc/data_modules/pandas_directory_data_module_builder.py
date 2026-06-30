import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Iterable

from pandas import DataFrame

from .dataframe_serializer import DataFrameSerializer, resolve_serializer
from .pandas_directory_data_module import PandasDirDataModule
from iwpc.types import PathLike
from iwpc.utils import dump_yaml


class PandasDirDataModuleBuilder:
    """
    Utility for building PandasDirDataModule's. Handles the common jobs like recording the file sizes, writing the
    ds_info and shuffling the dataset. Use as follows:

    with PandasDirDatamoduleBuilder("<some_dir>", file_size=5000000) as builder:
        for df in some_iter:
            builder.write(df)
    """
    def __init__(
        self,
        dataset_dir: PathLike,
        force: bool = False,
        file_size: Optional[int] = None,
        shuffle: bool = True,
        tags: Optional[Union[str, Iterable[str]]] = None,
        serializer: "DataFrameSerializer | str | None" = None,
    ):
        """
        Parameters
        ----------
        dataset_dir
            The directory into which the data should be written
        force
            Whether to overwrite existing files in the dataset_dir. Will raise an exception if the dataset already
            exists and force=False
        file_size
            Optional file size used to rebatch the final dataset
        shuffle
            Whether to shuffle the final dataset
        tags
            Any tags to add to the dataset's metadata. A creation time tag is automatically added
        serializer
            Selects the on-disk (de)serialization format. May be a DataFrameSerializer instance, the name of a built-in
            serializer (``"pickle"`` or ``"parquet"``), or None for the pickle default. The chosen format is recorded in
            ds_info.yml so the resulting dataset can be re-opened without re-specifying it
        """
        self.dataset_dir = Path(dataset_dir)
        self.force = force
        self.file_size = file_size
        self.shuffle = shuffle
        self.serializer = resolve_serializer(serializer)
        if tags is None:
            self.tags = []
        else:
            self.tags = [tags] if isinstance(tags, str) else list(tags)
        self.tags = [f"Created: {datetime.now().isoformat()}"] + self.tags

    def __enter__(self) -> "PandasDirDatamoduleBuilder":
        """
        Prepares the output directory and resets the file_sizes list

        Returns
        -------
        PandasDirDatamoduleBuilder
            This instance
        """
        if self.dataset_dir.exists():
            if self.force:
                if any(f.is_dir() for f in self.dataset_dir.glob('*')):
                    raise Exception("Refusing to overwrite a directory that contains sub-directories")
                shutil.rmtree(self.dataset_dir)
            else:
                raise FileExistsError(f'Directory {self.dataset_dir} already exists. Set `force=True` to overwrite.')

        self.dataset_dir.mkdir()
        self.file_sizes = []
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Writes the ds_info.yml file, rebatches, and shuffles the resulting dataset if specified
        """
        ds_info = {'file_sizes': self.file_sizes, 'serializer': self.serializer.name}
        if self.tags is not None:
            ds_info['tags'] = self.tags
        dump_yaml(ds_info, self.dataset_dir / 'ds_info.yml')
        dm = PandasDirDataModule(self.dataset_dir, serializer=self.serializer)

        if self.file_size is not None:
            dm.rebatch_files(self.file_size)
        if self.shuffle:
            dm.shuffle()

    def write(self, df: DataFrame) -> None:
        """
        Writes the DataFrame to the dataset_dir and records the file size

        Parameters
        ----------
        df
            The DataFrame to write
        """
        self.serializer.write(df, self.dataset_dir / f'file_{len(self.file_sizes)}{self.serializer.extension}')
        self.file_sizes.append(df.shape[0])
