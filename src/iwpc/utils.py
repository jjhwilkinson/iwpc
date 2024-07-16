import pickle
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Tuple, Optional

import numpy as np
import torch
import yaml
from numpy._typing import NDArray

from .types import TensorOrNDArray, PathLike


def split_by_mask(
    mask: TensorOrNDArray,
    *arrs: TensorOrNDArray
) -> Tuple[Iterable[TensorOrNDArray], Iterable[TensorOrNDArray]]:
    """
    Splits each array in arrs into two arrays, the first containing the values for which mask is 'True' and the second
    containing the values for which mask is 'False'

    Parameters
    ----------
    mask
        An array of bool values. Can be numpy, or pytorch, etc
    arrs
        A list of arrays to split. Must each have the same length as the mask array


    Returns
    -------
    Tuple[Iterable[TensorOrNDArray], Iterable[TensorOrNDArray]]
        A pair of tuples, each containing the same number of entries as arrs. The first containing the list values for
        which mask is 'True' and the second containing the list values for which mask is 'False'
    """
    if mask.dtype not in [np.bool, torch.bool]:
        mask = mask.astype(bool)
    return (arr[mask] for arr in arrs), (arr[~mask] for arr in arrs)


def read_yaml(path: PathLike) -> dict:
    """
    Reads a yaml file and returns the corresponding dictionary

    Parameters
    ----------
    path

    Returns
    -------
    dict
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dump_yaml(data: dict, path: PathLike) -> None:
    """
    Writes a yaml file containing the information in dict to the given path

    Parameters
    ----------
    data
    path
    """
    with open(path, 'w') as f:
        yaml.dump(data, f)


def read_pickle(path: PathLike) -> object:
    """
    Reads a pickle file and returns the contents

    Parameters
    ----------
    path

    Returns
    -------
    object
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def dump_pickle(obj: object, pth: PathLike) -> None:
    """
    Writes a pickle file containing the object to the given path

    Parameters
    ----------
    obj
    path
    """
    with open(pth, 'wb') as f:
        pickle.dump(obj, f)


@contextmanager
def temp_directory(dir_: Optional[PathLike] = None):
    """
    Context manager providing a directory for temporary storage. Uses the built in tempfile implementation unless `dir_`
    is provided, in which case a temporary directory is created in `dir_`

    usage:

    with temp_directory() as tmp_dir:
        ...

    Parameters
    ----------
    dir_
        Optional[PathLike]

    Returns
    -------
    Path
        The path to the temporary directory
    """
    if dir_ is None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
        return

    dir_ = Path(dir_)
    tmpdir = dir_ / uuid.uuid4().hex
    tmpdir.mkdir()

    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir)
    return


def bin_centers(bins: NDArray) -> NDArray:
    """
    Parameters
    ----------
    bins
        A numpy array of bin edges

    Returns
    -------
    NDArray
        The center of each bin
    """
    return (bins[1:] + bins[:-1]) / 2


def format_quantity_with_uncertainty(val: float, err: float) -> str:
    """
    Formats the given quantity and uncertainty so that a single digit of uncertainly is shown and the value is rounded
    to the same decimal place. eg 0.123456 with an error of 0.00012345 would be rendered as "1.234E-1 +- 1E-4"

    Parameters
    ----------
    val
        The value of some quantity
    err
        The uncertainty in the quantity

    Returns
    -------
    str
    """
    val_order = int(np.log10(val))
    err_order = int(np.log10(err))
    return f"{val:.{val_order - err_order}E} +- {err:.0E}"
