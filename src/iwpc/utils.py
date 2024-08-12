import hashlib
import logging
import pickle
import shutil
import tempfile
import uuid
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Iterable, Tuple, Optional, List, Callable, Any

import numpy as np
import torch
import yaml
from numpy._typing import NDArray
from torch import Tensor

from .types import TensorOrNDArray, PathLike


logger = logging.getLogger(__name__)


def split_by_mask(
    mask: TensorOrNDArray,
    *arrs: TensorOrNDArray
) -> Tuple[List[TensorOrNDArray], List[TensorOrNDArray]]:
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
    Tuple[List[TensorOrNDArray], List[TensorOrNDArray]]
        A pair of lists, each containing the same number of entries as arrs. The first containing the list values for
        which mask is 'True' and the second containing the list values for which mask is 'False'
    """
    if isinstance(mask, Tensor):
        mask = mask.cpu().detach().numpy()

    mask = mask.astype(bool)
    return [arr[mask] for arr in arrs], [arr[~mask] for arr in arrs]


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


def format_quantity_with_uncertainty(val: float, err: float, with_sig: bool = False) -> str:
    """
    Formats the given quantity and uncertainty so that a single digit of uncertainly is shown and the value is rounded
    to the same decimal place. eg 0.123456 with an error of 0.00012345 would be rendered as "1.2345E-1 +- 1E-4". If
    with_sig is True, this would be rendered as "1.234E-1 +- 1E-4 (1234.5)"

    Parameters
    ----------
    val
        The value of some quantity
    err
        The uncertainty in the quantity
    with_sig
        Whether to append the significance in brackets

    Returns
    -------
    str
    """
    if not np.isfinite(val) or not np.isfinite(err):
        return "NaN"

    val_order = int(np.log10(np.abs(val)))
    err_order = int(np.log10(np.abs(err)))
    string = f"{val:.{abs(val_order - err_order)}E} +- {err:.0E}"
    if with_sig:
        string += f" ({val / err:.1f})"
    return string


def pickle_cache(directory: PathLike) -> Callable[[Callable], Callable[[...], Any]]:
    """
    Handy utility decorator that attempts to cache a functions outputs using pickle files. Arguments are converted to
    strings using repr and hashed. Repeated calls with the same has, even after restarting the script, will result in
    the same output being loaded from the cached pickle file

    Parameters
    ----------
    directory
        The directory into which the cached pickle files should be placed
    """
    directory = Path(directory)

    def pickle_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            args_kwargs_repr = repr(args) + repr(kwargs)
            args_kwargs_hash = int(hashlib.sha256(args_kwargs_repr.encode('utf-8')).hexdigest(), 16) % 10 ** 8
            cache_file = directory / f'{args_kwargs_hash}.pkl'
            if cache_file.exists():
                logger.info("Function call exists in-cache, loading pickle file")
                with open(cache_file, 'rb') as f:
                    ret_val = pickle.load(f)
            else:
                logger.info("Function call does not exist in-cache")
                ret_val = fn(*args, **kwargs)
                with open(cache_file, 'wb') as f:
                    pickle.dump(ret_val, f)
            return ret_val
        return wrapper
    return pickle_decorator
