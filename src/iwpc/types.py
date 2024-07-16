from pathlib import Path
from typing import Union, Tuple

from numpy import ndarray
from torch import Tensor


TensorOrNDArray = Union[Tensor, ndarray]
PathLike = Union[str, Path]
Shape = Union[int, Tuple[int, ...]]
