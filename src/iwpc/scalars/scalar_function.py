from typing import Callable, Optional, Any

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame

from .scalar import Scalar


class ScalarFunction(Scalar):
    """
    Represents a single scalar value that may be calculated from some underlying data. Provides a function to calculate
    the value as well as the names, bin range and labels required for making a plot in with the quantity. An example
    might be the magnitude of a vector where the dataset might only provide the vector components
    """
    def __init__(
        self,
        fn: Callable[[Any], NDArray],
        label: str,
        latex_label: Optional[str] = None,
        bins: Optional[NDArray] = None,
    ):
        """
        Parameters
        ----------
        fn
            A callable function that accepts some input data in some form and calculates/returns the represented quantity
        bins
            A sequence of regularly spaced bins used when plotting or histogramming in this variable
        label
            A print friendly name for the quantity
        latex_label
            A version of the label that may use LaTeX for additional formatting. Defaults to the label above
        """
        super().__init__(label, latex_label, bins=bins)
        self.fn = fn

    def __call__(self, *args, **kwargs) -> NDArray:
        """
        Returns
        -------
        NDArray
            The output value of self.fn(*args, **kwargs)
        """
        return self.fn(*args, **kwargs)
