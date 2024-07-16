from typing import Callable, Optional

import numpy as np
from numpy._typing import NDArray
from pandas import DataFrame


class Scalar:
    """
    Represents a single scalar value that may be calculated from the features in a given sample in a dataset. Provides
    a function to calculate the value as well as the names, bin range and labels required for making a plot in with the
    quantity. An example might be the magnitude of a vector where the dataset might only provide the three components
    directly
    """
    def __init__(
        self,
        fn: Callable[[DataFrame], NDArray],
        bins: NDArray,
        label: str,
        latex_label: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        fn
            A callable function that accepts a Pandas DataFrame and calculates/returns the represented quantity
        bins
            A sequence of regularly spaced bins used when plotting or histogramming in this variable
        label
            A print friendly name for the quantity
        latex_label
            A version of the label that may use LaTeX for additional formatting. Defaults to the label above
        """
        self.fn = fn
        self.bins = bins
        self.label = label
        self.latex_label = latex_label or label

    def __call__(self, df: DataFrame) -> NDArray:
        """
        Parameters
        ----------
        df
            A DataFrame instance

        Returns
        -------
        NDArray
            The output value of self.fn(df)
        """
        return self.fn(df)
