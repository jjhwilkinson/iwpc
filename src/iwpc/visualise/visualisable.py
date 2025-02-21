from abc import ABC, abstractmethod
from typing import List, Any, Optional

from numpy._typing import NDArray, ArrayLike

from iwpc.scalars.scalar import Scalar
from iwpc.scalars.scalar_function import ScalarFunction


class Visualisable(ABC):
    """
    Abstract class defining the interfaces for objects representing functions that can be visualised using an iwpc
    function visualiser. For example, a custom neural network implementation might define this interface so its outputs
    may be explored using a BokehFunctionVisualiser
    """
    @abstractmethod
    def get_input_scalars(self) -> List[Scalar]:
        """
        Returns
        -------
        List[Scalar]
            Constructs and returns the list of input scalars in the order expected by self.evaluate
        """

    @abstractmethod
    def get_output_scalars(self) -> List[ScalarFunction]:
        """
        Returns
        -------
        List[ScalarFunction]
            Constructs and returns the list of output scalars compatible with the output of self.evaluate_for_visualiser
        """

    @abstractmethod
    def evaluate_for_visualiser(self, x: NDArray) -> Any:
        """
        Evaluates the function on the given array of points

        Parameters
        ----------
        x
            An array of points of shape (N, k) where N is some number of points and k is the number of input scalars

        Returns
        -------
        Any
            The output of the function on the given array of points in the format expected the output scalars
        """

    @property
    def center_point(self) -> Optional[ArrayLike]:
        return None
