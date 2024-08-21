from typing import Optional

from numpy._typing import NDArray


class Scalar:
    """
    Represents a single scalar quantity. Provides bin range and labels required for making a plot in with the quantity
    """
    def __init__(
        self,
        label: str,
        latex_label: Optional[str] = None,
        bins: Optional[NDArray] = None,
    ):
        self.bins = bins
        self.label = label
        self.latex_label = latex_label or label
