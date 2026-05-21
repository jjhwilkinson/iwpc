from abc import ABC, abstractmethod
from typing import Tuple, Iterator

from torch import Tensor


class CutKernelInterface(ABC):
    """
    Abstract interface for kernels that are based upon an underlying kernel which have had a portion of their sample space
    'cut-out'. Provides the methods required to calculate the log-probability that a sample from the base distribution
    passes the cut while maintaining gradient information
    """
    @abstractmethod
    def cut_pass_log_prob(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            The log-probability that a sample from the base kernel passing the cut for each row of conditioning information
        """

    @abstractmethod
    def cut_fail_log_prob(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tensor
            The log-probability that a sample from the base kernel fails the cut for each row of conditioning information
        """
