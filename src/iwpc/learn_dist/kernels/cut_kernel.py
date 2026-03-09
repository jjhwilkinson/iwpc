from abc import ABC, abstractmethod
from typing import Tuple, Iterator

from torch import Tensor


class CutKernelInterface(ABC):
    """
    Abstract interface for kernels that are based upon an underlying kernel which have had a portion its sample space
    'cut-out'. Provides the methods required to calculate the log-probability that a sample from the base distribution
    passes the cut while maintaining gradient information
    """
    @abstractmethod
    def draw_with_log_prob_and_cut_pass_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        Tuple[Tensor, Tensor, Tensor]
            1. A sample drawn from the cut-kernel for each row of conditioning information
            2. The log-probability of observing the above sample for the given conditioning information
            3. The log-probability that a sample from the base kernel passes the cut for each row of conditioning information
        """

    @abstractmethod
    def pass_log_prob_and_outcomes_with_log_prob_iter(self, cond: Tensor) -> tuple[Tensor, Iterator[tuple[Tensor, Tensor]]]:
        """
        Parameters
        ----------
        cond
            The conditioning information

        Returns
        -------
        1. The probability that a sample from the base kernel passes the cut for each row of conditioning information
        2. An iterator over the outcomes of the cut-kernel for each row of conditioning information and the
            log-probability of observing said outcome
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
