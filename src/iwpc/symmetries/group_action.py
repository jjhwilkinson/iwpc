from abc import ABC, abstractmethod
from typing import Tuple

from torch.nn import Module

from .group_action_element import GroupActionElement


class GroupAction(ABC, Module):
    """
    Abstract interface for group actions acting on the function space accessible to a NN from R^M -> R^N. We restrict
    ourselves to actions that act separately on the input and output spaces, that is group actions that can be expressed
    in the form [g⋅f](x) = g⋅(f(g⋅x)) for some action of G on R^M and R^N separately. In particular, provides the batch
    method enabling averaging over the group by averaging over batches of its action
    """
    @abstractmethod
    def batch(self) -> Tuple[GroupActionElement]:
        """
        Provides a batch of group action elements sampled from the Haar measure of the group. Small finite groups
        should return all elements in every batch, but larger and even infinite groups should return a batch of samples
        from the Haar measure of the group.

        Returns
        -------
        Tuple[GroupActionElement]
        """

    def symmetrize(self, base_model: Module) -> "SymmetrizedModel":
        """
        Helper function to wrap a model in a SymmetrizedModel resulting in a model symmetric with respect to this group
        action

        Parameters
        ----------
        base_model
            A base model to symmetrize

        Returns
        -------
        SymmetrizedModel
            A symmetrized model
        """
        from .symmetrized_model import SymmetrizedModel
        return SymmetrizedModel(self, base_model)

    def complement(self, base_model: Module) -> "ComplementModel":
        """
        Helper function to wrap a model in a ComplementModel resulting in a model in the complement of the symmetrization
        projection of this group action

        Parameters
        ----------
        base_model
            A base model to symmetrize

        Returns
        -------
        SymmetrizedModel
            A symmetrized model
        """
        from .complement_model import ComplementModel
        return ComplementModel(self, base_model)
