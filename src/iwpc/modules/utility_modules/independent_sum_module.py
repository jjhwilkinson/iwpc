from typing import List, Optional

import torch
from torch import nn


class IndependentSumModule(nn.Module):
    """
    Utility module that wraps a list of submodules. At evaluation time, each submodule is evaluated on a configurable
    subset of the input features, and the submodule output sum is returned
    """
    def __init__(
        self,
        sub_modules: List[nn.Module],
        training_indices: Optional[List[List[int]]] = None,
    ):
        """
        Parameters
        ----------
        sub_modules
            A list of submodules
        training_indices
            If None, each model is evaluated on all input feature. If not None, each entry must correspond to the list
            of indices within the set of overall input features provided to this IndependentSumModule instance that each
            submodule expects to be evaluated on. Must have the same number of
            entries as sub_modules
        """
        super().__init__()
        assert training_indices is None or len(sub_modules) == len(training_indices)

        self.models = sub_modules
        self.register_buffer('training_indices', torch.tensor(training_indices, dtype=torch.int))
        for i, model in enumerate(self.models):
            self.register_module(f"model_{i}", model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x
            The input tensor of features

        Returns
        -------
        Tensor
            The sum of the output of each submodule evaluated on their respective input features within x
        """
        if self.training_indices is None:
            return sum(
                [model(x) for model, index_set in zip(self.models, self.training_indices)],
            )
        return sum(
            [model(x[:, index_set]) for model, index_set in zip(self.models, self.training_indices)],
        )
