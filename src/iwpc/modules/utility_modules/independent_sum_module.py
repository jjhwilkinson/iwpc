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
            If None, each model is evaluated on all input feature. If not None, must have the same number of entries as
            sub_modules and each entry must correspond to the list of indices within the set of overall input features
            that each submodule expects to be evaluated on. Each entry may also be None in which case the corresponding
            model is evaluated on all input features
        """
        super().__init__()
        assert training_indices is None or len(sub_modules) == len(training_indices)
        if training_indices is None:
            training_indices = [None] * len(sub_modules)

        self.models = sub_modules
        self.training_indices = []
        for i, (indices, model) in enumerate(zip(training_indices, self.models)):
            if indices is not None:
                self.register_buffer(f"indices_{i}", torch.tensor(indices, dtype=torch.long))
                self.training_indices.append(getattr(self, f"indices_{i}"))
            else:
                self.training_indices.append(None)
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
        sum_ = 0
        for indices, model in zip(self.training_indices, self.models):
            if indices is not None:
                sum_ = model(x[:, indices]) + sum_
            else:
                sum_ = model(x) + sum_
        return sum_