from functools import partial
from typing import List, Optional, Callable

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
        feature_indices: Optional[List[List[int]]] = None,
        reduction: Callable[[torch.Tensor], torch.Tensor] = partial(torch.mean, dim=-1),
    ):
        """
        Parameters
        ----------
        sub_modules
            A list of submodules
        feature_indices
            If None, each model is evaluated on all input features. If not None, must have the same number of entries as
            sub_modules and each entry must correspond to the list of indices within the set of overall input features
            that each submodule expects to be evaluated on. Each entry may also be None in which case the corresponding
            model is evaluated on all input features
        reduction
            The function used to reduce the collection of outputs from each submodule. Is passed a Tensor where the last
            dimension indexes the submodules. Defaults to a mean over the submodules
        """
        super().__init__()
        assert feature_indices is None or len(sub_modules) == len(feature_indices)
        if feature_indices is None:
            feature_indices = [None] * len(sub_modules)

        self.models = sub_modules
        self.training_indices = []
        self.reduction = reduction
        for i, (indices, model) in enumerate(zip(feature_indices, self.models)):
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
            The output of each submodule evaluated on their respective input features within x reduced using self.reduction
        """
        outs = []
        for indices, model in zip(self.training_indices, self.models):
            if indices is not None:
                outs.append(model(x[:, indices]))
            else:
                outs.append(model(x))
        outs = torch.stack(outs, dim=-1)
        return self.reduction(outs)
