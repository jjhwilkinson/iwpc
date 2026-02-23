from functools import partial
from typing import List, Optional, Callable

import torch
from torch import nn

from iwpc.encodings.encoding_base import Encoding


class IndependentSumModule(nn.Module):
    """
    Utility module that wraps a list of submodules. At evaluation time, each submodule is evaluated on a configurable
    subset of the input features, and an encoded average of the submodule outputs is returned
    """
    def __init__(
        self,
        sub_modules: List[nn.Module],
        feature_indices: Optional[List[List[int]]] = None,
        output_encoding: Encoding | None = None,
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
        output_encoding
            An optional encoding to apply to the resulting average over sub_module outputs
        """
        super().__init__()
        assert feature_indices is None or len(sub_modules) == len(feature_indices)
        if feature_indices is None:
            feature_indices = [None] * len(sub_modules)

        self.models = sub_modules
        self.training_indices = []
        for i, (indices, model) in enumerate(zip(feature_indices, self.models)):
            if indices is not None:
                self.register_buffer(f"indices_{i}", torch.tensor(indices, dtype=torch.long))
                self.training_indices.append(getattr(self, f"indices_{i}"))
            else:
                self.training_indices.append(None)
            self.register_module(f"model_{i}", model)
        self.output_encoding = output_encoding

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
        outs = torch.mean(torch.stack(outs, dim=-1), dim=-1)
        if self.output_encoding is not None:
            outs = self.output_encoding(outs)
        return outs
