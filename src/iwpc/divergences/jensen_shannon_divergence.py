import numpy as np
import torch

from .base import DifferentiableFDivergence


class JensenShannonDivergence(DifferentiableFDivergence):
    """
    Implementation of the Jensen-Shannon divergence as described in https://arxiv.org/abs/2405.06397
    """
    def __init__(self):
        super().__init__("Jensen-Shannon", "JSD")

    def _f_torch(self, x):
        return 0.5 * (x * torch.log(x) - (x + 1) * torch.log((x+1) / 2))

    def _f_np(self, x):
        return 0.5 * (x * np.log(x) - (x + 1) * np.log((x+1) / 2))

    def _f_conj_torch(self, x):
        return - 0.5 * torch.log(2 - torch.exp(2 * x))

    def _f_conj_np(self, x):
        return - 0.5 * np.log(2 - np.exp(2 * x))

    def _f_dash_torch(self, x):
        return 0.5 * torch.log(2 * x / (x + 1))

    def _f_dash_np(self, x):
        return 0.5 * np.log(2 * x / (x + 1))
