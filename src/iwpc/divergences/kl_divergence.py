import numpy as np
import torch

from .base import DifferentiableFDivergence


class KLDivergence(DifferentiableFDivergence):
    """
    Implementation of the Kullback-Leibler divergence as described in https://arxiv.org/abs/2405.06397
    """

    def __init__(self):
        super().__init__("Kullback-Leibler", "KL")

    def _f_torch(self, x):
        return x * torch.log(x)

    def _f_np(self, x):
        return x * np.log(x)

    def _f_conj_torch(self, x):
        return torch.exp(x - 1)

    def _f_conj_np(self, x):
        return np.exp(x - 1)

    def _f_dash_torch(self, x):
        return 1 + torch.log(x)

    def _f_dash_np(self, x):
        return 1 + np.log(x)
