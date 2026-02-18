import numpy as np
import torch

from .base import DifferentiableFDivergence


class JensenShannonDivergence(DifferentiableFDivergence):
    """
    Implementation of the Jensen-Shannon divergence as described in https://arxiv.org/abs/2405.06397
    """
    def __init__(self):
        super().__init__("Jensen-Shannon", "JSD")
        self.log_two = torch.log(torch.tensor(2.))

    def _f_torch(self, x):
        return 0.5 * (x * torch.log(x) - (x + 1) * torch.log((x+1) / 2))

    def _f_np(self, x):
        return 0.5 * (x * np.log(x) - (x + 1) * np.log((x+1) / 2))

    def _f_conj_torch(self, x):
        return - 0.5 * (self.log_two + torch.log1p(-0.5 * torch.exp(2 * x)))

    def _f_conj_np(self, x):
        return - 0.5 * (np.log(2.) + np.log1p(-0.5 * np.exp(2 * x)))

    def _f_dash_given_log_torch(self, log_x):
        log_x = torch.clip(log_x, -10, 10)
        return 0.5 * (self.log_two + log_x - torch.logsumexp(torch.stack([log_x, torch.zeros_like(log_x)], dim=1), dim=1))

    def _f_dash_given_log_np(self, log_x):
        return 0.5 * (np.log(2) + log_x - np.logaddexp(log_x, 0.))
