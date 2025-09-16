from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from sympy.printing.pytorch import torch
from torch import Tensor, nn


class TrainableKernelBase(ABC, nn.Module):
    def __init__(
        self,
        sample_dimension: int,
        cond_dimension: int
    ):
        super().__init__()
        self.sample_dimension = sample_dimension
        self.cond_dimension = cond_dimension

    @abstractmethod
    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        pass

    @abstractmethod
    def _draw(self, cond: Tensor) -> Tensor:
        pass

    def draw(self, cond: Tensor) -> Tensor:
        with torch.no_grad():
            return self._draw(cond)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        samples = self.draw(cond)
        log_prob = self.log_prob(samples, cond)
        return samples, log_prob

    # def calculate_loss(self, cond, log_p_over_q_model, weights=None):
    #     weights = torch.ones(cond.shape[0], dtype=torch.float32, device=self.device) if weights is None else weights
    #     samples, log_prob = self.draw_with_log_prob(cond)
    #     with torch.no_grad():
    #         p_over_q = torch.exp(log_p_over_q_model(samples))
    #
    #     return -(weights * log_prob * p_over_q).mean()

    def __and__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
        return ConcatenatedKernel.merge(self, other)


class ConcatenatedKernel(TrainableKernelBase):
    def __init__(self, sub_kernels: List[TrainableKernelBase]):
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        super().__init__(sum(k.sample_dimension for k in sub_kernels), sub_kernels[0].cond_dimension)

        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels
        self.register_buffer(
            'cum_kernel_indices',
            torch.tensor(np.cumsum([0] + [k.sample_dimension for k in sub_kernels])).int()
        )

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        log_prob = 0.
        for i, sub_kernel in enumerate(self.sub_kernels):
            log_prob = (
                log_prob
                + sub_kernel.log_prob(samples[self.cum_kernel_indices[i]:self.cum_kernel_indices[i+1]], cond)
            )
        return log_prob

    def _draw(self, cond: Tensor) -> Tensor:
        return torch.cat([k.draw(cond) for k in self.sub_kernels], dim=-1)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        samples, log_probs = zip(k.draw_with_log_prob(cond) for k in self.sub_kernels)
        return (
            torch.cat([k.draw(cond) for k in self.sub_kernels], dim=-1),
            sum(log_probs, torch.tensor(0.)),
        )

    @classmethod
    def merge(cls, a: TrainableKernelBase, b: TrainableKernelBase) -> 'ConcatenatedKernel':
        a_kernels = a.sub_kernels if isinstance(a, ConcatenatedKernel) else [a]
        b_kernels = b.sub_kernels if isinstance(b, ConcatenatedKernel) else [b]
        return ConcatenatedKernel(a_kernels + b_kernels)
