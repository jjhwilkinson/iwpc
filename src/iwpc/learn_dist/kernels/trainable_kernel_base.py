from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
from sympy.printing.pytorch import torch
from torch import Tensor, nn

from iwpc.encodings.encoding_base import Encoding


class TrainableKernelBase(ABC, nn.Module):
    def __init__(
        self,
        sample_dimension: int,
        cond: Encoding | int
    ):
        super().__init__()
        self.sample_dimension = sample_dimension
        self.cond_dimension = int(cond.input_shape) if isinstance(cond, Encoding) else cond

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

    def __and__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
        return ConcatenatedKernel.merge(self, other, False)

    def __or__(self, other: 'TrainableKernelBase') -> 'ConcatenatedKernel':
        return ConcatenatedKernel.merge(self, other, True)


class ConcatenatedKernel(TrainableKernelBase):
    def __init__(self, sub_kernels: List[TrainableKernelBase], append_cond=False):
        assert append_cond or all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        cond_dimension = sum(k.cond_dimension for k in sub_kernels) if append_cond else sub_kernels[0].cond_dimension
        super().__init__(sum(k.sample_dimension for k in sub_kernels), cond_dimension)

        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels
        self.append_cond = append_cond
        cum_sample_sizes = np.cumsum([0] + [k.sample_dimension for k in sub_kernels])
        self.sample_edges = [slice(cum_sample_sizes[i], cum_sample_sizes[i+1]) for i in range(len(sub_kernels))]
        if self.append_cond:
            cum_cond_sizes = np.cumsum([0] + [k.cond_dimension for k in sub_kernels])
            self.cond_edges = [slice(cum_cond_sizes[i], cum_cond_sizes[i+1]) for i in range(len(sub_kernels))]
        else:
            self.cond_edges = [slice(0, self.cond_dimension) for _ in range(len(sub_kernels))]

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        log_prob = 0.
        for sample_edges, cond_edges, sub_kernel in zip(self.sample_edges, self.cond_edges, self.sub_kernels):
            log_prob = (
                log_prob
                + sub_kernel.log_prob(samples[:, sample_edges], cond[:, cond_edges])
            )
        return log_prob

    def _draw(self, cond: Tensor) -> Tensor:
        return torch.cat([k.draw(cond[:, cond_edges]) for k, cond_edges in zip(self.sub_kernels, self.cond_edges)], dim=-1)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        samples, log_probs = zip(*[k.draw_with_log_prob(cond[:, c]) for k, c in zip(self.sub_kernels, self.cond_edges)])
        return (
            torch.cat(samples, dim=-1),
            sum(log_probs, torch.tensor(0.)),
        )

    @classmethod
    def merge(cls, a: TrainableKernelBase, b: TrainableKernelBase, append_cond) -> 'ConcatenatedKernel':
        a_kernels = a.sub_kernels if (isinstance(a, ConcatenatedKernel) and a.append_cond==append_cond) else [a]
        b_kernels = b.sub_kernels if (isinstance(b, ConcatenatedKernel) and b.append_cond==append_cond) else [b]

        return ConcatenatedKernel(a_kernels + b_kernels, append_cond=append_cond)
