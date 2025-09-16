from typing import List

import torch
from numpy._typing import ArrayLike
from torch import Tensor

from iwpc.learn_dist.trainable_kernel.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MixtureKernel(TrainableKernelBase):
    def __init__(
        self,
        sub_kernels: List[TrainableKernelBase],
        init_weights: ArrayLike,
        weight_model = None
    ):
        assert all(k.sample_dimension == sub_kernels[0].sample_dimension for k in sub_kernels)
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        super().__init__(sub_kernels[0].sample_dimension, sub_kernels[0].cond_dimension)

        self.register_buffer("init_weights", torch.tensor(init_weights, dtype=torch.float32))
        self.init_weights = self.init_weights / self.init_weights.sum()
        self.register_buffer("log_init_weights", torch.log(self.init_weights))

        self.weight_model = basic_model_factory(self.cond_dimension, len(sub_kernels)) if weight_model is None else weight_model
        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels

    def construct_log_prob(self, cond: Tensor) -> Tensor:
        logits = self.weight_model(cond) + self.log_init_weights[None]
        return logits - logits.logsumexp(dim=-1, keepdim=True)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        sub_probs = torch.stack([k.log_prob(samples, cond) for k in self.sub_kernels], dim=-1)
        log_prob = self.construct_log_prob(cond)
        return (sub_probs + log_prob).logsumexp(dim=-1)

    def _draw(self, cond: Tensor) -> Tensor:
        sub_samples = torch.cat([k.draw(cond) for k in self.sub_kernels], dim=-1)
        cum_probs = torch.cumsum(torch.exp(self.construct_log_prob(cond)), dim=-1)
        rand = torch.rand(size=(cond.shape[0], 1), device=cond.device)
        return sub_samples[range(cond.shape[0]), ((cum_probs - rand) > 0).float().argmax(dim=-1)]
