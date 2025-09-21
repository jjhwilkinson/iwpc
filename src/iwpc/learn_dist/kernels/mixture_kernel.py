from typing import List

import torch
from torch import Tensor

from iwpc.encodings.discrete_log_prob_encoding import DiscreteLogProbEncoding
from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MixtureKernel(TrainableKernelBase):
    def __init__(
        self,
        cond: Encoding | int,
        sub_kernels: List[TrainableKernelBase],
        log_weight_model = None
    ):
        assert all(k.sample_dimension == sub_kernels[0].sample_dimension for k in sub_kernels)
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        super().__init__(sub_kernels[0].sample_dimension, cond)

        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels
        self.log_weight_model = basic_model_factory(
            cond,
            DiscreteLogProbEncoding(len(sub_kernels))
        ) if log_weight_model is None else log_weight_model

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        sub_probs = torch.stack([k.log_prob(samples, cond) for k in self.sub_kernels], dim=-1)
        log_prob = self.log_weight_model(cond)
        return (sub_probs + log_prob).logsumexp(dim=-1)

    def _draw(self, cond: Tensor) -> Tensor:
        sub_samples = torch.cat([k.draw(cond) for k in self.sub_kernels], dim=-1)
        cum_probs = torch.cumsum(torch.exp(self.log_weight_model(cond)), dim=-1)
        rand = torch.rand(size=(cond.shape[0], 1), device=cond.device)
        return sub_samples[range(cond.shape[0]), ((cum_probs - rand) > 0).float().argmax(dim=-1)]
