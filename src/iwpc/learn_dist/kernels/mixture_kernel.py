from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding
from iwpc.learn_dist.kernels.finite_kernel_interface import sample_idx_from_logits
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MixtureKernel(TrainableKernelBase):
    """
    Utility kernel representing a mixture distribution of its sub-kernels with learnable relative weights. The resulting
    probability of a sample is given by the weighted sum of each sub-kernel producing the sample
    """
    def __init__(
        self,
        sub_kernels: List[TrainableKernelBase],
        cond: Encoding | int,
        log_probability_model: Optional[Module] = None,
    ):
        """
        Parameters
        ----------
        sub_kernels
            The sub-kernels from which to draw with a learned relative probability
        cond
            The conditioning space encoding or dimension
        log_probability_model
            Optional model that constructs the logarithm of the probability of drawing from each sub-kernel. The
            LogSoftmaxEncoding might be useful
        """
        assert all(k.sample_dimension == sub_kernels[0].sample_dimension for k in sub_kernels)
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        super().__init__(sub_kernels[0].sample_dimension, cond)

        self.sub_kernels = ModuleList(sub_kernels)
        self.log_probability_model = basic_model_factory(
            cond,
            LogSoftmaxEncoding(len(sub_kernels))
        ) if log_probability_model is None else log_probability_model

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Log probability of samples given the conditioning information. The probability is taken as the weighted sum
            of the probability that each sub-kernel produces the sample
        """
        sub_probs = torch.stack([k.log_prob(samples, cond) for k in self.sub_kernels], dim=-1)
        log_prob = self.log_probability_model(cond)
        return (sub_probs + log_prob).logsumexp(dim=-1)

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            A sample from drawn from one of the sub-kernels for each row of conditioning information. The probability of
            each sub-kernel being chosen is given by self.log_probability_model
        """
        sub_samples = torch.stack([k.draw(cond) for k in self.sub_kernels], dim=-1)
        labels = sample_idx_from_logits(self.log_probability_model(cond))
        samples = sub_samples[range(cond.shape[0]), ..., labels]

        return samples
