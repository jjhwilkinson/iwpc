from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import Module

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class MixtureKernel(TrainableKernelBase):
    """
    Utility kernel representing a mixture distribution of its sub-kernels wit learnable relative weights. The resulting
    probability of a sample is given by the weighted sum of each sub-kernel producing the sample
    """
    def __init__(
        self,
        cond: Encoding | int,
        sub_kernels: List[TrainableKernelBase],
        log_probability_model: Optional[Module] = None,
        explicit_mixture_label: bool = False,
    ):
        """
        Parameters
        ----------
        cond
            The conditioning space encoding or dimension
        sub_kernels
            The sub-kernels from which to draw with a learned relative probability
        log_probability_model
            Optional model that constructs the logarithm of the probability of drawing from each sub-kernel. The
            LogSoftmaxEncoding might be useful
        explicit_mixture_label
            Whether the label determining which sub-kernel to sample from should be included in drawn samples
        """
        assert all(k.sample_dimension == sub_kernels[0].sample_dimension for k in sub_kernels)
        assert all(k.cond_dimension == sub_kernels[0].cond_dimension for k in sub_kernels)
        if explicit_mixture_label:
            super().__init__(sub_kernels[0].sample_dimension + 1, cond)
        else:
            super().__init__(sub_kernels[0].sample_dimension, cond)

        for i, sub_kernel in enumerate(sub_kernels):
            self.register_module(f"sub_kernel_{i}", sub_kernel)
        self.sub_kernels = sub_kernels
        self.log_probability_model = basic_model_factory(
            cond,
            LogSoftmaxEncoding(len(sub_kernels))
        ) if log_probability_model is None else log_probability_model
        self.explicit_mixture_label = explicit_mixture_label

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            Log probability of samples given the conditioning information. The probability is taken as the weighted sum
            of the probability that each sub-kernel produces the sample
        """
        if self.explicit_mixture_label:
            label_samples, kernel_samples = samples[:, 0], samples[:, 1:]
            sub_probs = torch.stack([k.log_prob(kernel_samples, cond) for k in self.sub_kernels], dim=-1)
            log_prob = self.log_probability_model(cond)
            return (sub_probs + log_prob)[range(samples.shape[0]), label_samples]
        else:
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
            each sub-kernel being chosen is given by self.log_probability_model. If explicit_mixture_label is True
            then the label of the chosen sub-kernel is included as the first component of the sample
        """
        sub_samples = torch.cat([k.draw(cond) for k in self.sub_kernels], dim=-1)
        cum_probs = torch.cumsum(torch.exp(self.log_probability_model(cond)), dim=-1)
        rand = torch.rand(size=(cond.shape[0], 1), device=cond.device)
        labels = ((cum_probs - rand) > 0).float().argmax(dim=-1)
        if self.explicit_mixture_label:
            return sub_samples[range(cond.shape[0]), labels]
        else:
            return torch.cat([
                labels[:, None],
                sub_samples[range(cond.shape[0]), labels]
            ], dim=-1)
