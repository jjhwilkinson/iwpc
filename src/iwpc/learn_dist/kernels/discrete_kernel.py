from typing import Iterable, Tuple

import numpy as np
import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.encodings.log_softmax_encoding import LogSoftmaxEncoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase
from iwpc.models.utils import basic_model_factory


class DiscreteKernel(TrainableKernelBase):
    """
    Kernel for discrete outcomes. Often discrete probability spaces are constructed as the cartesian product over
    variables. For example, consider the variables A, B, C that can either be true or false. There are 8 possible
    outcomes corresponding to {(not A and not B and not C), (not A and not B and C) , etc}. The sample space of this
    kernel is an integer vector of length equal to the number of distinct variables with each entry between 0 and the
    number of values said variables can take less one. In the above ABC example, samples are vectrs of length three
    and entries equal to 0 or 1
    """
    def __init__(
        self,
        num_outcomes: int | Iterable[int],
        cond: Encoding | int,
        logit_model: torch.nn.Module | None = None,
    ):
        """
        Parameters
        ----------
        num_outcomes
            A tuple of integers representing the number of possible values per variable. The product of the constituents
            gives the total number of possible outcomes. If an integer is given, it is interpreted as the tuple
            (num_outcomes,). In the ABC example, this would be (2, 2, 2)
        cond
            The conditioning space encoding or dimension
        """
        if isinstance(num_outcomes, int):
            num_outcomes = (num_outcomes,)
        super().__init__(len(num_outcomes), cond)
        self.num_outcomes = tuple(num_outcomes)
        self.total_outcomes = int(np.prod(num_outcomes))
        self.logit_model = basic_model_factory(
            cond,
            LogSoftmaxEncoding(self.total_outcomes)
        ) if logit_model is None else logit_model

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Calculates the log probability of the given samples for the given conditioning information

        Parameters
        ----------
        samples
            A tensor of size (N, len(num_outcomes)) of integers
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            A tensor of shape (N,)
        """
        logit_slice = [range(samples.shape[0])]
        samples = samples.int()
        for i in range(samples.shape[1]):
            logit_slice.append(samples[:, i])
        logits = self.logit_model(cond).reshape((-1,) + self.num_outcomes)
        return logits[*logit_slice]

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tensor
            A tensor of shape (cond.shape[0], len(num_outcomes)) of integers
        """
        probs = self.logit_model(cond).exp()
        cum_probs = torch.cumsum(probs, dim=-1)
        rand = torch.rand(size=(cond.shape[0], 1), device=cond.device)
        samples = ((cum_probs - rand) > 0).float().argmax(dim=-1)

        return torch.stack(torch.unravel_index(samples, self.num_outcomes), dim=-1)

    def draw_with_log_prob(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Parameters
        ----------
        cond
            A Tensor of conditioning vectors

        Returns
        -------
        Tuple[Tensor, Tensor]
            A tensor of shape (cond.shape[0], len(num_outcomes)) of integers and a Tensor of shape (N,) containing the
            corresponding log probabilities
        """
        probs = self.logit_model(cond).exp()
        cum_probs = torch.cumsum(probs, dim=-1)
        rand = torch.rand(size=(cond.shape[0], 1), device=cond.device)
        samples = ((cum_probs - rand) > 0).float().argmax(dim=-1)

        return (
            torch.stack(torch.unravel_index(samples, self.num_outcomes), dim=-1),
            probs[range(probs.shape[0]), samples]
        )
