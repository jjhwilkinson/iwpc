from typing import Any, Callable, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module

from ..types import Shape


class LambdaLayer(Module):
    """
    Wrapper Layer for a lambda function so it can be added to a pytorch Sequential model
    """
    def __init__(self, lambda_: Callable):
        super(LambdaLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x: Any) -> Any:
        return self.lambda_(x)


class RunningNormLayer(Module):
    """
    Custom normalisation layer that tracks a running average of the mean and standard deviation of each input
    component during training. During both validation and training, the current rolling mean is subtracted off each
    component and divided by the current value of the rolling standard deviation. Used to automatically handle the
    normalisation of a NN's input.
    """
    def __init__(self, input_shape: Shape, one_epoch_only: bool = False):
        """
        Parameters
        ----------
        input_shape
            The shape of the input tensors excluding the batch dimension. The batch dimension is assumed to be the first
            dimension
        one_epoch_only
            If true, the running mean and standard deviations will only be tracked for the first training epoch, then
            fixed to the attained value.
        """
        super(RunningNormLayer, self).__init__()

        self.register_buffer('sum_', torch.zeros(input_shape))
        self.register_buffer('sq_sum_', torch.zeros(input_shape))
        self.register_buffer('N_', torch.tensor(0.))
        self.prev_training = False
        self.current_epoch = 0
        self.one_epoch_only = one_epoch_only

    @property
    def shift(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            The running mean of each input component
        """
        return self.sum_ / self.N_

    @property
    def scale(self):
        """
        Returns
        -------
        Tensor
            The running standard deviation of each input component
        """
        return torch.sqrt(self.sq_sum_ / self.N_ - self.shift**2)

    def _update(self, x: torch.Tensor) -> None:
        """
        Given a batch of inputs, updates the running sum and sq_sum in each component as well as the total number of
        samples seen thus far

        Parameters
        ----------
        x
            A batch of data. The first dimension is assumed to be the batch dimension
        """
        self.sum_ += x.sum(dim=0)
        self.sq_sum_ += (x**2).sum(dim=0)
        self.N_ += x.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        """
        If training, updates the running mean and standard deviation in each input component and normalises each
        component by subtracting off the running mean and dividing by the running standard deviation. In validation,
        each component is normalised without updating the running values

        Parameters
        ----------
        x
            An input Tensor with the batch dimension in the first position

        Returns
        -------
        Tensor
            The input tensor with each component normalised by the running mean and standard deviation
        """
        if self.training:
            if not self.prev_training:
                self.current_epoch += 1
            if self.current_epoch < 2 or not self.one_epoch_only:
                self._update(x)
        self.prev_training = self.training

        return (x - self.shift) / self.scale
