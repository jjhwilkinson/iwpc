from typing import Any, Callable, Optional

import torch
from numpy._typing import ArrayLike
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
    normalisation of a NN's input. The running mean/std are only updated during training and if the number of samples
    seen is less than self.max_samples.
    """
    def __init__(self, input_shape: Shape, max_samples: int = 100000):
        """
        Parameters
        ----------
        input_shape
            The shape of the input tensors excluding the batch dimension. The batch dimension is assumed to be the first
            dimension
        max_samples
            The number of samples to record before the running average and std deviations are frozen
        """
        super(RunningNormLayer, self).__init__()

        self.register_buffer('sum_', torch.zeros(input_shape))
        self.register_buffer('sq_sum_', torch.zeros(input_shape))
        self.register_buffer('N_', torch.tensor(0.))
        self.max_samples = max_samples

    @property
    def shift(self) -> Tensor:
        """
        Returns
        -------
        Tensor
            The running mean of each input component
        """
        if self.N_ == 0:
            return torch.zeros_like(self.sum_)
        return self.sum_ / max(self.N_, 1.)

    @property
    def scale(self):
        """
        Returns
        -------
        Tensor
            The running standard deviation of each input component
        """
        if self.N_ == 0:
            return torch.ones_like(self.sum_)

        var = self.sq_sum_ / max(self.N_, 1.) - self.shift**2
        if (var <= 0).any():
            return torch.ones_like(self.sum_)

        return torch.sqrt(var)

    def _update(self, x: torch.Tensor) -> None:
        """
        Given a batch of inputs, updates the running sum and sq_sum in each component as well as the total number of
        samples seen thus far

        Parameters
        ----------
        x
            A batch of values. The first dimension is assumed to be the batch dimension
        """
        self.sum_ += x.sum(dim=0).detach()
        self.sq_sum_ += (x**2).sum(dim=0).detach()
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
        if self.training and self.N_ < self.max_samples:
            self._update(x)

        return (x - self.shift) / self.scale


class RunningDeNormLayer(Module):
    """
    Custom normalisation layer that tracks a running average of the mean and standard deviation of its output
    components during training. The output of the layer is defined as its inputs times the rolling standard plus the
    rolling mean of each component. During both validation and training, the input is multiplied by the rolling standard
    deviation of each component and the rolling mean is added. This ensures that the target function for the preceeding
    layers remains normalised
    """
    def __init__(self, input_shape: Shape, one_epoch_only: bool = False):
        super(RunningDeNormLayer, self).__init__()

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
        return self.sum_ / max(self.N_, 1)

    @property
    def scale(self):
        """
        Returns
        -------
        Tensor
            The running standard deviation of each input component
        """
        return torch.sqrt(self.sq_sum_ / max(self.N_, 1) - self.shift**2)

    def _update(self, x: torch.Tensor) -> None:
        """
        Given a batch of outputs, updates the running sum and sq_sum in each component as well as the total number of
        samples seen thus far

        Parameters
        ----------
        x
            A batch of output values. The first dimension is assumed to be the batch dimension
        """
        self.sum_ += x.sum(dim=0).detach()
        self.sq_sum_ += (x**2).sum(dim=0).detach()
        self.N_ += x.shape[0]

    def forward(self, x: Tensor) -> Tensor:
        """
        Multiplies each component by the running standard deviation and adds the running mean. If training, updates the
        running mean and standard deviation in each output component

        Parameters
        ----------
        x
            An input Tensor with the batch dimension in the first position

        Returns
        -------
        Tensor
            The input tensor with each component normalised by the running mean and standard deviation
        """
        x = x * self.scale + self.shift

        if self.training:
            if not self.prev_training:
                self.current_epoch += 1
            if self.current_epoch < 2 or not self.one_epoch_only:
                self._update(x)
        self.prev_training = self.training

        return x


class ConstantScaleLayer(Module):
    """
    A utility layer that scales and shifts the output by a constant value
    """
    @staticmethod
    def standardize(x: Optional[Tensor]) -> Tensor:
        """
        Ensures the given Tensor has the right shape and dtype

        Parameters
        ----------
        x
            A 0D or 1D Tensor or None

        Returns
        -------
        Tensor
            A 2D float Tensor. If x is None, the value is torch.nan
        """
        if x is None:
            return torch.tensor(torch.nan, dtype=torch.float32)
        else:
            x = torch.tensor(x, dtype=torch.float32)

        if x.ndim == 0:
            return x[None, None]
        if x.ndim == 1:
            return x[None, :]

        raise ValueError(f"{x} has more than one dimension")

    def __init__(self, shift: ArrayLike | None = None, scale: ArrayLike | None = None):
        """
        Parameters
        ----------
        shift
            A 0D or 1D Tensor or None used to shift the input by a constant value
        scale
            A 0D or 1D Tensor or None used to scale the input by a constant value
        """
        super().__init__()
        self.register_buffer('shift', ConstantScaleLayer.standardize(shift))
        self.register_buffer('scale', ConstantScaleLayer.standardize(scale))

    def forward(self, x: Tensor) -> Tensor:
        """
        Scales and shifts the input tensor by self.scale and self.shift

        Parameters
        ----------
        x
            A Tensor

        Returns
        -------
        Tensor
            The input tensor with each component scaled and shifted by self.scale and self.shift
        """
        if not self.scale.isnan().all():
            x = x * self.scale
        if not self.shift.isnan().all():
            x = x + self.shift
        return x
