from typing import List, Callable, Iterable

import numpy as np
from torch import Tensor
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Dropout, Module, Sequential, Flatten

from .layers import RunningNormLayer, LambdaLayer
from ..types import Shape


def make_layer_group(in_size: int, out_size: int, dropout: float = 0., batch_norm: bool = False) -> List[Module]:
    """
    Basic linear layer factory supporting dropout and batch normalization

    Parameters
    ----------
    in_size
        The input size of the layer
    out_size
        The output size of the layer
    dropout
        The desired dropout rate. No dropout will be applied if dropout=0
    batch_norm
        Whether to add a batch normalization layer to the output of the layer

    Returns
    -------
    List[Module]
        A list of modules containing a Linear layer and other specified properties
    """
    layers = []
    if dropout > 0:
        layers.append(Dropout(dropout))
    layers += [
        Linear(in_size, out_size),
        LeakyReLU(),
    ]
    if batch_norm:
        layers.append(BatchNorm1d(out_size))
    return layers


def basic_model_factory(
    input_shape: Shape,
    output_shape: Shape = 1,
    hidden_layer_sizes: Iterable[int] = (128, 64, 64, 64, 64),
    final_activation_factory: Callable[[], Callable[[Tensor], Tensor]] = None,
    dropout: float = 0.,
    batch_norm: bool = False,
) -> Module:
    """
    Parameters
    ----------
    input_shape
        The shape of the input
    output_shape
        The desired shape of the output
    hidden_layer_sizes
        A list of the desired shapes of each hidden layer
    final_activation_factory
        A callable that returns an instance of the desired activation layer
    dropout
        The desired dropout for the linear layers. No dropout will be applied if dropout=0
    batch_norm
        Whether to apply a batch normalization layer to the output of each linear layer

    Returns
    -------
    Module
        A nn.Module instance which takes in objects with the given input shape and outputs a Tensor of the given output
        shape
    """
    if isinstance(output_shape, int):
        out_shape = (output_shape,)
    input_size = int(np.prod(input_shape))
    out_size = int(np.prod(output_shape))
    shape = (input_size,) + tuple(hidden_layer_sizes) + (out_size,)
    norm_layer = RunningNormLayer(input_size)
    layers = [
        Flatten(),
        norm_layer,
    ]
    for i in range(len(shape) - 2):
        layers += make_layer_group(shape[i], shape[i + 1], dropout=dropout, batch_norm=batch_norm)
    layers += [
        Linear(shape[-2], shape[-1]),
        LambdaLayer(lambda x: x.reshape((-1,) + out_shape))
    ]

    if final_activation_factory:
        layers += [final_activation_factory()]

    return Sequential(*layers)
