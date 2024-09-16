from typing import List, Callable, Iterable, Optional, Union

import numpy as np
from torch import Tensor, nn
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Dropout, Module, Sequential, Flatten

from .layers import RunningNormLayer, LambdaLayer
from ..encodings.encoding_base import Encoding
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
    input: Union[Encoding, Shape],
    output: Union[Encoding, Shape] = 1,
    hidden_layer_sizes: Iterable[int] = (128, 64, 64, 64, 64),
    dropout: float = 0.,
    batch_norm: bool = False,
    initial_layers: Optional[Iterable[nn.Module]] = None,
    final_layers: Optional[Iterable[nn.Module]] = None,
) -> Sequential:
    """
    Parameters
    ----------
    input
        Either the shape of the input of the network (an int or tuple of ints), or the input encoding of the network.
        If an instance of Encoding, the input shape is inferred is from the encoding dimensions and the encoding is set
        as the first layer of the network
    output
        Either the desired shape of the output of the network (an int or tuple of ints), or the ouput encoding of the
        network. If an instance of Encoding, the output shape is inferred is from the encoding dimensions and the
        encoding is set as the last layer of the network
    hidden_layer_sizes
        A list of the desired shapes of each hidden layer
    dropout
        The desired dropout for the linear layers. No dropout will be applied if dropout=0
    batch_norm
        Whether to apply a batch normalization layer to the output of each linear layer
    initial_layers
        An optional list of any additional layers to insert at the start of the sequential model sequence
    final_layers
        An optional list of any additional layers to insert at the end of the sequential model sequence

    Returns
    -------
    Sequential
        A nn.Module instance which takes in objects with the given input shape and outputs a Tensor of the given output
        shape
    """
    initial_layers = initial_layers or []
    final_layers = final_layers or []
    if isinstance(input, Encoding):
        initial_layers.insert(0, input)
        input_shape = int(input.output_dimension)
    else:
        input_shape = input
    if isinstance(output, Encoding):
        final_layers = [*final_layers, output]
        output_shape = int(output.input_dimension)
    else:
        output_shape = output

    if isinstance(output_shape, int):
        output_shape = (output_shape,)
    input_size = int(np.prod(np.asarray(input_shape)))
    out_size = int(np.prod(np.asarray(output_shape)))
    shape = (input_size,) + tuple(hidden_layer_sizes) + (out_size,)

    norm_layer = RunningNormLayer(input_size)
    layers = [
        *list(initial_layers),
        Flatten(),
        norm_layer,
    ]
    for i in range(len(shape) - 2):
        layers += make_layer_group(shape[i], shape[i + 1], dropout=dropout, batch_norm=batch_norm)
    layers += [
        Linear(shape[-2], shape[-1]),
        LambdaLayer(lambda x: x.reshape((-1,) + output_shape)),
        *list(final_layers),
    ]

    return Sequential(*layers)
