from functools import partial, reduce
from operator import mul
from typing import List, Callable, Iterable, Optional, Union, Dict

import torch
from torch import nn, Tensor
from torch.nn import BatchNorm1d, LeakyReLU, Linear, Dropout, Module, Sequential, Flatten

from .layers import RunningNormLayer, LambdaLayer
from ..encodings.encoding_base import Encoding
from ..encodings.trivial_encoding import TrivialEncoding
from ..modules.utility_modules.independent_sum_module import IndependentSumModule
from ..symmetries.group_action import GroupAction
from ..types import Shape


def make_layer_group(
    in_size: int,
    out_size: int,
    dropout: float = 0.,
    batch_norm: bool = False,
    activation: Callable = LeakyReLU,
) -> List[Module]:
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
    activation
        The activation function class to apply to the output of layers

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
        activation(),
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
    symmetries: Optional[Union[GroupAction, Iterable[GroupAction]]] = None,
    complement_symmetries: Optional[Union[GroupAction, Iterable[GroupAction]]] = None,
    activation: Callable = LeakyReLU,
) -> Sequential:
    """
    Parameters
    ----------
    input
        Either the shape of the input of the network (an int or tuple of ints), or the input encoding of the network.
        If an instance of Encoding, the input shape is inferred is from the encoding dimensions and the encoding is set
        as the first layer of the network. Only Encodings with vector outputs are supported as an input encoding.
    output
        Either the desired shape of the output of the network (an int or tuple of ints), or the output encoding of the
        network. If an instance of Encoding, the output shape is inferred is from the encoding dimensions and the
        encoding is used as the last layer of the network. Only Encodings with vector inputs are supported as an output
        encoding.
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
    symmetries
        A symmetry group action under which the network should be invariant. To combine multiple symmetries,
        compose them declaratively with '*' (joint action on the same space) or '&' (direct product on disjoint
        dim ranges), e.g. ``G1 * G2``. An iterable of GroupActions is also accepted as a convenience and is folded
        via '*' to match the previous behaviour of symmetrizing over each group on the same space
    complement_symmetries
        A symmetry group action for which the network output should reside in the symmetrized complement.
        Compose multiple symmetries declaratively with '*' or '&'. An iterable of GroupActions is also accepted as
        a convenience and is folded via '*'
    activation
        The activation function class to apply to the output of layers

    Returns
    -------
    Sequential
        A nn.Module instance which takes in objects with the given input shape and outputs a Tensor of the given output
        shape
    """
    initial_layers = initial_layers or []
    final_layers = final_layers or []
    if isinstance(input, Encoding):
        if not input.is_vector_output:
            raise ValueError("Only vector output Encodings are supported as a basic_model_factory input Encoding")
        initial_layers.insert(0, input)
        input_shape = int(input.output_shape[0])
    else:
        input_shape = input
    if not isinstance(input_shape, Iterable) or (hasattr(input_shape, "shape") and input_shape.shape == tuple()):
        input_shape = (input_shape,)

    if isinstance(output, Encoding):
        if not output.is_vector_input:
            raise ValueError("Only vector input Encodings are supported as a basic_model_factory output Encoding")
        final_layers = [*final_layers, output]
        output_shape = int(output.input_shape[0])
    else:
        output_shape = output
    if not isinstance(output_shape, Iterable) or (hasattr(output_shape, "shape") and output_shape.shape == tuple()):
        output_shape = (output_shape,)

    input_size = int(torch.prod(torch.tensor(input_shape)))
    out_size = int(torch.prod(torch.tensor(output_shape)))
    shape = (input_size,) + tuple(hidden_layer_sizes) + (out_size,)

    norm_layer = RunningNormLayer(input_size)
    layers = [
        *list(initial_layers),
        Flatten(),
        norm_layer,
    ]
    for i in range(len(shape) - 2):
        layers += make_layer_group(shape[i], shape[i + 1], dropout=dropout, batch_norm=batch_norm, activation=activation)
    layers += [
        Linear(shape[-2], shape[-1]),
        LambdaLayer(lambda x: x.reshape((-1,) + output_shape)),
        *list(final_layers),
    ]

    model = Sequential(*layers)
    symmetry_group = _coerce_group_action(symmetries)
    if symmetry_group is not None:
        model = symmetry_group.symmetrize(model)
    complement_group = _coerce_group_action(complement_symmetries)
    if complement_group is not None:
        model = complement_group.complement(model)

    return model


def _coerce_group_action(
    symmetries: Optional[Union[GroupAction, Iterable[GroupAction]]],
) -> Optional[GroupAction]:
    """
    Coerces the symmetries argument of basic_model_factory into a single GroupAction. A None or empty iterable returns
    None. A single GroupAction is returned as-is. An iterable of GroupActions is folded together via the '*' operator
    (joint action on the same space) so the resulting model is symmetrized once over the joint group, matching the
    semantics of the previous nested-symmetrize loop

    Parameters
    ----------
    symmetries
        Either None, a single GroupAction, or an iterable of GroupActions

    Returns
    -------
    Optional[GroupAction]
        A single GroupAction representing the combined symmetries, or None if no symmetries were provided
    """
    if symmetries is None:
        return None
    if isinstance(symmetries, GroupAction):
        return symmetries
    groups = list(symmetries)
    if len(groups) == 0:
        return None
    return reduce(mul, groups)


def basic_model_factory_sum(
    specs: Iterable[Dict],
    output: Encoding | int,
    **common_spec,
) -> IndependentSumModule:
    """
    Shorthand for creating a model that is the sum of a number of sub models. Useful for when you want submodules with
    different symmetries or input encodings. Models are combined using an IndependentSumModule, and the final output is
    encoded using the 'output' parameter

    Parameters
    ----------
    specs
        A list of dictionaries describing the sub-modules to be passed to basic_model_factory
    output
        The output encoding of the resulting model. Defaults to a TrivialEncoding if an integer is provided
    common_spec
        Key word args to serve as the base for each sub-module to be passed basic_model_factory. Options are overridden
        by the specs above


    Returns
    -------
    IndependentSumModule
    """
    output = TrivialEncoding(output) if isinstance(output, int) else output

    models = []
    for spec_mods in specs:
        spec = common_spec.copy()
        spec.update(spec_mods)
        spec['output'] = tuple(int(dim) for dim in output.input_shape)
        models.append(basic_model_factory(**spec))

    return IndependentSumModule(
        models,
        output_encoding=output,
    )
