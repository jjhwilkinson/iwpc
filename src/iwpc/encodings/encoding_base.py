from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np
import torch
from numpy._typing import ArrayLike
from torch import nn, Tensor


def to_shape_tensor(x: ArrayLike) -> Tensor:
    """
    Converts the given object to a tensor that is a valid shape.
    """
    x = torch.tensor(x)
    if x.dim() == 0:
        x = x.unsqueeze(0)
    return x


class Encoding(nn.Module, ABC):
    """
    Base class for all encodings layers. An encoding is a simple transformation on its inputs intended to change the
    representation of said information into a form more suitable for the machine learning task at hand. For example, when
    learning a continuous function of an angle, theta, it is advantageous to provide a NN as input cos(theta) and
    sin(theta) rather than theta directly (see ContinuousPeriodicEncoding). Or, when learning a function known to be
    even under the inversion of one of its inputs, x -> -x, it would be advantageous to provide a NN as input |x| rather
    than x directly to enforce this property of the learnt function. Encoding layers can be placed at the start of a
    sequential model to perform these transformation without needing to store or manage the state of any additional
    values in the actual datasets themselves. While this might be marginally slower, this is a much better abstraction.
    A number of utilities are provided to make writing encodings very easy. In particular, the bitwise-and operation '&'
    between two encoding of input dimension d1 and d2 will return a new encoding of dimension d1+d2 wherein the first
    encoding is applied to the first d1 features of an input vector and the second encoding is applied to the remaining
    d2 features and the results concatenated. For example, an encoding for a feature vector containing a radius, which
    should be fed directly to the network, and an angle, which should be continuously represnted might be constructed
    using

    input_encoding = TrivialEncoding(1) & ContinuousPeriodicEncoding()

    In this case, the encoding of a feature vector (r, theta) would be the triplet (r, cos(theta), sin(theta))
    """
    def __init__(self, input_shape: Union[int, ArrayLike], output_shape: Union[int, ArrayLike]):
        """
        Parameters
        ----------
        input_shape
            The number of input features expected by the encoding layer.
        output_shape
            The number of output features produced by the encoding layer.
        """
        super().__init__()
        self.register_buffer('input_shape', to_shape_tensor(input_shape).int())
        self.register_buffer('output_shape', to_shape_tensor(output_shape).int())

    @property
    def is_vector_input(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the input to the Encoding is a feature vector as opposed to some higher rank object
        """
        return self.input_shape.shape[0] == 1
    
    @property
    def is_vector_output(self) -> bool:
        """
        Returns
        -------
        bool
            Whether the output of the Encoding is a feature vector as opposed to some higher rank object
        """
        return self.output_shape.shape[0] == 1

    @abstractmethod
    def _encode(self, x: Tensor) -> Tensor:
        """
        Perform the encoding and return the result. Subclasses must implement this method.

        Parameters
        ----------
        x
            A Tensor of dimension (..., *input_shape)

        Returns
        -------
        Tensor
            The encoded information with dimension (..., *output_shape)
        """

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluates the _encode function

        Parameters
        ----------
        x
            A Tensor of dimension (..., *input_shape)

        Returns
        -------
        Tensor
            The encoded information with dimension (..., *output_shape)
        """
        return self._encode(x)

    def __and__(self, other: 'Encoding') -> 'ConcatenatedEncoding':
        """
        Constructs a ConcatenatedEncoding instance that performs both the original encodings to adjacent features in a
        feature vector

        Parameters
        ----------
        other
            Any other Encoding

        Returns
        -------
        ConcatenatedEncoding
        """
        return ConcatenatedEncoding.merge(self, other)


class ConcatenatedEncoding(Encoding):
    """
    A wrapper encoding based on a list of 'sub-encodings' of input dimensions d1...dN and output dimensions o1...oN.
    Evaluates each successive sub-encoding on its respective section of an input feature vector of length d1+...+dN and
    returns the concatenated result of length o1+...+oN. Any collection of vector encodings, that is encodings take in
    and output a vector of some length, may be concatenated. Nested ConcatenatedEncoding instances are automatically
    un-curried when constructed using the bitwise and operator, '&', or ConcatenatedEncoding.merge
    """
    def __init__(self, sub_encodings: List[Encoding]):
        """
        Parameters
        ----------
        sub_encodings
            A list of encodings
        """
        if not all(encoding.is_vector_input for encoding in sub_encodings):
            raise ValueError('Can only concatenate encodings that operate on vector inputs')
        if not all(encoding.is_vector_output for encoding in sub_encodings):
            raise ValueError('Can only concatenate encodings that have vector outputs')
        
        super().__init__(
            sum(encoding.input_shape[0] for encoding in sub_encodings),
            sum(encoding.output_shape[0] for encoding in sub_encodings),
        )
        self.register_buffer(
            'cum_input_shapes',
            torch.tensor(np.cumsum([0] + [encoding.input_shape[0] for encoding in sub_encodings])).int()
        )

        self.sub_encodings = sub_encodings

    def _encode(self, x: Tensor) -> Tensor:
        """
        Applies the j'th sub-encoding to the subset of the feature vector between cum_input_shapes[j] and
        cum_input_shapes[j+1]. The resulting encoded features are concatenated

        Parameters
        ----------
        x
            A tensor of shape (..., *input_shape)

        Returns
        -------
        Tensor
            of shape (..., *output_shape)
        """
        return torch.concatenate([
            encoding(x[..., low:high]) for encoding, low, high in
            zip(self.sub_encodings, self.cum_input_shapes[:-1], self.cum_input_shapes[1:])
        ], dim=-1)

    @classmethod
    def merge(cls, a: Encoding, b: Encoding) -> 'ConcatenatedEncoding':
        """
        Constructs a ConcatenatedEncoding from the encodings a and b. If either is itself a ConcatenatedEncoding
        instance, the contents of the ConcatenatedEncoding is used. Note both a and b must input and output 1D vectors

        Parameters
        ----------
        a
            A vector encoding
        b
            A vector encoding

        Returns
        -------
        ConcatenatedEncoding
            The concatenation of a and b
        """
        a_encodings = a.sub_encodings if isinstance(a, ConcatenatedEncoding) else [a]
        b_encodings = b.sub_encodings if isinstance(b, ConcatenatedEncoding) else [b]
        return ConcatenatedEncoding(a_encodings + b_encodings)
