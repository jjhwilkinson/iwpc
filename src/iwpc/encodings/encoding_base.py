from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from torch import nn, Tensor


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
    def __init__(self, input_dimension: int, output_dimension: int):
        """
        Parameters
        ----------
        input_dimension
            The number of input features expected by the encoding layer.
        output_dimension
            The number of output features produced by the encoding layer.
        """
        super().__init__()
        self.register_buffer('input_dimension', torch.tensor(input_dimension).int())
        self.register_buffer('output_dimension', torch.tensor(output_dimension).int())

    @abstractmethod
    def _encode(self, x) -> Tensor:
        """
        Perform the encoding and return the result. Subclasses must implement this method.

        Parameters
        ----------
        x
            A Tensor of dimension (..., input_dimension)

        Returns
        -------
        Tensor
            The encoded information with dimension (..., output_dimension)
        """

    def forward(self, x: Tensor) -> Tensor:
        """
        Evaluates the _encode function

        Parameters
        ----------
        x
            A Tensor of dimension (..., input_dimension)

        Returns
        -------
        Tensor
            The encoded information with dimension (..., output_dimension)
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
    returns the concatenated result of length o1+...+oN. Any two encodings may be concatenated and nested
    ConcatenatedEncoding instances are automatically un-curried when using '&' or ConcatenatedEncoding.merge
    """
    def __init__(self, sub_encodings: List[Encoding]):
        """
        Parameters
        ----------
        sub_encodings
            A list of encodings
        """
        super().__init__(
            sum(encoding.input_dimension for encoding in sub_encodings),
            sum(encoding.output_dimension for encoding in sub_encodings),
        )
        self.register_buffer(
            'cum_input_dimensions',
            torch.tensor(np.cumsum([0] + [encoding.input_dimension for encoding in sub_encodings])).int()
        )

        self.sub_encodings = sub_encodings

    def _encode(self, x: Tensor) -> Tensor:
        """
        Applies the j'th sub-encoding to the subset of the feature vector between cum_input_dimensions[j] and
        cum_input_dimensions[j+1]. The resulting encoded features are concatenated

        Parameters
        ----------
        x
            A tensor of shape (..., input_dimension)

        Returns
        -------
        Tensor
            of shape (..., output_dimension)
        """
        return torch.concatenate([
            encoding(x[..., low:high]) for encoding, low, high in
            zip(self.sub_encodings, self.cum_input_dimensions[:-1], self.cum_input_dimensions[1:])
        ], dim=-1)

    @classmethod
    def merge(cls, a: Encoding, b: Encoding) -> 'ConcatenatedEncoding':
        """
        Constructs a ConcatenatedEncoding from the encodings a and b. If either is itself a ConcatenatedEncoding
        instance, the contents of the ConcatenatedEncoding is used

        Parameters
        ----------
        a
        b

        Returns
        -------

        """
        a_encodings = a.sub_encodings if isinstance(a, ConcatenatedEncoding) else [a]
        b_encodings = b.sub_encodings if isinstance(b, ConcatenatedEncoding) else [b]
        return ConcatenatedEncoding(a_encodings + b_encodings)
