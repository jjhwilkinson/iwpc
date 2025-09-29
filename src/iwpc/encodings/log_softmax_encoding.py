from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class LogSoftmaxEncoding(Encoding):
    """
    Encoding that applies a log-softmax to its input. Can be used to encode a discrete probability distribution
    """
    def __init__(self, num_classes: int):
        """
        Parameters
        ----------
        num_classes
            The number of classes in the input vector
        """
        super().__init__(num_classes, num_classes)

    def _encode(self, logits: Tensor) -> Tensor:
        """
        Returns the log-softmax of the input tensor over the last dimension

        Parameters
        ----------
        logits
            A tensor of logits

        Returns
        -------
        Tensor
            The log-softmax of the input tensor over the last dimension
        """
        return logits - logits.logsumexp(dim=-1, keepdim=True)
