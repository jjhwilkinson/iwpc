from torch import Tensor

from iwpc.encodings.encoding_base import Encoding


class DiscreteLogProbEncoding(Encoding):
    def __init__(self, num_classes: int):
        super().__init__(num_classes, num_classes)

    def _encode(self, logits: Tensor) -> Tensor:
        return logits - logits.logsumexp(dim=-1, keepdim=True)
