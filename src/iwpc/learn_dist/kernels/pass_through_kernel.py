import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class PassThroughKernel(TrainableKernelBase):
    """
    Utility Kernel to pass through the first num_components of the conditioning information as samples
    """
    def __init__(self, num_components: int, cond: int | Encoding):
        """
        Parameters
        ----------
        num_components
            The number of components to pass through from the conditioning information
        cond
            The dimension of the conditioning information
        """
        super().__init__(num_components, cond)

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        samples
        cond

        Returns
        -------
        Tensor
            A Tensor of zeros
        """
        return torch.zeros(samples.shape[0], dtype=samples.dtype, device=samples.device)

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information

        Returns
        -------
        Tensor
            A tensor of the first num_components of the conditioning information
        """
        return cond[:, :self.sample_dimension]
