import torch
from torch import Tensor

from iwpc.encodings.encoding_base import Encoding
from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class DiracKernel(TrainableKernelBase):
    """
    A kernel that returns the conditioning information as samples. Equivalent to a dirac delta distribution centered on
    the conditioning information
    """
    def __init__(
        self,
        cond_dimension: int | Encoding,
    ):
        """
        Parameters
        ----------
        cond_dimension
            The dimension of the conditioning space
        """
        super().__init__(
            cond_dimension if isinstance(cond_dimension, int) else cond_dimension.output_shape,
            cond_dimension,
        )

    def log_prob(self, samples: Tensor, cond: Tensor) -> Tensor:
        """
        Returns
        -------
        Tensor
            The log probability of each sample. Note that since the kernel has a constant value, this function should
            return log(1) if sample==cond and log(0) otherwise. However, for speed reasons, the check is skipped and
            log(1) is always returned for every sample. This may change in future
        """
        return torch.zeros(samples.shape[0], dtype=torch.float32, device=cond.device)

    def _draw(self, cond: Tensor) -> Tensor:
        """
        Parameters
        ----------
        cond
            A tensor of conditioning information

        Returns
        -------
        Tensor
            A copy of the conditioning information
        """
        return cond.detach().clone()
