from iwpc.learn_dist.kernels.trainable_kernel_base import TrainableKernelBase


class GaussianTwoSidedExponentialKernel(TrainableKernelBase):
    def __init__(self, init_mean, init_gaussian_std, ):
        self.init_mean = init_mean
        self.init_std = init_std