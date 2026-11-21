import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

from learning_agents.utils.tensor_utils import get_device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_padding(kernel_size):
    """Computes padding tuple."""
    # 4 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    assert len(kernel_size) == 2, kernel_size
    computed = [k // 2 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    return [computed[1] - 1 if kernel_size[0] % 2 == 0 else computed[1],
            computed[1],
            computed[0] - 1 if kernel_size[1] % 2 == 0 else computed[0],
            computed[0]]


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed separately for each channel
    in the input using a depth-wise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, verbose=False):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        mesh_grids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, m_grid in zip(kernel_size, sigma, mesh_grids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((m_grid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depth-wise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1)).float()
        self.kernel = kernel.to(device)
        self.kernel_shape = kernel.shape
        if verbose:
            print('[INFO] Gaussian Smoothing Kernel: ', kernel.shape)

        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        height, width = self.kernel_shape[-2:]
        padding_shape = compute_padding((height, width))
        input_pad = F.pad(x, padding_shape, mode='reflect')
        return self.conv(input_pad, weight=self.kernel, groups=self.groups)


class IntensityReducing(nn.Module):
    def __init__(self, degree=2):
        super(IntensityReducing, self).__init__()
        self.degree = degree

    def forward(self, x):
        x = x/float(self.degree)
        return x
