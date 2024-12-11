from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand


max_reduce = FastOps.reduce(operators.max, float("-inf"))


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    # Calculate the new weight and height
    new_height = height // kh
    new_width = width // kw
    # Reshape the input tensor to match dimensions: (batch, channel, new_height, kh, new_width, kw).
    reshaped = input.contiguous().view(batch, channel, new_height, kh, new_width, kw)
    # Permute the input tensor to match dimensions: (batch, channel, new_height, new_width, kh, kw).
    transposed = reshaped.permute(0, 1, 2, 4, 3, 5)
    # Reshape the input tensor to match the dimensions specified in the comment above.
    return (
        transposed.contiguous().view(batch, channel, new_height, new_width, kh * kw),
        new_height,
        new_width,
    )


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply an average pooling 2D operation to an input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    # Tile the tensor that was inputted.
    tiled, new_height, new_width = tile(input, kernel)
    # Calculate the average of the tiled tensor (avgpool2d).
    avg_tiled = tiled.mean(dim=4)
    # Reshape the input shape to match new dimentions
    return avg_tiled.view(avg_tiled.shape[0], avg_tiled.shape[1], new_height, new_width)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Finds the argmax of an input tensor along a particular dimension.

    Args:
    ----
        input: batch x channel x height x width
        dim: dimension to compute the argmax

    Returns:
    -------
        Tensor of size batch x channel x height x width with a one-hot encoding of the argmax.

    """
    max_tensor = max_reduce(input, dim)
    return max_tensor == input


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input: Tensor, dim: Tensor) -> Tensor:
        """Forward pass of Max Function."""
        parse_dim = int(dim._tensor._storage[0])
        ctx.save_for_backward(input, parse_dim)
        return max_reduce(input, parse_dim)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Backward pass of Max Function."""
        input, parse_dim = ctx.saved_values
        return (argmax(input, parse_dim) * grad_output, input._ensure_tensor(parse_dim))


def max(input: Tensor, dim: int) -> Tensor:
    """Apply a max reduction operation to an input tensor along a specified dimension.

    Args:
    ----
        input: tensor of any size
        dim: dimension to compute the max

    Returns:
    -------
        Tensor with a reduced size along the specified dimension.

    """
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Apply a softmax to a tensor along a specified dimension.

    Args:
    ----
        input: tensor of any size
        dim: dimension to calculate the softmax over

    Returns:
    -------
        Softmax Tensor

    """
    max_val = max_reduce(input, dim)
    exp = (input - max_val).exp()
    sum_exp = exp.sum(dim)

    return exp / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Apply a log softmax to a tensor along a specified dimension.

    Args:
    ----
        input: tensor of any size
        dim: dimension to calculate the log softmax over

    Returns:
    -------
        Log Softmax Tensor

    """
    # Implemented following the wikipedia link
    max_val = max_reduce(input, dim)
    diff = input - max_val
    log_sum = (diff).exp().sum(dim).log()
    return diff - log_sum


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply a max pooling 2D operation to an input tensor.

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width

    """
    tiled, new_height, new_width = tile(input, kernel)
    max_tiled = max(tiled, 4)
    return max_tiled.view(max_tiled.shape[0], max_tiled.shape[1], new_height, new_width)


def dropout(input: Tensor, p: float, ignore: bool = False) -> Tensor:
    """Apply a dropout operation to an input tensor base on a specified probability.

    Args:
    ----
        input: tensor of any size
        p: probability of dropping out a value
        ignore: bool to ignore dropout

    Returns:
    -------
        Tensor with dropout applied.

    """
    if ignore:
        return input
    mask = rand(input.shape) > p
    return input * mask
