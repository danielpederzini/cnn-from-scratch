import cupy as cp
from typing import Tuple


def get_im2col_indices(
    input_shape: Tuple[int, int, int, int],
    kernel_height: int,
    kernel_width: int,
    padding: int,
    stride: int
) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, int, int]:
    """
    Compute shared indexing tensors for im2col and col2im.

    Args:
        input_shape: Input tensor shape as (batch_size, num_channels, height, width)
        kernel_height: Height of the extraction kernel
        kernel_width: Width of the extraction kernel
        padding: Zero-padding applied around the input
        stride: Stride between adjacent windows

    Returns:
        Tuple containing channel and spatial index tensors along with output dimensions
    """
    _, num_channels, img_height, img_width = input_shape

    img_height_padded: int = img_height + 2 * padding
    img_width_padded: int = img_width + 2 * padding

    output_height: int = (img_height_padded - kernel_height) // stride + 1
    output_width: int = (img_width_padded - kernel_width) // stride + 1

    i_offset: cp.ndarray = cp.repeat(cp.arange(kernel_height), kernel_width)
    i_offset = cp.tile(i_offset, num_channels)

    j_offset: cp.ndarray = cp.tile(cp.arange(kernel_width), kernel_height)
    j_offset = cp.tile(j_offset, num_channels)

    i_output: cp.ndarray = stride * cp.repeat(cp.arange(output_height), output_width)
    j_output: cp.ndarray = stride * cp.tile(cp.arange(output_width), output_height)

    i: cp.ndarray = i_offset.reshape(-1, 1) + i_output.reshape(1, -1)
    j: cp.ndarray = j_offset.reshape(-1, 1) + j_output.reshape(1, -1)
    k: cp.ndarray = cp.repeat(
        cp.arange(num_channels),
        kernel_height * kernel_width
    ).reshape(-1, 1)

    return k, i, j, output_height, output_width


def im2col(
    input: cp.ndarray,
    kernel_height: int,
    kernel_width: int,
    padding: int,
    stride: int
) -> Tuple[cp.ndarray, int, int]:
    """
    Convert image batch to column matrix using the im2col algorithm.

    Args:
        input: Input batch of shape (batch_size, num_channels, height, width)
        kernel_height: Height of the extraction kernel
        kernel_width: Width of the extraction kernel
        padding: Zero-padding applied around the input
        stride: Stride between adjacent windows

    Returns:
        Tuple containing the flattened patches and output spatial dimensions
    """
    x_padded: cp.ndarray = cp.pad(
        input,
        ((0, 0), (0, 0), (padding, padding), (padding, padding))
    )

    k: cp.ndarray
    i: cp.ndarray
    j: cp.ndarray
    output_height: int
    output_width: int
    k, i, j, output_height, output_width = get_im2col_indices(
        input_shape=input.shape,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        padding=padding,
        stride=stride
    )

    cols: cp.ndarray = x_padded[:, k, i, j]

    num_channels: int = input.shape[1]
    cols = cols.transpose(0, 2, 1).reshape(
        -1,
        num_channels * kernel_height * kernel_width
    )

    return cols, output_height, output_width


def col2im(
    input_cols_grad: cp.ndarray,
    input_shape: Tuple[int, int, int, int],
    kernel_height: int,
    kernel_width: int,
    padding: int,
    stride: int
) -> cp.ndarray:
    """
    Reconstruct the input gradient from column gradients.

    Args:
        input_cols_grad: Gradient in im2col representation
        input_shape: Original input tensor shape as (batch_size, num_channels, height, width)
        kernel_height: Height of the extraction kernel
        kernel_width: Width of the extraction kernel
        padding: Zero-padding applied around the input
        stride: Stride between adjacent windows

    Returns:
        Input gradient tensor matching the original input shape
    """
    num_samples: int
    num_channels: int
    img_height: int
    img_width: int
    num_samples, num_channels, img_height, img_width = input_shape

    k: cp.ndarray
    i: cp.ndarray
    j: cp.ndarray
    output_height: int
    output_width: int
    k, i, j, output_height, output_width = get_im2col_indices(
        input_shape=input_shape,
        kernel_height=kernel_height,
        kernel_width=kernel_width,
        padding=padding,
        stride=stride
    )

    x_padded_grad: cp.ndarray = cp.zeros(
        (num_samples, num_channels, img_height + 2 * padding, img_width + 2 * padding),
        dtype=input_cols_grad.dtype
    )

    cols_reshaped: cp.ndarray = input_cols_grad.reshape(
        num_samples,
        output_height * output_width,
        -1
    ).transpose(0, 2, 1)

    sample_indices: cp.ndarray = cp.arange(num_samples).reshape(-1, 1, 1)
    channel_indices: cp.ndarray = k.reshape(1, -1, 1)
    row_indices: cp.ndarray = i.reshape(1, *i.shape)
    col_indices: cp.ndarray = j.reshape(1, *j.shape)

    cp.add.at(
        x_padded_grad,
        (sample_indices, channel_indices, row_indices, col_indices),
        cols_reshaped
    )

    if padding == 0:
        return x_padded_grad

    return x_padded_grad[
        :,
        :,
        padding:-padding,
        padding:-padding
    ]