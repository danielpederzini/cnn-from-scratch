import cupy as cp
from typing import Optional, Tuple, Dict, Any

class MaxPoolLayer:
    """
    Max pooling layer for neural network.
    
    Implements 2D max pooling operation using the im2col-like approach for
    efficient computation. Reduces spatial dimensions while preserving the
    maximum values in each pooling window.
    """
    
    def __init__(
        self,
        pool_height: int,
        pool_width: int,
        stride: int
    ) -> None:
        """
        Initialize the max pooling layer.
        
        Args:
            pool_height: Height of the pooling window
            pool_width: Width of the pooling window
            stride: Stride of the pooling operation
        """
        self.pool_height: int = pool_height
        self.pool_width: int = pool_width
        self.stride: int = stride
        self.last_input_shape: Optional[Tuple[int, int, int, int]] = None
        self.last_max_indices: Optional[cp.ndarray] = None
        self.last_output_shape: Optional[Tuple[int, int]] = None

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "MaxPoolLayer":
        """
        Create a MaxPoolLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'pool_height', 'pool_width', and 'stride' keys
        
        Returns:
            Initialized MaxPoolLayer instance
        """
        return MaxPoolLayer(
            pool_height=definition.get("pool_height"),
            pool_width=definition.get("pool_width"),
            stride=definition.get("stride")
        )
    
    def describe(self) -> str:
        """
        Get a formatted description of this layer.
        
        Returns:
            String description of the layer with pool shape and stride information
        """
        layer_type: str = type(self).__name__
        pool_shape: Tuple[int, int] = (self.pool_height, self.pool_width)
        
        return f"{layer_type}\n  Pool Shape: {pool_shape} | Stride: {self.stride}"

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: compute max pooling operation.
        
        Uses im2col-like method to extract patches, computes maximum values,
        and reshapes output to standard 4D format.
        
        Args:
            x_batch: Input batch of shape (batch_size, num_channels, height, width)
            
        Returns:
            Output feature maps of shape (batch_size, num_channels, output_height, output_width)
        """
        num_samples: int
        num_channels: int
        img_height: int
        img_width: int
        num_samples, num_channels, img_height, img_width = input.shape
        self.last_input_shape = input.shape

        output_height: int = (img_height - self.pool_height) // self.stride + 1
        output_width: int = (img_width - self.pool_width) // self.stride + 1
        self.last_output_shape = (output_height, output_width)

        i_offset: cp.ndarray = cp.repeat(cp.arange(self.pool_height), self.pool_width)
        i_offset = cp.tile(i_offset, num_channels)

        j_offset: cp.ndarray = cp.tile(cp.arange(self.pool_width), self.pool_height)
        j_offset = cp.tile(j_offset, num_channels)

        i_output: cp.ndarray = self.stride * cp.repeat(cp.arange(output_height), output_width)
        j_output: cp.ndarray = self.stride * cp.tile(cp.arange(output_width), output_height)

        i: cp.ndarray = i_offset.reshape(-1, 1) + i_output.reshape(1, -1)
        j: cp.ndarray = j_offset.reshape(-1, 1) + j_output.reshape(1, -1)

        k: cp.ndarray = cp.repeat(cp.arange(num_channels), self.pool_height * self.pool_width)
        k = k.reshape(-1, 1)

        patches: cp.ndarray = input[:, k, i, j]

        patches = patches.reshape(
            num_samples,
            num_channels,
            self.pool_height * self.pool_width,
            output_height * output_width
        )

        self.last_max_indices = cp.argmax(patches, axis=2)
    
        output: cp.ndarray = patches.max(axis=2)
        output = output.reshape(num_samples, num_channels, output_height, output_width)
        
        return output
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: route gradients to the maximum element in each pooling window.

        Uses the argmax indices cached during the forward pass to scatter each
        upstream gradient value back to the input location that produced the
        pooled output.

        Args:
            output_error: Gradient from the next layer of shape
                (batch_size, num_channels, output_height, output_width)

        Returns:
            Gradient with respect to the input of shape
            (batch_size, num_channels, input_height, input_width)
        """
        num_samples: int
        num_channels: int
        img_height: int
        img_width: int
        num_samples, num_channels, img_height, img_width = self.last_input_shape
        output_height, output_width = self.last_output_shape

        input_error: cp.ndarray = cp.zeros(self.last_input_shape, dtype=output_error.dtype)
        output_error_flat: cp.ndarray = output_error.reshape(num_samples, num_channels, -1)

        max_rows: cp.ndarray = self.last_max_indices // self.pool_width
        max_cols: cp.ndarray = self.last_max_indices % self.pool_width

        output_positions: cp.ndarray = cp.arange(output_height * output_width)
        base_rows: cp.ndarray = (output_positions // output_width) * self.stride
        base_cols: cp.ndarray = (output_positions % output_width) * self.stride

        target_rows: cp.ndarray = base_rows.reshape(1, 1, -1) + max_rows
        target_cols: cp.ndarray = base_cols.reshape(1, 1, -1) + max_cols

        sample_indices: cp.ndarray = cp.arange(num_samples).reshape(-1, 1, 1)
        channel_indices: cp.ndarray = cp.arange(num_channels).reshape(1, -1, 1)

        cp.add.at(
            input_error,
            (sample_indices, channel_indices, target_rows, target_cols),
            output_error_flat
        )

        return input_error

    def update_parameters(self, learning_rate: float) -> None:
        """
        No-op update because this layer has no trainable parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        del learning_rate