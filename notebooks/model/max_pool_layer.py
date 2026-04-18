import cupy as cp
from typing import Tuple, Dict, Any

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

        output_height: int = (img_height - self.pool_height) // self.stride + 1
        output_width: int = (img_width - self.pool_width) // self.stride + 1

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
    
        output: cp.ndarray = patches.max(axis=2)
        output = output.reshape(num_samples, num_channels, output_height, output_width)
        
        return output
