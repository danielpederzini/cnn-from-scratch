import cupy as cp
from typing import Optional, Dict, Any


class FlattenLayer:
    """
    Flatten layer for neural network.
    
    Reshapes multi-dimensional input (typically from convolutional or pooling layers)
    into a 2D array suitable for fully connected layers. No learnable parameters.
    """
    
    def __init__(self) -> None:
        """
        Initialize the flatten layer.
        
        This layer has no parameters to initialize.
        """
        self.last_input_shape: Optional[tuple[int, ...]] = None
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "FlattenLayer":
        """
        Create a FlattenLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary (not used, but kept for consistency with other layers)
        
        Returns:
            Initialized FlattenLayer instance
        """
        return FlattenLayer()
    
    def describe(self) -> str:
        """
        Get a formatted description of this layer.
        
        Returns:
            String description of the layer
        """
        layer_type: str = type(self).__name__
        return f"{layer_type}\n  Parameters: 0"
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: flatten multi-dimensional input to 2D.
        
        Converts input of shape (batch_size, ...) to (batch_size, total_features)
        by reshaping the last dimensions while preserving the batch dimension.
        
        Args:
            input: Input array of arbitrary shape (batch_size, ...)
            
        Returns:
            Flattened output array of shape (batch_size, flattened_features)
        """
        self.last_input_shape = input.shape
        batch_size: int = input.shape[0]
        flattened_features: int = int(cp.prod(cp.array(input.shape[1:])).item())
        
        return input.reshape(batch_size, flattened_features)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: restore the original tensor shape before flattening.

        Args:
            output_error: Gradient from the next layer of shape
                (batch_size, flattened_features)
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input of shape
            (batch_size, ...)
        """
        del batch_size
        return output_error.reshape(self.last_input_shape)

    def update_parameters(self, learning_rate: float) -> None:
        """
        No-op update because this layer has no trainable parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        del learning_rate
