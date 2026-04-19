import cupy as cp
from typing import Optional, Tuple, Dict, Any


class GlobalAvgPoolLayer:
    """
    Global average pooling layer for neural network.

    Reduces each channel to a single value by averaging across the full
    spatial extent of the feature map.
    """

    def __init__(self) -> None:
        """
        Initialize the global average pooling layer.
        """
        self.last_input_shape: Optional[Tuple[int, int, int, int]] = None

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "GlobalAvgPoolLayer":
        """
        Create a GlobalAvgPoolLayer instance from a definition dictionary.

        Args:
            definition: Layer definition dictionary. No parameters are required.

        Returns:
            Initialized GlobalAvgPoolLayer instance
        """
        del definition
        return GlobalAvgPoolLayer()

    def describe(self) -> str:
        """
        Get a formatted description of this layer.

        Returns:
            String description of the layer
        """
        layer_type: str = type(self).__name__
        return f"{layer_type}\n  Output Shape: (batch_size, channels, 1, 1)"

    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters.

        Returns:
            Zero because global average pooling has no trainable parameters
        """
        return 0

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: average each channel over its spatial dimensions.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Output tensor of shape (batch_size, num_channels, 1, 1)
        """
        self.last_input_shape = input.shape
        return cp.mean(input, axis=(2, 3), keepdims=True)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: distribute channel gradients uniformly across the
        original spatial dimensions.

        Args:
            output_error: Gradient from the next layer of shape
                (batch_size, num_channels, 1, 1)
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input of shape
            (batch_size, num_channels, input_height, input_width)
        """
        del batch_size

        _, _, input_height, input_width = self.last_input_shape
        spatial_area: int = input_height * input_width
        return cp.broadcast_to(output_error / spatial_area, self.last_input_shape)

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float) -> None:
        """
        No-op update because this layer has no trainable parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Weight decay coefficient for regularization
        """
        del learning_rate, weight_decay_lambda