import cupy as cp
from typing import Dict, Any
from .conv_layer import ConvLayer

class ReluConvLayer(ConvLayer):
    """
    Convolutional layer with ReLU activation.

    Applies a linear convolution followed by ReLU.
    """
    def __init__(
            self,
            num_filters: int,
            kernel_height: int,
            kernel_width: int,
            num_channels: int,
            padding: int,
            stride: int
    ) -> None:
        """
        Initialize the ReLU convolutional layer.

        Args:
            num_filters: Number of convolution filters (output channels)
            kernel_height: Height of the convolution kernel
            kernel_width: Width of the convolution kernel
            num_channels: Number of input channels
            padding: Zero-padding applied around the input
            stride: Stride between adjacent convolution windows
        """
        super().__init__(
            num_filters=num_filters,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            num_channels=num_channels,
            padding=padding,
            stride=stride
        )

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "ReluConvLayer":
        """
        Create a ReluConvLayer instance from a definition dictionary.

        Args:
            definition: Dictionary with convolution layer configuration keys

        Returns:
            Initialized ReluConvLayer instance
        """
        return ReluConvLayer(
            num_filters=definition.get("num_filters"),
            kernel_height=definition.get("kernel_height"),
            kernel_width=definition.get("kernel_width"),
            num_channels=definition.get("num_channels"),
            padding=definition.get("padding"),
            stride=definition.get("stride")
        )

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass using the shared convolution logic from the parent class.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            ReLU-activated convolution output
        """
        linear_output: cp.ndarray = super().forward(input=input)
        return cp.maximum(0, linear_output)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass using the shared convolution-gradient logic from the parent class.

        Args:
            output_error: Gradient from the next layer
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input
        """
        relu_grad: cp.ndarray = output_error * (self.last_linear_output > 0)
        return super().backward(output_error=relu_grad, batch_size=batch_size)

    def update_parameters(self, learning_rate: float, weight_decay_lambda: float = 0.0) -> None:
        """
        Update convolution parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
            weight_decay_lambda: Regularization parameter for weight decay
        """
        super().update_parameters(learning_rate=learning_rate, weight_decay_lambda=weight_decay_lambda)

    def parameter_count(self) -> int:
        """
        Count convolution parameters.

        Returns:
            Total number of trainable parameters
        """
        return super().parameter_count()