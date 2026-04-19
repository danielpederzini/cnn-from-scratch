import cupy as cp
from typing import Dict, Any
from .conv_layer import ConvLayer


class ReluConvLayer(ConvLayer):
    """
    Convolutional layer with ReLU activation.

    Applies a linear convolution followed by ReLU.
    """

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