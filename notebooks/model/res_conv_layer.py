import cupy as cp
from typing import Optional, Dict, Any
from .relu_conv_layer import ReluConvLayer
from .convolution_utils import im2col, col2im


class ResConvLayer(ReluConvLayer):
    """
    Residual convolutional layer with identity or projected skip connection.

    Applies a ReLU-activated convolution and adds the original input tensor
    through either an identity shortcut or a learned 1x1 projection when the
    input and output shapes differ.
    """

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "ResConvLayer":
        """
        Create a ResConvLayer instance from a definition dictionary.

        Args:
            definition: Dictionary with convolution layer configuration keys

        Returns:
            Initialized ResConvLayer instance
        """
        return ResConvLayer(
            num_filters=definition.get("num_filters"),
            kernel_height=definition.get("kernel_height"),
            kernel_width=definition.get("kernel_width"),
            num_channels=definition.get("num_channels"),
            padding=definition.get("padding"),
            stride=definition.get("stride")
        )

    def __init__(
        self,
        num_filters: int,
        kernel_height: int,
        kernel_width: int,
        num_channels: int,
        padding: int,
        stride: int
    ) -> None:
        super().__init__(
            num_filters=num_filters,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            num_channels=num_channels,
            padding=padding,
            stride=stride
        )
        self.last_residual_input: Optional[cp.ndarray] = None
        self.last_shortcut_cols: Optional[cp.ndarray] = None
        self.last_shortcut_input_shape: Optional[tuple[int, int, int, int]] = None
        self.use_projection: bool = False
        self.projection_filters: Optional[cp.ndarray] = None
        self.projection_biases: Optional[cp.ndarray] = None
        self.projection_w_grad: Optional[cp.ndarray] = None
        self.projection_b_grad: Optional[cp.ndarray] = None

    def initialize_projection(self) -> None:
        """
        Lazily initialize the 1x1 projection parameters.
        """
        if self.projection_filters is not None:
            return

        std: float = float(cp.sqrt(2.0 / self.num_channels).item())
        self.projection_filters = cp.random.normal(
            0,
            std,
            size=(self.num_filters, self.num_channels, 1, 1)
        )
        self.projection_biases = cp.zeros(shape=(self.num_filters,), dtype=cp.float32)

    def flatten_projection_filters(self) -> cp.ndarray:
        """
        Flatten the 1x1 projection filters for matrix multiplication.

        Returns:
            Projection weights of shape (num_filters, num_channels)
        """
        if self.projection_filters is None:
            raise ValueError("Projection filters are not initialized.")

        return self.projection_filters.reshape(self.num_filters, self.num_channels)

    def project_shortcut(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply the learned 1x1 projection on the shortcut path.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Projected shortcut tensor
        """
        self.initialize_projection()

        shortcut_cols: cp.ndarray
        out_h: int
        out_w: int
        shortcut_cols, out_h, out_w = im2col(
            input=input,
            kernel_height=1,
            kernel_width=1,
            padding=0,
            stride=self.stride
        )

        self.last_shortcut_cols = shortcut_cols
        self.last_shortcut_input_shape = input.shape

        shortcut_output: cp.ndarray = shortcut_cols @ self.flatten_projection_filters().T
        num_samples: int = input.shape[0]
        shortcut_output = shortcut_output.reshape(num_samples, out_h, out_w, self.num_filters)
        shortcut_output = shortcut_output.transpose(0, 3, 1, 2)
        shortcut_output += self.projection_biases.reshape(1, self.num_filters, 1, 1)

        return shortcut_output

    def describe(self) -> str:
        """
        Get a formatted description of this residual layer.

        Returns:
            String description of the layer with residual shortcut details
        """
        shortcut_type: str = "identity or 1x1 projection"
        return super().describe() + f"\n  Shortcut: {shortcut_type}"

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: ReLU convolution plus residual shortcut.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Residual output with the same shape as the convolution output
        """
        conv_output: cp.ndarray = super().forward(input)

        self.last_residual_input = input

        if conv_output.shape == input.shape:
            self.use_projection = False
            self.last_shortcut_cols = None
            self.last_shortcut_input_shape = None
            return conv_output + input

        self.use_projection = True
        projected_input: cp.ndarray = self.project_shortcut(input)

        return conv_output + projected_input

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: sum gradients from the convolution and shortcut paths.

        Args:
            output_error: Gradient from the next layer
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the residual input
        """
        conv_input_grad: cp.ndarray = super().backward(output_error=output_error, batch_size=batch_size)

        if not self.use_projection:
            self.projection_w_grad = None
            self.projection_b_grad = None
            return conv_input_grad + output_error

        shortcut_error: cp.ndarray = output_error.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)
        projection_filters_grad: cp.ndarray = shortcut_error.T @ self.last_shortcut_cols / batch_size
        self.projection_w_grad = self.clip_grad(projection_filters_grad.reshape(self.projection_filters.shape))
        self.projection_b_grad = self.clip_grad(cp.mean(output_error, axis=(0, 2, 3)))

        shortcut_input_cols_grad: cp.ndarray = shortcut_error @ self.flatten_projection_filters()
        shortcut_input_grad: cp.ndarray = col2im(
            input_cols_grad=shortcut_input_cols_grad,
            input_shape=self.last_shortcut_input_shape,
            kernel_height=1,
            kernel_width=1,
            padding=0,
            stride=self.stride
        )

        return conv_input_grad + shortcut_input_grad

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update both main-path and projection-path parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        super().update_parameters(learning_rate=learning_rate)

        if self.projection_w_grad is not None:
            self.projection_filters -= self.projection_w_grad * learning_rate

        if self.projection_b_grad is not None:
            self.projection_biases -= self.projection_b_grad * learning_rate