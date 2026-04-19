import cupy as cp
from typing import Dict, Any, Optional
from .conv_layer import ConvLayer
from .utils.network_utils import NetworkUtils


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
        super().__init__(
            num_filters=num_filters,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            num_channels=num_channels,
            padding=padding,
            stride=stride
        )
        self.bn_gamma: cp.ndarray = cp.ones((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn_beta: cp.ndarray = cp.zeros((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn_cache: Optional[Dict[str, Any]] = None
        self.last_batch_norm_output: Optional[cp.ndarray] = None
        self.bn_gamma_grad: Optional[cp.ndarray] = None
        self.bn_beta_grad: Optional[cp.ndarray] = None
        self.bn_running_mean: cp.ndarray = cp.zeros((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn_running_var: cp.ndarray = cp.ones((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn_momentum: float = 0.1

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
        batch_norm_output: cp.ndarray
        batch_norm_output, self.bn_cache = NetworkUtils.batch_norm(
            input=linear_output,
            gamma=self.bn_gamma,
            beta=self.bn_beta,
            training=self.training,
            running_mean=self.bn_running_mean,
            running_var=self.bn_running_var,
            momentum=self.bn_momentum
        )
        self.last_batch_norm_output = batch_norm_output
        return cp.maximum(0, batch_norm_output)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass using the shared convolution-gradient logic from the parent class.

        Args:
            output_error: Gradient from the next layer
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input
        """
        relu_grad: cp.ndarray = output_error * (self.last_batch_norm_output > 0)
        batch_norm_input_grad: cp.ndarray
        gamma_grad: cp.ndarray
        beta_grad: cp.ndarray
        batch_norm_input_grad, gamma_grad, beta_grad = NetworkUtils.batch_norm_backward(
            output_error=relu_grad,
            cache=self.bn_cache
        )
        self.bn_gamma_grad = self.clip_grad(gamma_grad)
        self.bn_beta_grad = self.clip_grad(beta_grad)
        return super().backward(output_error=batch_norm_input_grad, batch_size=batch_size)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update convolution and batch-normalization parameters.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        super().update_parameters(learning_rate=learning_rate)

        if self.bn_gamma_grad is not None:
            self.bn_gamma -= self.bn_gamma_grad * learning_rate

        if self.bn_beta_grad is not None:
            self.bn_beta -= self.bn_beta_grad * learning_rate

    def train(self) -> None:
        """
        Put the layer in training mode.
        """
        super().train()

    def eval(self) -> None:
        """
        Put the layer in evaluation mode.
        """
        super().eval()

    def parameter_count(self) -> int:
        """
        Count convolution and batch-normalization parameters.

        Returns:
            Total number of trainable parameters
        """
        return super().parameter_count() + int(
            cp.prod(cp.array(self.bn_gamma.shape)).item()
            + cp.prod(cp.array(self.bn_beta.shape)).item()
        )