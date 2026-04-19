import cupy as cp
from typing import Optional, Dict, Any
from .conv_layer import ConvLayer
from .relu_conv_layer import ReluConvLayer
from .utils.network_utils import NetworkUtils


class ResBlock:
    """
    ResNet-style basic residual block.

    The block applies a `conv-bn-relu` main-path layer followed by a second
    `conv-bn` layer, adds an identity or projected shortcut, and then applies
    a final ReLU after the residual addition.
    """

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "ResBlock":
        """
        Create a ResBlock instance from a definition dictionary.

        Args:
            definition: Dictionary with convolution block configuration keys

        Returns:
            Initialized ResBlock instance
        """
        return ResBlock(
            num_filters=definition.get("num_filters"),
            num_channels=definition.get("num_channels"),
            stride=definition.get("stride", 1),
            kernel_height=definition.get("kernel_height", 3),
            kernel_width=definition.get("kernel_width", 3),
            padding=definition.get("padding", 1)
        )

    def __init__(
        self,
        num_filters: int,
        num_channels: int,
        stride: int = 1,
        kernel_height: int = 3,
        kernel_width: int = 3,
        padding: int = 1
    ) -> None:
        """
        Initialize a residual block.

        Args:
            num_filters: Number of output channels for the block
            num_channels: Number of input channels
            stride: Stride used by the first convolution and shortcut projection
            kernel_height: Height of both main-path convolution kernels
            kernel_width: Width of both main-path convolution kernels
            padding: Padding applied to both main-path convolutions
        """
        self.num_filters: int = num_filters
        self.num_channels: int = num_channels
        self.stride: int = stride
        self.kernel_height: int = kernel_height
        self.kernel_width: int = kernel_width
        self.padding: int = padding

        self.conv1: ReluConvLayer = ReluConvLayer(
            num_filters=num_filters,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            num_channels=num_channels,
            padding=padding,
            stride=stride
        )
        self.conv2: ConvLayer = ConvLayer(
            num_filters=num_filters,
            kernel_height=kernel_height,
            kernel_width=kernel_width,
            num_channels=num_filters,
            padding=padding,
            stride=1
        )

        self.bn2_gamma: cp.ndarray = cp.ones((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn2_beta: cp.ndarray = cp.zeros((1, num_filters, 1, 1), dtype=cp.float32)
        self.bn2_cache: Optional[Dict[str, Any]] = None
        self.bn2_gamma_grad: Optional[cp.ndarray] = None
        self.bn2_beta_grad: Optional[cp.ndarray] = None
        self.last_block_output: Optional[cp.ndarray] = None
        self.last_input_shape: Optional[tuple[int, int, int, int]] = None

        self.shortcut_projection: Optional[ConvLayer] = None
        self.shortcut_bn_gamma: Optional[cp.ndarray] = None
        self.shortcut_bn_beta: Optional[cp.ndarray] = None
        self.shortcut_bn_cache: Optional[Dict[str, Any]] = None
        self.shortcut_bn_gamma_grad: Optional[cp.ndarray] = None
        self.shortcut_bn_beta_grad: Optional[cp.ndarray] = None

        if stride != 1 or num_channels != num_filters:
            self.shortcut_projection = ConvLayer(
                num_filters=num_filters,
                kernel_height=1,
                kernel_width=1,
                num_channels=num_channels,
                padding=0,
                stride=stride
            )
            self.shortcut_bn_gamma = cp.ones((1, num_filters, 1, 1), dtype=cp.float32)
            self.shortcut_bn_beta = cp.zeros((1, num_filters, 1, 1), dtype=cp.float32)

    def describe(self) -> str:
        """
        Get a formatted description of this residual block.

        Returns:
            String description of the block structure and parameter information
        """
        layer_type: str = type(self).__name__
        shortcut_type: str = "projection" if self.shortcut_projection is not None else "identity"
        return (
            f"{layer_type}\n"
            f"  Main Path: conv-bn-relu -> conv-bn\n"
            f"  Shortcut: {shortcut_type}\n"
            f"  Channels: {self.num_channels} -> {self.num_filters} | Stride: {self.stride}\n"
            f"  Parameters: {self.parameter_count():,}"
        )

    def parameter_count(self) -> int:
        """
        Count all trainable parameters in the block.

        Returns:
            Total number of trainable parameters
        """
        layer_params: int = self.conv1.parameter_count() + self.conv2.parameter_count()
        layer_params += int(
            cp.prod(cp.array(self.bn2_gamma.shape)).item()
            + cp.prod(cp.array(self.bn2_beta.shape)).item()
        )

        if self.shortcut_projection is not None:
            layer_params += self.shortcut_projection.parameter_count()
            layer_params += int(
                cp.prod(cp.array(self.shortcut_bn_gamma.shape)).item()
                + cp.prod(cp.array(self.shortcut_bn_beta.shape)).item()
            )

        return layer_params

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass through the residual block.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Output tensor after residual addition and final ReLU
        """
        self.last_input_shape = input.shape

        main_path: cp.ndarray = self.conv1.forward(input)
        main_path = self.conv2.forward(main_path)
        main_path, self.bn2_cache = NetworkUtils.batch_norm(
            input=main_path,
            gamma=self.bn2_gamma,
            beta=self.bn2_beta
        )

        shortcut: cp.ndarray = input
        if self.shortcut_projection is not None:
            shortcut = self.shortcut_projection.forward(input)
            shortcut, self.shortcut_bn_cache = NetworkUtils.batch_norm(
                input=shortcut,
                gamma=self.shortcut_bn_gamma,
                beta=self.shortcut_bn_beta
            )
        else:
            self.shortcut_bn_cache = None

        block_output: cp.ndarray = main_path + shortcut
        self.last_block_output = block_output
        return cp.maximum(0, block_output)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass through the residual block.

        Args:
            output_error: Gradient from the next layer
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the block input
        """
        relu_grad: cp.ndarray = output_error * (self.last_block_output > 0)

        main_path_grad: cp.ndarray
        gamma_grad: cp.ndarray
        beta_grad: cp.ndarray
        main_path_grad, gamma_grad, beta_grad = NetworkUtils.batch_norm_backward(
            output_error=relu_grad,
            cache=self.bn2_cache
        )
        self.bn2_gamma_grad = self.conv2.clip_grad(gamma_grad)
        self.bn2_beta_grad = self.conv2.clip_grad(beta_grad)
        main_path_grad = self.conv2.backward(output_error=main_path_grad, batch_size=batch_size)
        main_input_grad: cp.ndarray = self.conv1.backward(output_error=main_path_grad, batch_size=batch_size)

        shortcut_grad: cp.ndarray = relu_grad
        if self.shortcut_projection is not None:
            shortcut_grad, shortcut_gamma_grad, shortcut_beta_grad = NetworkUtils.batch_norm_backward(
                output_error=shortcut_grad,
                cache=self.shortcut_bn_cache
            )
            self.shortcut_bn_gamma_grad = self.shortcut_projection.clip_grad(shortcut_gamma_grad)
            self.shortcut_bn_beta_grad = self.shortcut_projection.clip_grad(shortcut_beta_grad)
            shortcut_grad = self.shortcut_projection.backward(output_error=shortcut_grad, batch_size=batch_size)
        else:
            self.shortcut_bn_gamma_grad = None
            self.shortcut_bn_beta_grad = None

        return main_input_grad + shortcut_grad

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update all trainable parameters in the block.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        self.conv1.update_parameters(learning_rate=learning_rate)
        self.conv2.update_parameters(learning_rate=learning_rate)

        if self.bn2_gamma_grad is not None:
            self.bn2_gamma -= self.bn2_gamma_grad * learning_rate

        if self.bn2_beta_grad is not None:
            self.bn2_beta -= self.bn2_beta_grad * learning_rate

        if self.shortcut_projection is not None:
            self.shortcut_projection.update_parameters(learning_rate=learning_rate)

            if self.shortcut_bn_gamma_grad is not None:
                self.shortcut_bn_gamma -= self.shortcut_bn_gamma_grad * learning_rate

            if self.shortcut_bn_beta_grad is not None:
                self.shortcut_bn_beta -= self.shortcut_bn_beta_grad * learning_rate