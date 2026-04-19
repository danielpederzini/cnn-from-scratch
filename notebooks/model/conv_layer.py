import cupy as cp
from typing import Optional, Tuple, Dict, Any
from .utils.convolution_utils import im2col, col2im

class ConvLayer:
    """
    Convolutional layer for neural network.

    Implements a 2D convolution operation using the im2col (image-to-column)
    method for efficient computation. This base variant performs only the
    linear convolution; subclasses can add nonlinearities or skip connections.
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
        Initialize the convolutional layer.
        
        Args:
            num_filters: Number of filters (output channels)
            kernel_height: Height of the convolution kernel
            kernel_width: Width of the convolution kernel
            num_channels: Number of input channels
            padding: Zero-padding to apply to input
            stride: Stride of the convolution operation
        """
        self.num_filters: int = num_filters
        self.kernel_height: int = kernel_height
        self.kernel_width: int = kernel_width
        self.num_channels: int = num_channels
        self.padding: int = padding
        self.stride: int = stride
        
        fan_in: int = num_channels * kernel_height * kernel_width
        std: float = float(cp.sqrt(2.0 / fan_in).item())
        self.filters: cp.ndarray = cp.random.normal(
            0,
            std,
            size=(num_filters, num_channels, kernel_height, kernel_width)
        )
        self.biases: cp.ndarray = cp.zeros(shape=(num_filters,), dtype=cp.float32)
        self.last_input_shape: Optional[Tuple[int, int, int, int]] = None
        self.last_input_cols: Optional[cp.ndarray] = None
        self.last_linear_output: Optional[cp.ndarray] = None
        self.last_output_shape: Optional[Tuple[int, int]] = None
        self.w_grad: Optional[cp.ndarray] = None
        self.b_grad: Optional[cp.ndarray] = None
        self.training: bool = True

    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "ConvLayer":
        """
        Create a ConvLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'num_filters', 'kernel_height', 'kernel_width',
                       'num_channels', 'padding', and 'stride' keys
        
        Returns:
            Initialized ConvLayer instance
        """
        return ConvLayer(
            num_filters=definition.get("num_filters"),
            kernel_height=definition.get("kernel_height"),
            kernel_width=definition.get("kernel_width"),
            num_channels=definition.get("num_channels"),
            padding=definition.get("padding"),
            stride=definition.get("stride")
        )
    
    def describe(self) -> str:
        """
        Get a formatted description of this layer.
        
        Returns:
            String description of the layer with filter and parameter information
        """
        layer_type: str = type(self).__name__
        filters_shape: tuple = self.filters.shape
        biases_shape: tuple = self.biases.shape
        layer_params: int = self.parameter_count()
        
        return f"{layer_type}\n  Filters Shape: {filters_shape} | Biases Shape: {biases_shape}\n  Parameters: {layer_params:,}"

    def parameter_count(self) -> int:
        """
        Count this layer's trainable parameters.

        Returns:
            Total number of trainable parameters
        """
        return int(cp.prod(cp.array(self.filters.shape)).item() + self.biases.shape[0])


    def flatten_filters(self) -> cp.ndarray:
        """
        Flatten filters into a 2D matrix.
        
        Reshapes filters from shape (num_filters, 3, kernel_height, kernel_width)
        to (num_filters, 3 * kernel_height * kernel_width) for matrix multiplication.
        
        Returns:
            Flattened filters matrix of shape (num_filters, num_channels * kernel_height * kernel_width)
        """
        filter_columns: int = cp.prod(cp.array(self.filters.shape[-3:])).item()
        filters_matrix: cp.ndarray = cp.reshape(
            self.filters,
            newshape=(self.num_filters, filter_columns)
        )
        return filters_matrix

    def clip_grad(self, grad: cp.ndarray, clip_value: Optional[float] = None) -> cp.ndarray:
        """
        Optionally clip gradient using L2 norm.

        Args:
            grad: Gradient array to clip
            clip_value: Maximum allowed L2 norm for the gradient. If None,
                clipping is disabled.

        Returns:
            Gradient array, clipped only when a clip value is provided
        """
        if clip_value is None:
            return grad

        norm = cp.linalg.norm(grad)
        if norm > clip_value:
            grad = grad * (clip_value / norm)

        return grad

    def im2col(self, input: cp.ndarray) -> Tuple[cp.ndarray, int, int]:
        """
        Convert image batch to column matrix using im2col algorithm.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Tuple containing:
                - cols: Column matrix of shape (batch_size * output_height * output_width,
                         num_channels * kernel_height * kernel_width)
                - output_height: Height of the output feature map
                - output_width: Width of the output feature map
        """
        return im2col(
            input=input,
            kernel_height=self.kernel_height,
            kernel_width=self.kernel_width,
            padding=self.padding,
            stride=self.stride
        )

    def col2im(self, input_cols_grad: cp.ndarray, input_shape: Tuple[int, int, int, int]) -> cp.ndarray:
        """
        Reconstruct the input gradient from column gradients.

        Args:
            input_cols_grad: Gradient in im2col representation
            input_shape: Original input tensor shape as (batch_size, num_channels, height, width)

        Returns:
            Input gradient tensor matching the original input shape
        """
        return col2im(
            input_cols_grad=input_cols_grad,
            input_shape=input_shape,
            kernel_height=self.kernel_height,
            kernel_width=self.kernel_width,
            padding=self.padding,
            stride=self.stride
        )

    def convolve(self, input: cp.ndarray) -> cp.ndarray:
        """
        Compute the linear convolution output and cache intermediates.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Linear convolution output before activation
        """
        cols: cp.ndarray
        out_h: int
        out_w: int
        cols, out_h, out_w = self.im2col(input)
        self.last_input_shape = input.shape
        self.last_input_cols = cols
        self.last_output_shape = (out_h, out_w)

        output: cp.ndarray = cols @ self.flatten_filters().T
        num_samples: int = input.shape[0]
        output = output.reshape(num_samples, out_h, out_w, self.num_filters)
        output = output.transpose(0, 3, 1, 2)
        output += self.biases.reshape(1, self.num_filters, 1, 1)
        self.last_linear_output = output

        return output

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: compute the linear convolution.

        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)

        Returns:
            Output feature maps of shape (batch_size, num_filters, output_height, output_width)
        """
        return self.convolve(input)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: compute filter gradients and propagate input gradients.

        Args:
            output_error: Gradient from the next layer of shape
                (batch_size, num_filters, output_height, output_width)
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input of shape
            (batch_size, num_channels, input_height, input_width)
        """
        output_error_reshaped: cp.ndarray = output_error.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)

        filters_grad: cp.ndarray = output_error_reshaped.T @ self.last_input_cols / batch_size
        self.w_grad = self.clip_grad(filters_grad.reshape(self.filters.shape))
        self.b_grad = self.clip_grad(cp.mean(output_error, axis=(0, 2, 3)))

        input_cols_grad: cp.ndarray = output_error_reshaped @ self.flatten_filters()
        return self.col2im(input_cols_grad=input_cols_grad, input_shape=self.last_input_shape)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update this layer's filters using the stored gradients.

        Args:
            learning_rate: Learning rate for gradient descent update
        """
        if self.w_grad is not None:
            self.filters -= self.w_grad * learning_rate

        if self.b_grad is not None:
            self.biases -= self.b_grad * learning_rate

    def train(self) -> None:
        """
        Put the layer in training mode.
        """
        self.training = True

    def eval(self) -> None:
        """
        Put the layer in evaluation mode.
        """
        self.training = False

    def apply_weight_decay(self, learning_rate: float, weight_decay: float) -> None:
        """
        Apply decoupled weight decay to convolution filters.

        Args:
            learning_rate: Current optimizer learning rate
            weight_decay: Weight decay coefficient
        """
        self.filters *= (1.0 - learning_rate * weight_decay)
