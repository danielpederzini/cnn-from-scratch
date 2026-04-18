import cupy as cp
from typing import Optional, Tuple, Dict, Any

class ConvLayer:
    """
    Convolutional layer for neural network.
    
    Implements a 2D convolution operation using the im2col (image-to-column)
    method for efficient computation. Filters are randomly initialized.
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
        layer_params: int = int(cp.prod(cp.array(filters_shape)).item() + biases_shape[0])
        
        return f"{layer_type}\n  Filters Shape: {filters_shape} | Biases Shape: {biases_shape}\n  Parameters: {layer_params:,}"


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

    def get_im2col_indices(self, input_shape: Tuple[int, int, int, int]) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray, int, int]:
        """
        Compute shared indexing tensors for im2col and col2im.

        Args:
            input_shape: Input tensor shape as (batch_size, num_channels, height, width)

        Returns:
            Tuple containing channel and spatial index tensors along with output dimensions
        """
        _, num_channels, img_height, img_width = input_shape

        img_height_padded: int = img_height + 2 * self.padding
        img_width_padded: int = img_width + 2 * self.padding

        output_height: int = (img_height_padded - self.kernel_height) // self.stride + 1
        output_width: int = (img_width_padded - self.kernel_width) // self.stride + 1

        i_offset: cp.ndarray = cp.repeat(cp.arange(self.kernel_height), self.kernel_width)
        i_offset = cp.tile(i_offset, num_channels)

        j_offset: cp.ndarray = cp.tile(cp.arange(self.kernel_width), self.kernel_height)
        j_offset = cp.tile(j_offset, num_channels)

        i_output: cp.ndarray = self.stride * cp.repeat(cp.arange(output_height), output_width)
        j_output: cp.ndarray = self.stride * cp.tile(cp.arange(output_width), output_height)

        i: cp.ndarray = i_offset.reshape(-1, 1) + i_output.reshape(1, -1)
        j: cp.ndarray = j_offset.reshape(-1, 1) + j_output.reshape(1, -1)
        k: cp.ndarray = cp.repeat(
            cp.arange(num_channels),
            self.kernel_height * self.kernel_width
        ).reshape(-1, 1)

        return k, i, j, output_height, output_width
    
    def im2col(self, input: cp.ndarray) -> Tuple[cp.ndarray, int, int]:
        """
        Convert image batch to column matrix using im2col algorithm.
        
        Transforms image patches into columns for efficient convolution computation
        via matrix multiplication. Applies padding and extracts patches according
        to kernel size and stride.
        
        Args:
            x_batch: Input batch of shape (batch_size, num_channels, height, width)
            
        Returns:
            Tuple containing:
                - cols: Column matrix of shape (batch_size * output_height * output_width, 
                         num_channels * kernel_height * kernel_width)
                - output_height: Height of the output feature map
                - output_width: Width of the output feature map
        """
        x_padded: cp.ndarray = cp.pad(
            input,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        )

        k: cp.ndarray
        i: cp.ndarray
        j: cp.ndarray
        output_height: int
        output_width: int
        k, i, j, output_height, output_width = self.get_im2col_indices(input.shape)

        cols: cp.ndarray = x_padded[:, k, i, j]

        num_channels: int = input.shape[1]
        cols = cols.transpose(0, 2, 1).reshape(
            -1,
            num_channels * self.kernel_height * self.kernel_width
        )

        return cols, output_height, output_width

    def col2im(self, input_cols_grad: cp.ndarray, input_shape: Tuple[int, int, int, int]) -> cp.ndarray:
        """
        Reconstruct the input gradient from column gradients.

        Args:
            input_cols_grad: Gradient in im2col representation
            input_shape: Original input tensor shape as (batch_size, num_channels, height, width)

        Returns:
            Input gradient tensor matching the original input shape
        """
        num_samples: int
        num_channels: int
        img_height: int
        img_width: int
        num_samples, num_channels, img_height, img_width = input_shape

        k: cp.ndarray
        i: cp.ndarray
        j: cp.ndarray
        output_height: int
        output_width: int
        k, i, j, output_height, output_width = self.get_im2col_indices(input_shape)

        x_padded_grad: cp.ndarray = cp.zeros(
            (num_samples, num_channels, img_height + 2 * self.padding, img_width + 2 * self.padding),
            dtype=input_cols_grad.dtype
        )

        cols_reshaped: cp.ndarray = input_cols_grad.reshape(
            num_samples,
            output_height * output_width,
            -1
        ).transpose(0, 2, 1)

        sample_indices: cp.ndarray = cp.arange(num_samples).reshape(-1, 1, 1)
        channel_indices: cp.ndarray = k.reshape(1, -1, 1)
        row_indices: cp.ndarray = i.reshape(1, *i.shape)
        col_indices: cp.ndarray = j.reshape(1, *j.shape)

        cp.add.at(
            x_padded_grad,
            (sample_indices, channel_indices, row_indices, col_indices),
            cols_reshaped
        )

        if self.padding == 0:
            return x_padded_grad

        return x_padded_grad[
            :,
            :,
            self.padding:-self.padding,
            self.padding:-self.padding
        ]
    
    def relu(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply ReLU activation function.
        
        Args:
            input: Input array
            
        Returns:
            Activated output with negative values set to 0
        """
        return cp.maximum(0, input)

    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: compute convolution operation.
        
        Uses im2col method to convert input to columns, performs matrix multiplication
        with flattened filters, and reshapes output to standard 4D format.
        
        Args:
            input: Input batch of shape (batch_size, num_channels, height, width)
            
        Returns:
            Output feature maps of shape (batch_size, num_filters, output_height, output_width)
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
        
        return self.relu(output)

    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: apply ReLU derivative, compute filter gradients, and propagate input gradients.

        Args:
            output_error: Gradient from the next layer of shape
                (batch_size, num_filters, output_height, output_width)
            batch_size: Size of the batch for gradient averaging

        Returns:
            Gradient with respect to the input of shape
            (batch_size, num_channels, input_height, input_width)
        """
        relu_grad: cp.ndarray = output_error * (self.last_linear_output > 0)
        relu_grad_reshaped: cp.ndarray = relu_grad.transpose(0, 2, 3, 1).reshape(-1, self.num_filters)

        filters_grad: cp.ndarray = relu_grad_reshaped.T @ self.last_input_cols / batch_size
        self.w_grad = self.clip_grad(filters_grad.reshape(self.filters.shape))
        self.b_grad = self.clip_grad(cp.mean(relu_grad, axis=(0, 2, 3)))

        input_cols_grad: cp.ndarray = relu_grad_reshaped @ self.flatten_filters()
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
