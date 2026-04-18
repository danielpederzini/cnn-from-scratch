import cupy as cp
from typing import Tuple, Dict, Any

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
        layer_params: int = int(cp.prod(cp.array(filters_shape)).item())
        
        return f"{layer_type}\n  Filters Shape: {filters_shape}\n  Parameters: {layer_params:,}"


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
        num_samples: int
        num_channels: int
        img_height: int
        img_width: int
        num_samples, num_channels, img_height, img_width = input.shape

        x_padded: cp.ndarray = cp.pad(
            input,
            ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding))
        )

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

        cols: cp.ndarray = x_padded[:, k, i, j]

        cols = cols.transpose(0, 2, 1).reshape(
            -1,
            num_channels * self.kernel_height * self.kernel_width
        )

        return cols, output_height, output_width
    
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
        
        output: cp.ndarray = cols @ self.flatten_filters().T
        num_samples: int = input.shape[0]
        output = output.reshape(num_samples, out_h, out_w, self.num_filters)
        output = output.transpose(0, 3, 1, 2)
        
        return self.relu(output)
