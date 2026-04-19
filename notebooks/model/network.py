from typing import Union
import cupy as cp
from .layer import Layer
from .relu_layer import ReluLayer
from .sigmoid_layer import SigmoidLayer
from .softmax_layer import SoftmaxLayer
from .conv_layer import ConvLayer
from .relu_conv_layer import ReluConvLayer
from .max_pool_layer import MaxPoolLayer
from .global_avg_pool_layer import GlobalAvgPoolLayer
from .flatten_layer import FlattenLayer

class Network:
    """
    Neural network composed of multiple layers.

    Orchestrates forward passes, backward passes, and parameter updates for
    networks built from dense, convolutional, pooling, and reshaping layers.
    """
    
    LAYER_TYPES: dict[str, type] = {
        "Conv": ConvLayer,
        "ReLUConv": ReluConvLayer,
        "MaxPool": MaxPoolLayer,
        "GlobalAvgPool": GlobalAvgPoolLayer,
        "Flatten": FlattenLayer,
        "ReLU": ReluLayer,
        "Sigmoid": SigmoidLayer,
        "Softmax": SoftmaxLayer,
    }
    
    def __init__(self, layer_definitions: list[dict]) -> None:
        """
        Initialize network with layer definitions.
        
        Args:
            layer_definitions: List of dictionaries defining each layer.
                For fully connected layers: 'type' (layer type), 'input_size', and 'num_neurons'.
                For convolutional layers: 'type': 'Conv' or 'ReLUConv', 'num_filters', 'kernel_height', 
                'kernel_width', 'num_channels', 'padding', and 'stride'.
                For max pooling layers: 'type': 'MaxPool', 'pool_height', 'pool_width', and 'stride'.
        """
        self.layers: list[Union[Layer, ConvLayer]] = self.initialize_layers(layer_definitions=layer_definitions)
    
    def initialize_layers(self, layer_definitions: list[dict]) -> list[Union[Layer, ConvLayer]]:
        """
        Create layer instances based on definitions.
        
        Each layer type is responsible for creating itself from its definition.
        
        Args:
            layer_definitions: List of layer configuration dictionaries
            
        Returns:
            List of initialized Layer or ConvLayer objects
        """
        layers: list[Union[Layer, ConvLayer]] = []
        
        for definition in layer_definitions:
            layer_type: str = definition.get("type", "Layer")
            layer_class: type = self.LAYER_TYPES.get(layer_type, Layer)
            layers.append(layer_class.from_definition(definition))
        
        return layers
    
    def describe(self) -> None:
        """
        Print a formatted description of the network architecture.
        
        Delegates description to each layer.
        """
        description_lines: list[str] = [
            "=" * 80,
            "Network Architecture",
            "=" * 80,
        ]
        
        total_params: int = 0
        
        for layer_idx, layer in enumerate(self.layers, start=1):
            layer_desc: str = layer.describe()
            layer_params: int = layer.parameter_count()
            
            total_params += layer_params
            
            description_lines.append(f"\nLayer {layer_idx}: {layer_desc}")
        
        description_lines.append("\n" + "=" * 80)
        description_lines.append(f"Total Parameters: {total_params:,}")
        description_lines.append("=" * 80)
        
        print("\n".join(description_lines))

    def forward(self, input) -> list:
        """
        Forward pass through all layers.
        
        Args:
            input: Input tensor expected by the first layer in the network
            
        Returns:
            List of output arrays from each layer
        """
        outputs: list = []
        
        for layer in self.layers:
            output = layer.forward(input=input)
            outputs.append(output)
            input = output
            
        return outputs

    def backward(self, output_error, batch_size: int) -> None:
        """
        Backward pass through all layers (reverse order).
        
        Computes gradients and accumulates error for parameter updates.
        
        Args:
            output_error: Error gradient from loss function
            batch_size: Size of the batch
        """
        gradients = [output_error]
        
        for layer in reversed(self.layers):
            gradient = layer.backward(output_error=gradients[-1], batch_size=batch_size)
            gradients.append(gradient)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update all layer parameters using computed gradients.
        
        Args:
            learning_rate: Learning rate for gradient descent update
        """
        for layer in self.layers:
            layer.update_parameters(learning_rate=learning_rate)

    def train(self) -> None:
        """
        Put the network in training mode.
        """
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()

    def eval(self) -> None:
        """
        Put the network in evaluation mode.
        """
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()

    def cce_loss(self, y_pred: cp.ndarray, y_true: cp.ndarray, epsilon=1e-15) -> cp.ndarray:
        """
        Compute categorical cross-entropy loss.

        Args:
            y_pred: Predicted class probabilities of shape (batch_size, num_classes)
            y_true: One-hot encoded target labels of shape (batch_size, num_classes)
            epsilon: Small value used to avoid taking the log of zero

        Returns:
            Scalar mean categorical cross-entropy loss
        """
        y_pred = cp.clip(y_pred, epsilon, 1. - epsilon)
        return -cp.mean(cp.sum(y_true * cp.log(y_pred), axis=1))