import cupy as cp
from typing import Optional

class Layer:
    """
    Base layer class for neural network.
    
    Implements forward and backward pass for a fully connected layer with
    optional activation function.
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize the layer with weights and biases.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        self.weights = weights
        self.biases = biases
        self.last_input: Optional[cp.ndarray] = None
        self.w_grad: Optional[cp.ndarray] = None
        self.b_grad: Optional[cp.ndarray] = None
        
    def clip_grad(self, grad: cp.ndarray, clip_value: float = 10.0) -> cp.ndarray:
        """
        Clip gradient to prevent explosion using L2 norm.
        
        Args:
            grad: Gradient array to clip
            clip_value: Maximum allowed L2 norm for the gradient
            
        Returns:
            Clipped gradient array
        """
        norm = cp.linalg.norm(grad)
        if norm > clip_value:
            grad = grad * (clip_value / norm)
            
        return grad
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: compute linear transformation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array of shape (batch_size, num_neurons)
        """
        self.last_input = input
        dot_product = input @ self.weights
        return dot_product + self.biases
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: compute gradients and propagate error.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            
        Returns:
            Error gradient to propagate to previous layer
        """
        w_grad = self.last_input.T @ output_error / batch_size
        self.w_grad = self.clip_grad(grad=w_grad)
        self.b_grad = self.clip_grad(grad=cp.mean(output_error, axis=0))
        
        return output_error @ self.weights.T


class ReluLayer(Layer):
    """
    Layer with ReLU activation function.
    
    Applies ReLU (Rectified Linear Unit) activation: f(x) = max(0, x)
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize ReLU layer.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        super().__init__(weights=weights, biases=biases)
        self.last_linear_output: Optional[cp.ndarray] = None

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
        Forward pass: linear transformation followed by ReLU activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Activated output array of shape (batch_size, num_neurons)
        """
        linear_output = super().forward(input=input)
        self.last_linear_output = linear_output
        return self.relu(input=linear_output)
    
    def backward(self, output_error: cp.ndarray, batch_size: int) -> cp.ndarray:
        """
        Backward pass: ReLU gradient followed by linear layer gradient.
        
        Args:
            output_error: Error gradient from the next layer
            batch_size: Size of the batch for gradient averaging
            
        Returns:
            Error gradient to propagate to previous layer
        """
        relu_grad = output_error * (self.last_linear_output > 0)
        input_error = super().backward(output_error=relu_grad, batch_size=batch_size)
        return input_error


class SigmoidLayer(Layer):
    """
    Layer with Sigmoid activation function.
    
    Applies Sigmoid activation: f(x) = 1 / (1 + exp(-x))
    Typically used for binary classification output layer.
    """
    
    def __init__(self, weights: cp.ndarray, biases: cp.ndarray) -> None:
        """
        Initialize Sigmoid layer.
        
        Args:
            weights: Weight matrix of shape (input_size, num_neurons)
            biases: Bias vector of shape (num_neurons,)
        """
        super().__init__(weights=weights, biases=biases)
    
    def sigmoid(self, input: cp.ndarray) -> cp.ndarray:
        """
        Apply Sigmoid activation function.
        
        Args:
            input: Input array
            
        Returns:
            Sigmoid output in range (0, 1)
        """
        return 1 / (1 + cp.exp(-input))
    
    def forward(self, input: cp.ndarray) -> cp.ndarray:
        """
        Forward pass: linear transformation followed by Sigmoid activation.
        
        Args:
            input: Input array of shape (batch_size, input_size)
            
        Returns:
            Activated output array of shape (batch_size, num_neurons)
        """
        linear_output = super().forward(input=input)
        return self.sigmoid(input=linear_output)


class Network:
    """
    Neural network composed of multiple layers.
    
    Implements forward pass, backward pass, and parameter updates for a
    fully connected neural network with various activation functions.
    """
    
    def __init__(self, layer_definitions: list[dict]) -> None:
        """
        Initialize network with layer definitions.
        
        Args:
            layer_definitions: List of dictionaries defining each layer.
                Each dict should contain: 'type' (layer type), 'input_size', 
                and 'num_neurons'.
        """
        self.layers: list[Layer] = self.initialize_layers(layer_definitions=layer_definitions)
    
    def initialize_layers(self, layer_definitions: list[dict]) -> list[Layer]:
        """
        Create layer instances based on definitions.
        
        Args:
            layer_definitions: List of layer configuration dictionaries
            
        Returns:
            List of initialized Layer objects
        """
        layers: list[Layer] = []
        
        for layer_definition in layer_definitions:
            input_size: int = layer_definition.get("input_size")
            num_neurons: int = layer_definition.get("num_neurons")

            weights: cp.ndarray = cp.random.normal(
                0, 
                cp.sqrt(2.0 / input_size), 
                size=(input_size, num_neurons)
            )
            biases: cp.ndarray = cp.zeros(shape=(num_neurons,))

            layer_type: Optional[str] = layer_definition.get("type")
            
            if layer_type == "ReLU":
                layers.append(ReluLayer(weights=weights, biases=biases))
            elif layer_type == "Sigmoid":
                layers.append(SigmoidLayer(weights=weights, biases=biases))
            else:
                layers.append(Layer(weights=weights, biases=biases))
        
        return layers
    
    def describe(self) -> str:
        """
        Print a formatted description of the network architecture.
        """
        description_lines: list[str] = [
            "=" * 80,
            "Network Architecture",
            "=" * 80,
        ]
        
        total_params: int = 0
        
        for layer_idx, layer in enumerate(self.layers, start=1):
            layer_type: str = type(layer).__name__
            weights_shape: tuple = layer.weights.shape
            biases_shape: tuple = layer.biases.shape
            layer_params: int = int(weights_shape[0] * weights_shape[1] + biases_shape[0])
            total_params += layer_params
            
            description_lines.append(
                f"\nLayer {layer_idx}: {layer_type}"
            )
            description_lines.append(
                f"  Weights Shape: {weights_shape} | Biases Shape: {biases_shape}"
            )
            description_lines.append(
                f"  Parameters: {layer_params:,}"
            )
        
        description_lines.append("\n" + "=" * 80)
        description_lines.append(f"Total Parameters: {total_params:,}")
        description_lines.append("=" * 80)
        
        print("\n".join(description_lines))

    def forward(self, input: cp.ndarray) -> list[cp.ndarray]:
        """
        Forward pass through all layers.
        
        Args:
            input: Input array of shape (batch_size, input_features)
            
        Returns:
            List of output arrays from each layer
        """
        outputs: list[cp.ndarray] = []
        
        for layer in self.layers:
            output = layer.forward(input=input)
            outputs.append(output)
            input = output
            
        return outputs

    def backward(self, output_error: cp.ndarray, batch_size: int) -> None:
        """
        Backward pass through all layers (reverse order).
        
        Computes gradients and accumulates error for parameter updates.
        
        Args:
            output_error: Error gradient from loss function
            batch_size: Size of the batch
        """
        accumulated_grad: cp.ndarray = output_error
        
        for layer in reversed(self.layers):
            accumulated_grad = layer.backward(output_error=accumulated_grad, batch_size=batch_size)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update all layer parameters using computed gradients.
        
        Args:
            learning_rate: Learning rate for gradient descent update
        """
        for layer in self.layers:
            layer.weights -= layer.w_grad * learning_rate
            layer.biases -= layer.b_grad * learning_rate