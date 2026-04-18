import cupy as cp
from typing import Dict, Any
from .layer import Layer

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
    
    @staticmethod
    def from_definition(definition: Dict[str, Any]) -> "SigmoidLayer":
        """
        Create a SigmoidLayer instance from a definition dictionary.
        
        Args:
            definition: Dictionary with 'input_size' and 'num_neurons' keys
            
        Returns:
            Initialized SigmoidLayer instance
        """
        input_size: int = definition.get("input_size")
        num_neurons: int = definition.get("num_neurons")
        
        weights: cp.ndarray = cp.random.normal(
            0,
            cp.sqrt(2.0 / input_size),
            size=(input_size, num_neurons)
        )
        biases: cp.ndarray = cp.zeros(shape=(num_neurons,))
        
        return SigmoidLayer(weights=weights, biases=biases)
    
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
