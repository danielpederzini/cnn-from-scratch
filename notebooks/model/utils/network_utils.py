import cupy as cp
from typing import Optional, Dict, Any, Tuple


class NetworkUtils:
	"""
	Utility helpers shared across neural network components.
	"""

	@staticmethod
	def batch_norm(
		input: cp.ndarray,
		gamma: Optional[cp.ndarray] = None,
		beta: Optional[cp.ndarray] = None,
		epsilon: float = 1e-5
	) -> Tuple[cp.ndarray, Dict[str, Any]]:
		"""
		Apply batch normalization to dense or convolutional activations.

		For 2D tensors shaped (batch_size, features), normalization is applied
		per feature across the batch axis. For 4D tensors shaped
		(batch_size, channels, height, width), normalization is applied per
		channel across the batch and spatial axes.

		Args:
			input: Activation tensor of shape (N, F) or (N, C, H, W)
			gamma: Optional scale parameter. Must broadcast to the normalized
				statistics shape. Defaults to ones when omitted.
			beta: Optional shift parameter. Must broadcast to the normalized
				statistics shape. Defaults to zeros when omitted.
			epsilon: Small constant added to the variance for numerical stability

		Returns:
			Tuple containing:
				- normalized_output: Batch-normalized tensor with the same shape as the input
				- cache: Dictionary with normalization intermediates for reuse in other code
		"""
		reduction_axes: tuple[int, ...]
		parameter_shape: tuple[int, ...]

		if input.ndim == 2:
			reduction_axes = (0,)
			parameter_shape = (1, input.shape[1])
		else:
			reduction_axes = (0, 2, 3)
			parameter_shape = (1, input.shape[1], 1, 1)

		mean: cp.ndarray = cp.mean(input, axis=reduction_axes, keepdims=True)
		variance: cp.ndarray = cp.var(input, axis=reduction_axes, keepdims=True)
		inv_std: cp.ndarray = 1.0 / cp.sqrt(variance + epsilon)
		normalized_input: cp.ndarray = (input - mean) * inv_std

		if gamma is None:
			gamma = cp.ones(parameter_shape, dtype=input.dtype)
		else:
			gamma = gamma.reshape(parameter_shape)

		if beta is None:
			beta = cp.zeros(parameter_shape, dtype=input.dtype)
		else:
			beta = beta.reshape(parameter_shape)

		output: cp.ndarray = gamma * normalized_input + beta

		cache: Dict[str, Any] = {
			"input": input,
			"mean": mean,
			"variance": variance,
			"inv_std": inv_std,
			"normalized_input": normalized_input,
			"gamma": gamma,
			"beta": beta,
			"reduction_axes": reduction_axes,
			"parameter_shape": parameter_shape,
			"epsilon": epsilon,
		}

		return output, cache

	@staticmethod
	def batch_norm_backward(
		output_error: cp.ndarray,
		cache: Dict[str, Any]
	) -> Tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
		"""
		Backpropagate through batch normalization.

		Args:
			output_error: Gradient with respect to the batch-norm output
			cache: Cache returned by `batch_norm`

		Returns:
			Tuple containing gradients for:
				- input
				- gamma
				- beta
		"""
		input: cp.ndarray = cache["input"]
		normalized_input: cp.ndarray = cache["normalized_input"]
		gamma: cp.ndarray = cache["gamma"]
		inv_std: cp.ndarray = cache["inv_std"]
		reduction_axes: tuple[int, ...] = cache["reduction_axes"]

		if input.ndim == 2:
			normalized_count: int = input.shape[0]
		else:
			normalized_count = input.shape[0] * input.shape[2] * input.shape[3]

		gamma_grad: cp.ndarray = cp.sum(output_error * normalized_input, axis=reduction_axes, keepdims=True)
		beta_grad: cp.ndarray = cp.sum(output_error, axis=reduction_axes, keepdims=True)

		normalized_input_grad: cp.ndarray = output_error * gamma
		sum_normalized_input_grad: cp.ndarray = cp.sum(
			normalized_input_grad,
			axis=reduction_axes,
			keepdims=True
		)
		sum_normalized_product: cp.ndarray = cp.sum(
			normalized_input_grad * normalized_input,
			axis=reduction_axes,
			keepdims=True
		)

		input_grad: cp.ndarray = (
			inv_std / normalized_count
		) * (
			normalized_count * normalized_input_grad
			- sum_normalized_input_grad
			- normalized_input * sum_normalized_product
		)

		return input_grad, gamma_grad, beta_grad
