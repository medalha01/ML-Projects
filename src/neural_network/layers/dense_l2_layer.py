# src/neural_network/layers/regularized_fully_connected_layer.py

from typing import Optional, Dict
from .abstract_layer import Layer
from ..utils import initialize_weights
from ..activations import get_activation
import numpy as np
import logging

logger = logging.getLogger(__name__)


class L2DenseLayer(Layer):
    # Regularização L2 para punir pesos grandes

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        regularization_lambda: float = 0.01,
        initialization: str = "he",
        name: Optional[str] = None,
    ):

        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.regularization_lambda = regularization_lambda

        self.weights = initialize_weights(
            self.input_size, 
            self.output_size,
            method=initialization
        )
        self.biases = np.zeros((1, self.output_size))

        activation_funcs = get_activation(activation)
        self.activation_func = activation_funcs["function"]
        self.activation_derivative = activation_funcs["derivative"]

        self.input_data: Optional[np.ndarray] = None
        self.linear_output: Optional[np.ndarray] = None

        #
        self.grad_weights: Optional[np.ndarray] = None
        self.grad_biases: Optional[np.ndarray] = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:

        self.input_data = input_data
        self.linear_output = np.dot(input_data, self.weights) + self.biases
        activated_output = self.activation_func(self.linear_output)
        return activated_output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        activation_grad = self.activation_derivative(self.linear_output)
        delta = output_grad * activation_grad

        grad_weights = np.dot(self.input_data.T, delta)
        grad_biases = np.sum(delta, axis=0, keepdims=True)
        input_grad = np.dot(delta, self.weights.T)

        grad_weights += self.regularization_lambda * self.weights

        self.grad_weights = grad_weights
        self.grad_biases = grad_biases

        return input_grad

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """
        Retorna os gradientes calculados para os pesos e vieses.

        Returns:
            dict: Gradientes dos pesos e vieses.
        """
        return {"weights": self.grad_weights, "biases": self.grad_biases}

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {"weights": self.weights, "biases": self.biases}

    def set_parameters(self, parameters: Dict[str, np.ndarray]):

        self.weights = parameters["weights"]
        self.biases = parameters["biases"]

    def compute_regularization_loss(self) -> float:

        reg_loss = 0.5 * self.regularization_lambda * np.sum(self.weights**2)
        return reg_loss
