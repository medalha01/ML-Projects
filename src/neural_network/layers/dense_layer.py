from typing import Optional, Dict
from .abstract_layer import Layer
from ..utils import initialize_weights
from ..activations import get_activation
import numpy as np


class DenseLayer(Layer):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Optional[str] = None,
        initialization: str = "he",
        name: Optional[str] = None,
    ):
        """
        Inicializa uma camada densa genérica.

        Args:
            input_size (int): Número de entradas.
            output_size (int): Número de saídas (neurônios).
            initialization (str): Método de inicialização dos pesos ('he', 'xavier', ou 'random').
            name (str): Nome da camada.
        """
        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size

        self.weights = initialize_weights(
            input_size, output_size, method=initialization
        )
        self.biases = np.zeros((1, output_size))

        if activation:
            activation_funcs = get_activation(activation)
            self.activation_func = activation_funcs["function"]
            self.activation_derivative = activation_funcs["derivative"]
        else:
            self.activation_func = lambda x: x
            self.activation_derivative = lambda x: np.ones_like(x)

        self.input_data: Optional[np.ndarray] = None
        self.linear_output: Optional[np.ndarray] = None

        self.grad_weights: Optional[np.ndarray] = None
        self.grad_biases: Optional[np.ndarray] = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realiza o forward pass da camada.

        Args:
            input_data (np.ndarray): Dados de entrada (shape: [batch_size, input_size]).

        Returns:
            np.ndarray: Saída da camada após a ativação (shape: [batch_size, output_size]).
        """
        self.input_data = input_data
        self.linear_output = np.dot(input_data, self.weights) + self.biases
        return self.activation_func(self.linear_output)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza o backward pass da camada.

        Args:
            grad_output (np.ndarray): Gradiente vindo da próxima camada (shape: [batch_size, output_size]).

        Returns:
            np.ndarray: Gradiente para a camada anterior (shape: [batch_size, input_size]).
        """
        activation_grad = self.activation_derivative(self.linear_output)
        delta = grad_output * activation_grad

        self.grad_weights = np.dot(self.input_data.T, delta)
        self.grad_biases = np.sum(delta, axis=0, keepdims=True)
        return np.dot(delta, self.weights.T)

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """
        Retorna os gradientes calculados para os pesos e vieses.

        Returns:
            dict: Gradientes dos pesos e vieses.
        """
        return {"weights": self.grad_weights, "biases": self.grad_biases}

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Retorna os parâmetros da camada.

        Returns:
            dict: Pesos e vieses.
        """
        return {"weights": self.weights, "biases": self.biases}

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        Configura os parâmetros da camada.

        Args:
            parameters (dict): Pesos e vieses.
        """
        self.weights = parameters["weights"]
        self.biases = parameters["biases"]
