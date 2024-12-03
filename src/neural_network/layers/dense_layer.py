# neural_network/layers/dense_layer.py

from neural_network.layers.abstract_layer import Layer
import numpy as np
from typing import Dict


class DenseLayer(Layer):
    def __init__(self, input_size: int, output_size: int, initialization: str = "he", name: str = "Dense"):
        """
        Inicializa uma camada densa genérica.

        Args:
            input_size (int): Número de entradas.
            output_size (int): Número de saídas (neurônios).
            initialization (str): Método de inicialização dos pesos ('he', 'xavier', ou 'random').
            name (str): Nome da camada.
        """
        super().__init__(name)
        self.input_shape = (input_size,)
        self.output_shape = (output_size,)

        # Inicializar pesos e vieses
        self.weights = self.initialize_weights(input_size, output_size, initialization)
        self.biases = np.zeros((1, output_size))  # Biases inicializados como 0

    def initialize_weights(self, input_size: int, output_size: int, method: str) -> np.ndarray:
        """
        Inicializa os pesos da camada com base no método especificado.

        Args:
            input_size (int): Número de entradas.
            output_size (int): Número de saídas.
            method (str): Método de inicialização ('he', 'xavier', ou 'random').

        Returns:
            np.ndarray: Matriz de pesos inicializada.
        """
        if method == "he":
            return np.random.randn(input_size, output_size) * np.sqrt(2 / input_size)
        elif method == "xavier":
            return np.random.randn(input_size, output_size) * np.sqrt(1 / input_size)
        elif method == "random":
            return np.random.uniform(-1, 1, (input_size, output_size))
        else:
            raise ValueError(f"Método de inicialização '{method}' não é suportado.")

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realiza o forward pass da camada.

        Args:
            input_data (np.ndarray): Dados de entrada (shape: [batch_size, input_size]).

        Returns:
            np.ndarray: Saída da camada antes da ativação (shape: [batch_size, output_size]).
        """
        self.input_data = input_data  # Armazena entrada para o backward
        self.z = np.dot(input_data, self.weights) + self.biases  # Soma ponderada
        return self.z  # Saída sem ativação

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Realiza o backward pass da camada.

        Args:
            grad_output (np.ndarray): Gradiente vindo da próxima camada (shape: [batch_size, output_size]).

        Returns:
            np.ndarray: Gradiente para a camada anterior (shape: [batch_size, input_size]).
        """
        # Gradiente dos pesos, vieses e entrada
        self.grad_weights = np.dot(self.input_data.T, grad_output)
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        grad_input = np.dot(grad_output, self.weights.T)

        return grad_input

    def get_gradients(self) -> Dict[str, np.ndarray]:
        """
        Retorna os gradientes calculados para os pesos e vieses.

        Returns:
            dict: Gradientes dos pesos e vieses.
        """
        return {"weights": self.grad_weights, "biases": self.grad_biases}

    def apply_gradients(self, grad_weights: np.ndarray, grad_biases: np.ndarray, learning_rate: float):
        """
        Aplica gradientes para atualizar pesos e vieses.

        Args:
            grad_weights (np.ndarray): Gradientes dos pesos.
            grad_biases (np.ndarray): Gradientes dos vieses.
            learning_rate (float): Taxa de aprendizado.
        """
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

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
