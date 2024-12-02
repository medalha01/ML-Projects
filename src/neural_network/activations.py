# neural_network/activations.py

import numpy as np
from typing import Callable, Dict

# Registro para funções de ativação
activation_registry: Dict[str, Dict[str, Callable]] = {}


def register_activation(name: str, func: Callable, derivative: Callable) -> None:
    """
    Registra uma função de ativação e sua derivada.

    Args:
        name (str): Nome da função de ativação.
        func (Callable): Função de ativação.
        derivative (Callable): Derivada da função de ativação.

    Raises:
        ValueError: Se a função já estiver registrada.
    """
    name = name.lower()
    if name in activation_registry:
        raise ValueError(f"A função de ativação '{name}' já está registrada.")
    activation_registry[name] = {"function": func, "derivative": derivative}


def get_activation(name: str) -> Dict[str, Callable]:
    """
    Obtém a função de ativação e sua derivada pelo nome.

    Args:
        name (str): Nome da função de ativação.

    Returns:
        Dict[str, Callable]: Um dicionário contendo 'function' e 'derivative'.

    Raises:
        ValueError: Se a função de ativação não estiver registrada.
    """
    activation = activation_registry.get(name.lower())
    if activation is None:
        raise ValueError(f"A função de ativação '{name}' não está registrada.")
    return activation


# Funções de ativação e suas derivadas
def relu(z: np.ndarray) -> np.ndarray:
    """
    Função de ativação ReLU.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Saída após aplicar ReLU.
    """
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivada da função de ativação ReLU.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Derivada de ReLU.
    """
    return (z > 0).astype(float)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Função de ativação Sigmoid.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Saída após aplicar Sigmoid.
    """
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivada da função de ativação Sigmoid.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Derivada de Sigmoid.
    """
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z: np.ndarray) -> np.ndarray:
    """
    Função de ativação Tanh.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Saída após aplicar Tanh.
    """
    return np.tanh(z)


def tanh_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivada da função de ativação Tanh.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Derivada de Tanh.
    """
    return 1 - np.tanh(z) ** 2


def leaky_relu(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Função de ativação Leaky ReLU.

    Args:
        z (np.ndarray): Entrada.
        alpha (float): Inclinação para entradas negativas (padrão é 0.01).

    Returns:
        np.ndarray: Saída após aplicar Leaky ReLU.
    """
    return np.where(z > 0, z, alpha * z)


def leaky_relu_derivative(z: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """
    Derivada da função de ativação Leaky ReLU.

    Args:
        z (np.ndarray): Entrada.
        alpha (float): Inclinação para entradas negativas (padrão é 0.01).

    Returns:
        np.ndarray: Derivada de Leaky ReLU.
    """
    dz = np.ones_like(z)
    dz[z < 0] = alpha
    return dz


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Função de ativação Softmax.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Saída após aplicar Softmax.
    """
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))  # Para estabilidade numérica
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def softmax_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivada da função de ativação Softmax.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Derivada de Softmax.
    """
    s = softmax(z)
    return s * (1 - s)


def linear(z: np.ndarray) -> np.ndarray:
    """
    Função de ativação Linear (identidade).

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Saída idêntica à entrada.
    """
    return z


def linear_derivative(z: np.ndarray) -> np.ndarray:
    """
    Derivada da função de ativação Linear.

    Args:
        z (np.ndarray): Entrada.

    Returns:
        np.ndarray: Derivada constante (1).
    """
    return np.ones_like(z)


def elu(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Função de ativação ELU (Exponential Linear Unit).

    Args:
        z (np.ndarray): Entrada.
        alpha (float): Constante para ajustar o valor quando z < 0 (padrão é 1.0).

    Returns:
        np.ndarray: Saída após aplicar ELU.
    """
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))


def elu_derivative(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Derivada da função de ativação ELU.

    Args:
        z (np.ndarray): Entrada.
        alpha (float): Constante para ajustar o valor quando z < 0 (padrão é 1.0).

    Returns:
        np.ndarray: Derivada de ELU.
    """
    dz = np.where(z > 0, 1, alpha * np.exp(z))
    return dz


# Registra as funções de ativação
def initialize_activation_registry() -> None:
    """
    Inicializa e registra todas as funções de ativação disponíveis.
    """
    register_activation("relu", relu, relu_derivative)
    register_activation("sigmoid", sigmoid, sigmoid_derivative)
    register_activation("tanh", tanh, tanh_derivative)
    register_activation("leaky_relu", leaky_relu, leaky_relu_derivative)
    register_activation("softmax", softmax, softmax_derivative)
    register_activation("linear", linear, linear_derivative)
    register_activation("elu", elu, elu_derivative)


# Inicializa o registro ao importar o módulo
initialize_activation_registry()
