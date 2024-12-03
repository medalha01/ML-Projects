# neural_network/utils.py


import numpy as np


def initialize_weights(
    input_size: int, output_size: int, method: str = "he"
) -> np.ndarray:
    """
    Inicializa os pesos com base no método especificado.

    Args:
        input_size (int): Número de entradas.
        output_size (int): Número de saídas.
        method (str): Método de inicialização ('he', 'xavier', ou 'random').

    Returns:
        np.ndarray: Matriz de pesos inicializada.
    """
    rng = np.random.default_rng(8989898989)
    if method == "he":
        return rng.standard_normal((input_size, output_size)) * np.sqrt(2 / input_size)
    elif method == "xavier":
        return rng.standard_normal((input_size, output_size)) * np.sqrt(1 / input_size)
    elif method == "random":
        return rng.uniform(-1, 1, (input_size, output_size))
    else:
        raise ValueError(f"Método de inicialização '{method}' não é suportado.")


def orthogonal_init(input_size: int, output_size: int) -> np.ndarray:

    rng = np.random.default_rng(8989898989)
    a = rng.standard_normal((input_size, output_size))
    u, _, v = np.linalg.svd(a, full_matrices=False)
    return u if u.shape == (input_size, output_size) else v


def normalize_data(x: np.ndarray) -> np.ndarray:
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-12  ##zero
    return (x - mean) / std


def min_max_scale(x: np.ndarray, feature_range: tuple = (0, 1)) -> np.ndarray:

    min_val, max_val = feature_range
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    return (x - x_min) / (x_max - x_min + 1e-12) * (max_val - min_val) + min_val

def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Realiza o one-hot encoding em um array de rótulos categóricos.

    """

    # Verifica se os rótulos estão no formato correto (inteiros de 0 a num_classes-1)
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Os rótulos devem ser inteiros para o one-hot encoding.")

    # Cria a matriz de zeros
    one_hot_matrix = np.zeros((labels.size, num_classes))

    # Atribui 1 na posição correspondente ao rótulo
    one_hot_matrix[np.arange(labels.size), labels] = 1

    return one_hot_matrix
