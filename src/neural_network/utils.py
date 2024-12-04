# neural_network/utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import KFold
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


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Gera e exibe a matriz de confusão.

    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
        labels (list): Lista de rótulos de classe, opcional.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalização por linha
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Matriz de Confusão (Percentagens)")
    plt.show()

def plot_loss_and_accuracy_curve(loss_history, total_loss_history, accuracy_history):
    """
    Gera gráficos de perdas e acurácia durante o treinamento.

    Args:
        loss_history (list): Histórico de perda de treino.
        total_loss_history (list): Histórico de perda total (com regularização).
        accuracy_history (list): Histórico de acurácia de treino.
    """
    plt.figure(figsize=(12, 5))

    # Subplot para perdas
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(loss_history) + 1), loss_history, label='Train Loss')
    plt.plot(range(1, len(total_loss_history) + 1), total_loss_history, label='Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Losses')

    # Subplot para acurácia
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(accuracy_history) + 1), accuracy_history, label='Train Accuracy', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training Accuracy')

    plt.tight_layout()
    plt.show()