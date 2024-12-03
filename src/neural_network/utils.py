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
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
    plt.title("Matriz de Confusão")
    plt.show()

def plot_learning_curve_with_accuracy(model, X_train, y_train, test_accuracy, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)):
    """
    Gera e exibe a curva de aprendizado para um modelo customizado e inclui a acurácia de teste no título.

    Args:
        model: Modelo customizado com métodos `fit`, `forward`, e `update`.
        X_train (np.ndarray): Dados de entrada de treino.
        y_train (np.ndarray): Rótulos de treino.
        test_accuracy (float): Acurácia do teste para incluir no título.
        cv (int): Número de divisões para validação cruzada.
        train_sizes (np.ndarray): Proporções de dados de treino usadas.
    """
   

    train_scores = []
    val_scores = []

    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)

    for train_size in train_sizes:
        train_subset_scores = []
        val_subset_scores = []

        train_size = int(len(X_train) * train_size)
        X_partial = X_train[:train_size]
        y_partial = y_train[:train_size]

        for train_idx, val_idx in kfold.split(X_partial):
            X_fold_train, X_fold_val = X_partial[train_idx], X_partial[val_idx]
            y_fold_train, y_fold_val = y_partial[train_idx], y_partial[val_idx]

            # Treinar o modelo
            for epoch in range(10):  # Treinamos por algumas épocas para cada subdivisão
                y_pred = model.forward(X_fold_train)
                model.backward(y_fold_train, y_pred)
                model.update()

            # Avaliar no conjunto de treino
            y_train_pred = model.forward(X_fold_train)
            y_train_pred_class = np.argmax(y_train_pred, axis=1)
            y_train_true_class = np.argmax(y_fold_train, axis=1)
            train_subset_scores.append(accuracy_score(y_train_true_class, y_train_pred_class))

            # Avaliar no conjunto de validação
            y_val_pred = model.forward(X_fold_val)
            y_val_pred_class = np.argmax(y_val_pred, axis=1)
            y_val_true_class = np.argmax(y_fold_val, axis=1)
            val_subset_scores.append(accuracy_score(y_val_true_class, y_val_pred_class))

        train_scores.append(np.mean(train_subset_scores))
        val_scores.append(np.mean(val_subset_scores))

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores, 'o-', label="Treino", color="blue")
    plt.plot(train_sizes, val_scores, 'o-', label="Validação", color="orange")
    plt.title(f"Curva de Aprendizado (Test Accuracy: {test_accuracy:.4f})")
    plt.xlabel("Tamanho do Conjunto de Treino")
    plt.ylabel("Acurácia")
    plt.legend(loc="best")
    plt.grid()
    plt.show()