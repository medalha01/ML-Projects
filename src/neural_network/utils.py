# neural_network/utils.py

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd


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
    rng = np.random.default_rng(100)
    if method == "he":
        return rng.standard_normal((input_size, output_size)) * np.sqrt(2 / input_size)
    elif method == "xavier":
        return rng.standard_normal((input_size, output_size)) * np.sqrt(1 / input_size)
    elif method == "random":
        return rng.uniform(-1, 1, (input_size, output_size))
    else:
        raise ValueError(f"Método de inicialização '{method}' não é suportado.")


def orthogonal_init(input_size: int, output_size: int) -> np.ndarray:

    rng = np.random.default_rng(100)
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


def plot_single_confusion_matrix(y_true, y_pred, labels=None, title="Matriz de Confusão"):
    """
    Gera e exibe a matriz de confusão para um único conjunto de dados.

    Args:
        y_true (np.ndarray): Valores reais.
        y_pred (np.ndarray): Valores previstos.
        labels (list): Lista de rótulos de classe, opcional.
        title (str): Título para a matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred, normalize='true')  # Normalização por linha
    fig, ax = plt.subplots(figsize=(7, 6))  # Ajustar o tamanho da figura

    # Plotar a matriz de confusão
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=ax, colorbar=False)

    # Adicionar valores e rótulos em cada célula
    for (i, j), value in np.ndenumerate(cm):
        if i == j:
            cell_label = "VP"  # Verdadeiros Positivos
        elif i > j:
            cell_label = "FN"  # Falsos Negativos
        else:
            cell_label = "FP"  # Falsos Positivos

        ax.text(
            j, i, f"\n({cell_label})",
            ha="center", va="center", fontsize=10, color="black"
        )

    # Adicionar título e ajustar layout
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrices_side_by_side(y_train_true, y_train_pred, y_val_true, y_val_pred, labels=None):
    """
    Plota as matrizes de confusão do treinamento e validação lado a lado.

    Args:
        y_train_true (np.ndarray): Valores verdadeiros do conjunto de treinamento.
        y_train_pred (np.ndarray): Previsões do conjunto de treinamento.
        y_val_true (np.ndarray): Valores verdadeiros do conjunto de validação.
        y_val_pred (np.ndarray): Previsões do conjunto de validação.
        labels (list): Lista de rótulos de classe, opcional.
    """
    cm_train = confusion_matrix(y_train_true, y_train_pred, normalize='true')
    cm_val = confusion_matrix(y_val_true, y_val_pred, normalize='true')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Matriz de Confusão do TREINAMENTO
    disp_train = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=labels)
    disp_train.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=axes[0], colorbar=False)
    axes[0].set_title("Matriz de Confusão - TREINAMENTO")

    # Adicionar valores e rótulos em cada célula
    for (i, j), value in np.ndenumerate(cm_train):
        if i == j:
            cell_label = "VP"  # Verdadeiros Positivos
        elif i > j:
            cell_label = "FN"  # Falsos Negativos
        else:
            cell_label = "FP"  # Falsos Positivos

        axes[0].text(
            j, i, f"\n({cell_label})",
            ha="center", va="center", fontsize=10, color="black"
        )

    # Matriz de Confusão da VALIDAÇÃO
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=labels)
    disp_val.plot(cmap=plt.cm.Blues, xticks_rotation=45, ax=axes[1], colorbar=False)
    axes[1].set_title("Matriz de Confusão - VALIDAÇÃO")

    # Adicionar valores e rótulos em cada célula
    for (i, j), value in np.ndenumerate(cm_val):
        if i == j:
            cell_label = "VP"  # Verdadeiros Positivos
        elif i > j:
            cell_label = "FN"  # Falsos Negativos
        else:
            cell_label = "FP"  # Falsos Positivos

        axes[1].text(
            j, i, f"\n({cell_label})",
            ha="center", va="center", fontsize=10, color="black"
        )

    plt.tight_layout()
    plt.show()

def plot_loss_and_accuracy_curve(
    train_loss, val_loss, train_accuracy, val_accuracy
):
    """
    Plota curvas de perda e acurácia para treinamento e validação.

    Args:
        train_loss (list): Histórico de perdas do treinamento.
        val_loss (list): Histórico de perdas da validação.
        train_accuracy (list): Histórico de acurácia do treinamento.
        val_accuracy (list): Histórico de acurácia da validação.
    """

    plt.figure(figsize=(12, 5))

    # Subplot para perdas
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss: Training vs Validation")

    # Subplot para acurácia
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label="Train Accuracy")
    plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy: Training vs Validation")

    plt.tight_layout()
    plt.show()

def display_final_metrics(
    epoch_accuracy=None, epoch_loss=None, val_accuracy=None, val_loss=None,
    test_accuracy=None, test_precision=None, test_recall=None, test_f1=None,
    val_precision=None, val_recall=None, val_f1=None,
    model=None, optimizer=None, batch_size=None, elapsed_time=None
):
    """
    Exibe as métricas finais de treinamento, validação ou teste em formato tabular.
    Diferencia automaticamente entre métricas de treino/validação e métricas de teste.

    Args:
        epoch_accuracy (float): Acurácia final no conjunto de treino.
        epoch_loss (float): Perda final no conjunto de treino.
        val_accuracy (float): Acurácia final no conjunto de validação.
        val_loss (float): Perda final no conjunto de validação.
        test_accuracy (float): Acurácia final no conjunto de teste.
        test_precision (float): Precisão final no conjunto de teste.
        test_recall (float): Recall final no conjunto de teste.
        test_f1 (float): F1-score final no conjunto de teste.
        val_precision (float): Precisão final no conjunto de validação.
        val_recall (float): Recall final no conjunto de validação.
        val_f1 (float): F1-score final no conjunto de validação.
        model (object): Modelo treinado, contendo as camadas.
        optimizer (object): Otimizador usado, contendo a taxa de aprendizado.
        batch_size (int): Tamanho final do batch.
        elapsed_time (float): Tempo total de treinamento ou teste.
    """
    # Criar um dicionário para as métricas
    metrics_data = []
    if model is not None:
        metrics_data.append(("Camadas", len(model.layers)))
    if optimizer is not None:
        metrics_data.append(("Learning Rate", f"{optimizer.learning_rate:.6f}"))
    if batch_size is not None:
        metrics_data.append(("Batch Size Final", batch_size))
    if elapsed_time is not None:
        metrics_data.append(("Elapsed Time", f"{elapsed_time:.2f}s"))

    # Adicionar métricas de treino
    if epoch_accuracy is not None:
        metrics_data.append(("Train Accuracy", f"{epoch_accuracy:.4f}"))
        metrics_data.append(("Train Error", f"{1 - epoch_accuracy:.4f}"))
    if epoch_loss is not None:
        metrics_data.append(("Train Loss", f"{epoch_loss:.4f}"))

    # Adicionar métricas de validação
    if val_accuracy is not None:
        metrics_data.append(("Validation Accuracy", f"{val_accuracy:.4f}"))
        metrics_data.append(("Validation Error", f"{1 - val_accuracy:.4f}"))
    if val_loss is not None:
        metrics_data.append(("Validation Loss", f"{val_loss:.4f}"))
    if val_precision is not None:
        metrics_data.append(("Validation Precision", f"{val_precision:.4f}"))
    if val_recall is not None:
        metrics_data.append(("Validation Recall", f"{val_recall:.4f}"))
    if val_f1 is not None:
        metrics_data.append(("Validation F1-Score", f"{val_f1:.4f}"))

    # Adicionar métricas de teste
    if test_accuracy is not None:
        metrics_data.append(("Test Accuracy", f"{test_accuracy:.4f}"))
        metrics_data.append(("Test Error", f"{1 - test_accuracy:.4f}"))
    if test_precision is not None:
        metrics_data.append(("Test Precision", f"{test_precision:.4f}"))
    if test_recall is not None:
        metrics_data.append(("Test Recall", f"{test_recall:.4f}"))
    if test_f1 is not None:
        metrics_data.append(("Test F1-Score", f"{test_f1:.4f}"))

    # Converter para DataFrame
    metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])

    # Exibir como tabela
    print("\nMétricas Finais:")
    print(metrics_df.to_string(index=False))
    print_metrics_legend()


def print_metrics_legend():
    """
    Exibe a legenda explicativa das métricas usadas.
    """
    legend = """
    Legenda de Métricas:

    - Acurácia (Accuracy): Proporção de previsões corretas sobre o total de amostras.
      Fórmula: (VP + VN) / (VP + VN + FP + FN)
      
    - Precisão (Precision): Proporção de previsões corretas entre todas as previsões positivas feitas.
      Fórmula: VP / (VP + FP)
      Indica o quão confiável é o modelo quando prevê uma classe como positiva.

    - Sensibilidade (Recall): Proporção de positivos corretamente identificados pelo modelo.
      Fórmula: VP / (VP + FN)
      Mede a capacidade do modelo de encontrar todas as ocorrências da classe positiva.

    - F1-Score: Média harmônica entre Precisão e Sensibilidade.
      Fórmula: 2 * (Precision * Recall) / (Precision + Recall)
      Útil quando há um balanço entre Precisão e Sensibilidade necessário.

    - Erro (Error Rate): Proporção de previsões incorretas sobre o total de amostras.
      Fórmula: 1 - Acurácia
    """
    print(legend)
