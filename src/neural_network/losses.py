import numpy as np
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a perda entre os valores reais e preditos.
        """

    @abstractmethod
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente da perda em relação às predições.
        """


class MeanSquaredError(Loss):
    """
    Função de perda Mean Squared Error.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class CrossEntropyLoss(Loss):
    """
    Função de perda Cross-Entropy.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (-(y_true / y_pred) + (1 - y_true) / (1 - y_pred)) / y_true.size


class HingeLoss(Loss):
    """
    Função de perda Hinge para classificadores de margem.
    """

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.maximum(0, 1 - y_true * y_pred))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        grad = np.where(y_true * y_pred < 1, -y_true, 0)
        return grad / y_true.size
    
class CategoricalCrossEntropyLoss(Loss):
    """
    Implementação da Categorical Cross-Entropy Loss.
    Usada para problemas de classificação multiclasse com softmax.
    """
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calcula a perda entre os valores reais e preditos.
        
        Args:
            y_true: One-hot encoded ground truth labels
            y_pred: Predicted probabilities (após softmax)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente da perda.
        Quando usado com softmax, simplifica para (y_pred - y_true).
        
        Args:
            y_true: One-hot encoded ground truth labels
            y_pred: Predicted probabilities (após softmax)
        """
        return (y_pred - y_true) / y_true.shape[0]
