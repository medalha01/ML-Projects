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
    
class CategoricalCrossEntropyLoss:
    """
    Implementação da Categorical Cross-Entropy Loss.
    """
    
    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        # Evita log(0) usando um pequeno epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        return np.mean(loss)

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # Evita divisões por zero
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        grad = -y_true / y_pred
        return grad / y_true.shape[0]
