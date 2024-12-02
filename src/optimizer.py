import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class Optimizer(ABC):

    @abstractmethod
    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        grad_weights: np.ndarray,
        grad_biases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Atualiza os pesos e biases com base nos gradientes.
        """
        pass


class GradientDescent(Optimizer):

    def __init__(self, learning_rate: float = 0.01):

        self.learning_rate = learning_rate
        logger.debug(
            f"GradientDescentOptimizer inicializado com learning_rate={self.learning_rate}"
        )

    def update(
        self,
        weights: np.ndarray,
        biases: np.ndarray,
        grad_weights: np.ndarray,
        grad_biases: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        weights -= self.learning_rate * grad_weights
        biases -= self.learning_rate * grad_biases
        logger.debug("GradientDescentOptimizer atualizou pesos e biases.")
        return weights, biases
