# neural_network/layers/base.py

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional


class Layer(ABC):

    def __init__(self, name: Optional[str] = None):

        self.name = name
        self.input_shape: Optional[tuple] = None
        self.output_shape: Optional[tuple] = None

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Realiza o forward pass.
        """
        raise NotImplementedError()

    @abstractmethod
    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Realiza o backward pass.
        """

        raise NotImplementedError()

    def get_parameters(self) -> Dict[str, np.ndarray]:

        raise NotImplementedError()

    def set_parameters(self, parameters: Dict[str, np.ndarray]):

        raise NotImplementedError()
