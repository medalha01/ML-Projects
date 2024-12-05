# neural_network/layers/dropout_layer.py

import numpy as np
from typing import Optional, Dict
from .abstract_layer import Layer


class DropoutLayer(Layer):

    def __init__(self, rate: float, name: Optional[str] = None):

        super().__init__(name)
        if not 0 <= rate <= 1:
            raise ValueError("Dropout deve estar entre 0 e 1.")
        self.rate = rate
        self.mask: Optional[np.ndarray] = None
        self.training: bool = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:

        if self.training:
            rng = np.random.default_rng(100)
            self.mask = rng.binomial(1, 1 - self.rate, size=input_data.shape)
            output = input_data * self.mask
        else:
            output = input_data * (1 - self.rate)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:

        if self.training:
            input_grad = output_grad * self.mask
        else:
            input_grad = output_grad * (1 - self.rate)
        return input_grad

    def get_parameters(self) -> Dict[str, np.ndarray]:
        return {}

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        "Dropout não tem parametros"

    def set_training(self, training: bool):
        self.training = training
