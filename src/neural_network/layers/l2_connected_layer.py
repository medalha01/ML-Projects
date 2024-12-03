# src/neural_network/layers/regularized_fully_connected_layer.py

from typing import Optional, Dict
from .abstract_layer import Layer
from ..utils import initialize_weights
from ..activations import get_activation
import numpy as np
import logging

logger = logging.getLogger(__name__)


class L2DenseLayer(Layer):
    # Regularização L2 para punir pesos grandes
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "relu",
        l2_lambda: float = 0.01,
        name: Optional[str] = None,
    ):

        super().__init__(name)
        self.input_size = input_size
        self.output_size = output_size
        self.l2_lambda = l2_lambda

        self.weights = initialize_weights(self.input_size, self.output_size)
        self.biases = np.zeros((1, self.output_size))

        activation_funcs = get_activation(activation)
        self.activation_func = activation_funcs["function"]
        self.activation_derivative = activation_funcs["derivative"]

        self.input_data: Optional[np.ndarray] = None
        self.linear_output: Optional[np.ndarray] = None
