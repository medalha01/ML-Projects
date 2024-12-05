from typing import Optional, Dict
import numpy as np
from .abstract_layer import Layer


class DropoutLayer(Layer):
    """
    Implementa uma camada de Dropout para redes neurais.
    Dropout é uma técnica de regularização que reduz o overfitting,
    desativando aleatoriamente frações dos neurônios durante o treinamento.
    """

    def __init__(self, rate: float, name: Optional[str] = None):
        """
        Inicializa a camada de Dropout.

        Args:
            rate (float): A fração de unidades a ser desativada (entre 0 e 1).
            name (Optional[str]): Nome opcional da camada.

        Raises:
            ValueError: Se o valor de `rate` não estiver no intervalo [0, 1].
        """
        super().__init__(name)
        if not 0 <= rate <= 1:
            raise ValueError("Dropout deve estar entre 0 e 1.")
        self.rate = rate
        self.mask: Optional[np.ndarray] = None
        self.training: bool = True

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Executa a propagação para frente (forward pass).

        Durante o treinamento, aplica uma máscara binária aleatória para desativar neurônios.
        Durante a avaliação, escala os valores de entrada de acordo com a taxa de dropout.

        Args:
            input_data (np.ndarray): Dados de entrada para a camada.

        Returns:
            np.ndarray: Dados de saída após aplicar dropout.
        """
        if self.training:
            rng = np.random.default_rng(100)
            self.mask = rng.binomial(1, 1 - self.rate, size=input_data.shape)

            output = input_data * self.mask
        else:
            output = input_data * (1 - self.rate)
        return output

    def backward(self, output_grad: np.ndarray) -> np.ndarray:
        """
        Executa a propagação para trás (backward pass).

        Multiplica o gradiente de saída pela máscara binária para calcular o gradiente de entrada.

        Args:
            output_grad (np.ndarray): Gradiente dos dados de saída.

        Returns:
            np.ndarray: Gradiente dos dados de entrada.
        """
        if self.training:
            input_grad = output_grad * self.mask
        else:
            input_grad = output_grad * (1 - self.rate)
        return input_grad

    def get_parameters(self) -> Dict[str, np.ndarray]:
        """
        Obtém os parâmetros treináveis da camada.

        Returns:
            Dict[str, np.ndarray]: Um dicionário vazio, pois Dropout não tem parâmetros treináveis.
        """
        return {}

    def set_parameters(self, parameters: Dict[str, np.ndarray]):
        """
        Define os parâmetros treináveis da camada.

        Args:
            parameters (Dict[str, np.ndarray]): Dicionário de parâmetros.

        Observação:
            A camada de Dropout não possui parâmetros, portanto esta função não faz nada.
        """

    def set_training(self, training: bool):
        """
        Define o estado de treinamento ou avaliação da camada.

        Args:
            training (bool): True para treinamento, False para avaliação.
        """
        self.training = training
