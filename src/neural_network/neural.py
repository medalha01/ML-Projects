class NeuralNetwork:
    def __init__(self, layers, loss, optimizer):
        self.layers = layers
        self.loss = loss
        self.optimizer = optimizer

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true, y_pred):
        grad = self.loss.backward(y_true, y_pred)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self):
        for layer in self.layers:
            params = layer.get_parameters()
            if params:
                grads = layer.get_gradients()
                weights_updated, biases_updated = self.optimizer.update(
                    params["weights"],
                    params["biases"],
                    grads["weights"],
                    grads["biases"],
                )
                layer.set_parameters(
                    {"weights": weights_updated, "biases": biases_updated}
                )
