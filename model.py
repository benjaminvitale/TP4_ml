import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


class MLP(object):

    def __init__(self, layers=[15,10, 8,4,1], activations=["relu", "relu","relu","relu"], verbose=False, plot=False) -> None:
        """
        Initializes the MLP with specified layers, activations, and optional verbosity/plotting settings.
        Inputs:
            layers: List of integers representing the number of nodes in each layer.
            activations: List of activation functions for each layer.
            verbose: Boolean flag for logging output.
            plot: Boolean flag for plotting learning curves.
        """
        assert len(layers) == len(activations) + 1, "Number of layers and activations mismatch"
        self.layers = layers
        self.num_layers = len(layers)
        self.activations = activations
        self.verbose = verbose
        self.plot = plot

        # Initialize weights and biases randomly
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def forward_pass(self, x):
        """
        Performs forward propagation of input data through the MLP.
        Inputs:
            x: Features vector (input data).
        Returns:
            a: List of preactivations for each layer.
            z: List of activations for each layer.
        """
        z = [np.array(x).reshape(-1, 1)]  # Input activation (reshape to column vector)
        a = []  # To store preactivations

        for l in range(1, self.num_layers):
            a_l = np.dot(self.weights[l - 1], z[l - 1]) + self.biases[l - 1]
            a.append(np.copy(a_l))
            h = self.getActivationFunction(self.activations[l - 1])
            z_l = h(a_l)
            z.append(np.copy(z_l))

        return a, z
    
    def backward_pass(self, a, z, y):
        """
        Performs backward propagation to compute gradients of the loss with respect to weights and biases.
        Inputs:
            a: List of preactivations from forward pass.
            z: List of activations from forward pass.
            y: True target values.
        Returns:
            nabla_b: List of gradients for biases.
            nabla_w: List of gradients for weights.
            loss: Calculated loss value.
        """
        delta = [np.zeros(w.shape) for w in self.weights]
        h_prime = self.getDerivitiveActivationFunction(self.activations[-1])
        output = z[-1]
        delta[-1] = (output - y)  # Derivative of binary cross-entropy loss

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        nabla_b[-1] = delta[-1]
        nabla_w[-1] = np.dot(delta[-1], z[-2].T)

        for l in reversed(range(1, len(delta))):
            h_prime = self.getDerivitiveActivationFunction(self.activations[l - 1])
            delta[l - 1] = np.dot(self.weights[l].T, delta[l]) * h_prime(a[l - 1])
            nabla_b[l - 1] = delta[l - 1]
            nabla_w[l - 1] = np.dot(delta[l - 1], z[l - 1].T)

        # Binary cross-entropy loss
        loss = np.sum((output - y) ** 2) / (2 * y.shape[0])
        eps = 1e-9 # Add small constant 1e-9 to avoid log of zero
        #loss = -np.sum(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps)) / y.shape[0] 
        return nabla_b, nabla_w, loss

    def update_mini_batch(self, mini_batch, lr):
        """
        Updates model weights and biases using gradients computed from a mini-batch.
        Inputs:
            mini_batch: List of training samples (features and targets).
            lr: Learning rate for gradient updates.
        Returns:
            Average loss for the mini-batch.
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        total_loss = 0

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w, loss = self.backward_pass(*self.forward_pass(x), y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            total_loss += loss

        self.weights = [w - lr * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - lr * nb for b, nb in zip(self.biases, nabla_b)]
        return total_loss / len(mini_batch)

    def fit(self, training_data, epochs, mini_batch_size, lr, val_data=None, verbose=0):
        """
        Trains the MLP using the provided training data, with options for validation and verbosity.
        Inputs:
            training_data: List of tuples (features, targets) for training.
            epochs: Number of epochs to train.
            mini_batch_size: Number of samples per mini-batch.
            lr: Learning rate.
            val_data: Optional validation data for performance monitoring.
            verbose: Verbosity level for progress output.
        Returns:
            train_losses: List of training loss values per epoch.
            val_losses: List of validation loss values per epoch (if validation data is provided).
        """
        train_losses = []
        val_losses = []
        n = len(training_data)
        
        # Determine whether to use tqdm progress bar and detailed printout
        use_tqdm = verbose == 0 or verbose == 2
        print_detailed = verbose == 1 or verbose == 2
        progress_bar = tqdm(total=epochs, desc="Training Epochs") if use_tqdm else None

        for e in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[i:i + mini_batch_size] for i in range(0, n, mini_batch_size)]
            
            epoch_train_losses = []

            for mini_batch in mini_batches:
                train_loss = self.update_mini_batch(mini_batch, lr)
                epoch_train_losses.append(train_loss)

            avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
            train_losses.append(avg_train_loss)
            
            if val_data:
                val_loss = self.evaluate(val_data)
                val_losses.append(val_loss)
            
            if print_detailed:
                if val_data:
                    print(f"Epoch {e + 1}: Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                else:
                    print(f"Epoch {e + 1}: Train Loss: {avg_train_loss:.4f}")

            if use_tqdm:
                progress_bar.update(1)

        if use_tqdm:
            progress_bar.close()

        if self.plot: # Plot the training and validation loss curves
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            if val_losses:
                plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss Curves')
            plt.legend()
            plt.grid()
            plt.show()

        return train_losses, val_losses
    
    def evaluate(self, test_data):
        """
        Evaluates the model on a given test dataset.
        Inputs:
            test_data: List of tuples (features, targets) for evaluation.
        Returns:
            Average sum of squared errors on the test data.
        """
        sum_loss = 0
        for x, y in test_data:
            prediction = self.forward_pass(x)[-1][-1]
            # Compute sum of squared errors loss
            sum_loss += np.sum((prediction - y) ** 2) / 2
        return sum_loss / len(test_data)


    def predict(self, X):
        """
        Predicts output labels for input data.
        Inputs:
            X: Array-like input data for prediction.
        Returns:
            Predictions as a numpy array.
        """
        predictions = []
        for x in X:
            prediction = self.forward_pass(x)[-1][-1].flatten()
            predictions.append(prediction)
        return np.array(predictions)

    @staticmethod
    def getActivationFunction(name):
        """
        Returns the activation function based on the provided name.
        Inputs:
            name: String representing the activation function ('sigmoid' or 'relu').
        Returns:
            Activation function corresponding to the name.
        """
        if name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-x))
        elif name == 'relu':
            return lambda x: np.maximum(x, 0)
        else:
            print('Unknown activation function. Using linear by default.')
            return lambda x: x

    @staticmethod
    def getDerivitiveActivationFunction(name):
        """
        Returns the derivative of the activation function based on the provided name.
        Inputs:
            name: String representing the activation function ('sigmoid' or 'relu').
        Returns:
            Derivative of the activation function.
        """
        if name == 'sigmoid':
            sig = lambda x: 1 / (1 + np.exp(-x))
            return lambda x: sig(x) * (1 - sig(x))
        elif name == 'relu':
            def relu_diff(x):
                y = np.copy(x)
                y[y >= 0] = 1
                y[y < 0] = 0
                return y
            return relu_diff
        else:
            print('Unknown activation function. Using linear by default.')
            return lambda x: 1



import numpy as np

class NeuralNetwork:
    def __init__(self,layer_sizes):
        """
        Inicializa la red neuronal.
        layer_sizes: Lista con la cantidad de neuronas por capa (incluye la capa de entrada y la de salida).
        learning_rate: Tasa de aprendizaje para la optimización.
        """

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.learning_rate = None
        
        # Inicializar los pesos y sesgos
        self.weights = [np.random.randn(layer_sizes[i], layer_sizes[i-1]) for i in range(1, self.num_layers)]
        self.biases = [np.random.randn(layer_sizes[i], 1) for i in range(1, self.num_layers)]
        

    def relu(self, z):
        """Función de activación ReLU."""
        return np.maximum(0, z)

    def relu_derivative(self, z):
        """Derivada de la función de activación ReLU."""
        return np.where(z > 0, 1, 0)

    def forward(self, x):
        """
        Realiza la propagación hacia adelante.
        x: Vector de entrada (n_features, 1).
        Retorna las activaciones y los valores Z de cada capa.
        """
        activations = [x]
        zs = []  # Valores Z en cada capa

        for i in range(self.num_layers - 2):
            w = self.weights[i]
            b = self.biases[i]
            z = np.dot(w, activations[-1]) + b
            zs.append(z)
            a = self.relu(z)
            activations.append(a)
        
        # Para la capa de salida (última capa), aplicar función identidad
        w = self.weights[-1]
        b = self.biases[-1]
        z = np.dot(w, activations[-1]) + b
        zs.append(z)
        activations.append(z)  # Sin función de activación en la capa de salida
        
        return activations, zs

    def compute_loss(self, y_true, y_pred):
        """Calcula la pérdida de la red (suma de errores cuadráticos)."""
        return 0.5 * np.sum((y_true - y_pred) ** 2)

    def backward(self, x, y):
        """
        Realiza el backpropagation para calcular los gradientes.
        x: Vector de entrada (n_features, 1).
        y: Valor verdadero (output esperado).
        """
        # Propagación hacia adelante
        activations, zs = self.forward(x)

        # Inicializar los gradientes
        delta_weights = [np.zeros_like(w) for w in self.weights]
        delta_biases = [np.zeros_like(b) for b in self.biases]

        # Calcular el error de la capa de salida
        delta = (activations[-1] - y)  # Sin derivada de activación para la capa de salida

        # Gradiente de la última capa
        delta_weights[-1] = np.dot(delta, activations[-2].T)
        delta_biases[-1] = delta

        # Backpropagation para las capas ocultas
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.relu_derivative(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            delta_weights[-l] = np.dot(delta, activations[-l-1].T)
            delta_biases[-l] = delta

        return delta_weights, delta_biases

    def update_parameters(self, delta_weights, delta_biases):
        """
        Actualiza los parámetros (pesos y sesgos) de la red.
        delta_weights: Gradientes de los pesos.
        delta_biases: Gradientes de los sesgos.
        """
        self.weights = [w - self.learning_rate * dw for w, dw in zip(self.weights, delta_weights)]
        self.biases = [b - self.learning_rate * db for b, db in zip(self.biases, delta_biases)]

    def train(self, x_train, y_train, epochs,lr):
        """
        Entrena la red neuronal utilizando descenso por gradiente.
        x_train: Datos de entrada de entrenamiento (n_features, n_samples).
        y_train: Salida esperada (1, n_samples).
        epochs: Número de épocas de entrenamiento.
        """
        self.learning_rate = lr
        

        n_samples = x_train.shape[1]

        for epoch in range(epochs):
            total_loss = 0
            for i in range(n_samples):
                # Seleccionar la muestra i-ésima
                x = x_train[:, i].reshape(-1, 1)
                y = y_train[:, i].reshape(-1, 1)

                # Calcular los gradientes usando backpropagation
                delta_weights, delta_biases = self.backward(x, y)

                # Actualizar los parámetros
                self.update_parameters(delta_weights, delta_biases)

                # Calcular la pérdida acumulada
                y_pred = self.forward(x)[0][-1]
                total_loss += self.compute_loss(y, y_pred)

            # Imprimir la pérdida cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / n_samples}")
        print(self.weights)
        

    def predict(self, x):
        """
        Realiza una predicción para múltiples muestras.
        x: Datos de entrada (n_features, n_samples).
        Retorna un array con las predicciones.
        """
        n_samples = x.shape[1]
        predictions = []

        for i in range(n_samples):
            sample = x[:, i].reshape(-1, 1)

            #prediction = self.forward(sample)[0][-1]
            #predictions.append(prediction)

            prediction = self.forward(sample)[0][-1].flatten()
            predictions.append(prediction)
        return np.array(predictions)

