import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.01, max_iter=1000):
        """
        layers: List that defines the architecture of the network.
                Example: [2, 4, 1] -> 2 inputs, 4 neurons in one hidden layer, 1 output
        learning_rate: Learning rate for gradient descent
        max_iter: Number of iterations for training
        """
        self.layers = layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        np.random.seed(42)
        for i in range(1, len(layers)):
            # Weight matrix between layer i-1 and i
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            # Bias vector for layer i
            self.biases.append(np.random.randn(layers[i], 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward_propagation(self, X):
        """
        Perform forward propagation and store all layer activations.
        X: Input matrix of shape (n_features, n_samples)
        """
        activations = [X]
        z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            a = self._sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values

    def _backward_propagation(self, X, Y, activations, z_values):
        """
        Perform backward propagation to calculate gradients.
        """
        m = Y.shape[1]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        gradients_b = [np.zeros(b.shape) for b in self.biases]

        # Calculate the error for the output layer
        error = activations[-1] - Y
        delta = error * self._sigmoid_derivative(z_values[-1])

        # Backpropagate the error
        for l in reversed(range(len(self.weights))):
            gradients_w[l] = np.dot(delta, activations[l].T) / m
            gradients_b[l] = np.sum(delta, axis=1, keepdims=True) / m

            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self._sigmoid_derivative(z_values[l-1])

        return gradients_w, gradients_b

    def _update_parameters(self, gradients_w, gradients_b):
        """
        Update weights and biases using the calculated gradients.
        """
        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * gradients_w[l]
            self.biases[l] -= self.learning_rate * gradients_b[l]

    def fit(self, X, Y):
        """
        Train the neural network using gradient descent.
        X: Input data matrix of shape (n_samples, n_features)
        Y: Output data matrix of shape (n_samples, n_outputs)
        """
        X = X.T  # Shape should be (n_features, n_samples)
        Y = Y.T  # Shape should be (n_outputs, n_samples)

        for i in range(self.max_iter):
            # Forward propagation
            activations, z_values = self._forward_propagation(X)

            # Compute gradients
            gradients_w, gradients_b = self._backward_propagation(X, Y, activations, z_values)

            # Update parameters
            self._update_parameters(gradients_w, gradients_b)

            # Compute the loss every 100 iterations
            if i % 100 == 0:
                loss = np.mean(np.square(activations[-1] - Y))
                print(f"Iteration {i}: Loss = {loss:.4f}")

    def predict(self, X):
        """
        Predict the output for a given input.
        X: Input data matrix of shape (n_samples, n_features)
        """
        X = X.T
        activations, _ = self._forward_propagation(X)
        return activations[-1].T



class MLP(object):

    def __init__(self, layers=[4, 5, 1], activations=["relu", "sigmoid"], verbose=True, plot=False) -> None:
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
        eps = 1e-9 # Add small constant 1e-9 to avoid log of zero
        loss = -np.sum(y * np.log(output + eps) + (1 - y) * np.log(1 - output + eps)) / y.shape[0] 
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
            Average binary cross-entropy loss on the test data.
        """
        sum_loss = 0
        for x, y in test_data:
            prediction = self.forward_pass(x)[-1][-1]
            # Compute binary cross-entropy loss
            sum_loss += -np.sum(y * np.log(prediction + 1e-9) + (1 - y) * np.log(1 - prediction + 1e-9))
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



class NeuralNetworkWithScheduler:
    def __init__(self, layers, initial_lr=0.01, max_iter=1000, scheduler_type='linear', decay_rate=0.01, power=1):
        """
        layers: List that defines the architecture of the network.
                Example: [2, 4, 1] -> 2 inputs, 4 neurons in one hidden layer, 1 output
        initial_lr: Initial learning rate for gradient descent
        max_iter: Number of iterations (epochs) for training
        scheduler_type: Type of scheduler ('linear', 'power', 'exponential')
        decay_rate: Decay rate for learning rate adjustment
        power: Power factor for 'power law' scheduler
        """
        self.layers = layers
        self.initial_lr = initial_lr
        self.max_iter = max_iter
        self.scheduler_type = scheduler_type
        self.decay_rate = decay_rate
        self.power = power
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        np.random.seed(42)
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward_propagation(self, X):
        activations = [X]
        z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            a = self._sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values

    def _backward_propagation(self, X, Y, activations, z_values):
        m = Y.shape[1]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        gradients_b = [np.zeros(b.shape) for b in self.biases]

        error = activations[-1] - Y
        delta = error * self._sigmoid_derivative(z_values[-1])

        for l in reversed(range(len(self.weights))):
            gradients_w[l] = np.dot(delta, activations[l].T) / m
            gradients_b[l] = np.sum(delta, axis=1, keepdims=True) / m

            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self._sigmoid_derivative(z_values[l-1])

        return gradients_w, gradients_b

    def _update_parameters(self, gradients_w, gradients_b, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * gradients_w[l]
            self.biases[l] -= learning_rate * gradients_b[l]

    def _adjust_learning_rate(self, epoch):
        if self.scheduler_type == 'linear':
            # Linear decay: lr = initial_lr - decay_rate * epoch
            return self.initial_lr - self.decay_rate * epoch
        elif self.scheduler_type == 'power':
            # Power law: lr = initial_lr / (1 + decay_rate * epoch)^power
            return self.initial_lr / (1 + self.decay_rate * epoch) ** self.power
        elif self.scheduler_type == 'exponential':
            # Exponential decay: lr = initial_lr * exp(-decay_rate * epoch)
            return self.initial_lr * np.exp(-self.decay_rate * epoch)
        else:
            # Default to no adjustment
            return self.initial_lr

    def fit(self, X, Y):
        X = X.T
        Y = Y.T
        losses = []

        for epoch in range(self.max_iter):
            learning_rate = self._adjust_learning_rate(epoch)
            
            activations, z_values = self._forward_propagation(X)
            gradients_w, gradients_b = self._backward_propagation(X, Y, activations, z_values)
            self._update_parameters(gradients_w, gradients_b, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(activations[-1] - Y))
                losses.append(loss)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Learning Rate = {learning_rate:.6f}")
        
        # Plotting the loss curve
        plt.plot(range(0, self.max_iter, 100), losses, label=self.scheduler_type)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.show()

    def predict(self, X):
        X = X.T
        activations, _ = self._forward_propagation(X)
        return activations[-1].T

# Crear instancias de la red con diferentes estrategias de ajuste del learning rate
nn_linear = NeuralNetworkWithScheduler(layers=[2, 4, 1], initial_lr=0.1, max_iter=1000, scheduler_type='linear', decay_rate=0.001)
nn_power = NeuralNetworkWithScheduler(layers=[2, 4, 1], initial_lr=0.1, max_iter=1000, scheduler_type='power', decay_rate=0.001, power=1)
nn_exponential = NeuralNetworkWithScheduler(layers=[2, 4, 1], initial_lr=0.1, max_iter=1000, scheduler_type='exponential', decay_rate=0.001)





class NeuralNetworkWithL2:
    def __init__(self, layers, initial_lr=0.01, max_iter=1000, lambda_=0.01):
        """
        layers: List that defines the architecture of the network.
                Example: [2, 4, 1] -> 2 inputs, 4 neurons in one hidden layer, 1 output
        initial_lr: Initial learning rate for gradient descent
        max_iter: Number of iterations (epochs) for training
        lambda_: L2 regularization parameter
        """
        self.layers = layers
        self.initial_lr = initial_lr
        self.max_iter = max_iter
        self.lambda_ = lambda_
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        np.random.seed(42)
        for i in range(1, len(layers)):
            self.weights.append(np.random.randn(layers[i], layers[i-1]))
            self.biases.append(np.random.randn(layers[i], 1))

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _forward_propagation(self, X):
        activations = [X]
        z_values = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(w, activations[-1]) + b
            a = self._sigmoid(z)
            z_values.append(z)
            activations.append(a)

        return activations, z_values

    def _backward_propagation(self, X, Y, activations, z_values):
        m = Y.shape[1]
        gradients_w = [np.zeros(w.shape) for w in self.weights]
        gradients_b = [np.zeros(b.shape) for b in self.biases]

        error = activations[-1] - Y
        delta = error * self._sigmoid_derivative(z_values[-1])

        for l in reversed(range(len(self.weights))):
            gradients_w[l] = (np.dot(delta, activations[l].T) / m) + (self.lambda_ / m) * self.weights[l]
            gradients_b[l] = np.sum(delta, axis=1, keepdims=True) / m

            if l > 0:
                delta = np.dot(self.weights[l].T, delta) * self._sigmoid_derivative(z_values[l-1])

        return gradients_w, gradients_b

    def _update_parameters(self, gradients_w, gradients_b, learning_rate):
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * gradients_w[l]
            self.biases[l] -= learning_rate * gradients_b[l]

    def fit(self, X, Y, X_val=None, Y_val=None):
        """
        X: Training data
        Y: Training labels
        X_val: Validation data
        Y_val: Validation labels
        """
        X = X.T
        Y = Y.T
        losses = []
        val_losses = []

        for epoch in range(self.max_iter):
            learning_rate = self.initial_lr
            
            # Forward propagation
            activations, z_values = self._forward_propagation(X)
            
            # Backward propagation
            gradients_w, gradients_b = self._backward_propagation(X, Y, activations, z_values)
            
            # Update parameters
            self._update_parameters(gradients_w, gradients_b, learning_rate)

            # Calculate training loss with regularization
            loss = np.mean(np.square(activations[-1] - Y)) + (self.lambda_ / (2 * Y.shape[1])) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
            losses.append(loss)
            
            # Calculate validation loss if validation data is provided
            if X_val is not None and Y_val is not None:
                val_activations, _ = self._forward_propagation(X_val.T)
                val_loss = np.mean(np.square(val_activations[-1] - Y_val.T))
                val_losses.append(val_loss)
            
            # Print the loss every 100 epochs
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Training Loss = {loss:.4f}")

        # Plot the training and validation loss
        plt.plot(losses, label='Training Loss')
        if X_val is not None and Y_val is not None:
            plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Curve with L2 Regularization')
        plt.legend()
        plt.show()

    def predict(self, X):
        X = X.T
        activations, _ = self._forward_propagation(X)
        return activations[-1].T

nn_with_l2 = NeuralNetworkWithL2(layers=[2, 4, 1], initial_lr=0.1, max_iter=1000, lambda_=0.01)


