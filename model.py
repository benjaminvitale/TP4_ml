import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


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
        self.regularization_param = 0
        
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

    def compute_loss(self, y_true, y_pred,regs):
        """Calcula la pérdida de la red (suma de errores cuadráticos)."""
        loss = 0.5 * np.sum((y_true - y_pred) ** 2)
        if regs:
            l2_regularization = 0.5 * self.regularization_param * sum(np.sum(w**2) for w in self.weights)
            return loss + l2_regularization
        return loss

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


    def update_learning_rate(self,lr, epoch, decay_type, decay_rate, power):
        """
        Actualiza la tasa de aprendizaje según el tipo de decaimiento especificado.
        Parámetros:
            lr: Tasa de aprendizaje actual.
            epoch: Número de época actual.
            decay_type: Tipo de decaimiento ("linear", "power", "exponential").
            decay_rate: Tasa de decaimiento (utilizado para decaimiento lineal y exponencial).
            power: Factor de potencia (utilizado para decaimiento por ley de potencia).
        Retorna:
            Nueva tasa de aprendizaje.
        """
        if decay_type == "linear":
            # Decaimiento lineal
            new_lr = lr - (decay_rate * epoch)
            return new_lr # Asegurar que la tasa de aprendizaje no sea menor a un valor mínimo
        elif decay_type == "power":
            # Decaimiento por ley de potencia
            new_lr = lr * (1 + epoch) ** -power
            return new_lr
        elif decay_type == "exponential":
            # Decaimiento exponencial
            new_lr = lr * np.exp(-decay_rate * epoch)
            return new_lr
        else:
            raise ValueError("Tipo de decaimiento desconocido. Elija 'linear', 'power' o 'exponential'.")


    def train2(self,plot, x_train, y_train,x_val,y_val, epochs, initial_lr, decay_type, decay_rate, power=1.0):
        """
        Entrena la red neuronal utilizando descenso por gradiente con ajuste dinámico de la tasa de aprendizaje.
        x_train: Datos de entrada de entrenamiento (n_features, n_samples).
        y_train: Salida esperada (1, n_samples).
        epochs: Número de épocas de entrenamiento.
        initial_lr: Tasa de aprendizaje inicial.
        decay_type: Tipo de decaimiento de la tasa de aprendizaje ("linear", "power", "exponential").
        decay_rate: Tasa de decaimiento para el ajuste.
        power: Factor de potencia (para el decaimiento por ley de potencia).
        """
        self.learning_rate = initial_lr
        n_samples = x_train.shape[1]
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Actualizar la tasa de aprendizaje según la época actual
            self.learning_rate = self.update_learning_rate(initial_lr, epoch, decay_type, decay_rate, power)

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
                total_loss += self.compute_loss(y, y_pred,False)


            avg_train_loss = total_loss / n_samples
            train_losses.append(avg_train_loss)
            # Imprimir la pérdida cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / n_samples}, Learning Rate: {self.learning_rate}")
            
            if x_val is not None:
                val_loss = np.mean([self.compute_loss(y_val[:, j].reshape(-1, 1), self.forward(x_val[:, j].reshape(-1, 1))[0][-1],False) for j in range(x_val.shape[1])])
                val_losses.append(val_loss)
            
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Error vs Epochs for Decay Rate {decay_rate} ({decay_type} decay)')
            plt.legend()
            plt.grid()
            plt.show()

    def train(self,plot, x_train, y_train,x_val,y_val, epochs,lr,regs,lambda_):
        """
        Entrena la red neuronal utilizando descenso por gradiente.
        x_train: Datos de entrada de entrenamiento (n_features, n_samples).
        y_train: Salida esperada (1, n_samples).
        epochs: Número de épocas de entrenamiento.
        """
        self.learning_rate = lr
        self.regularization_param = lambda_
        
        train_losses = []
        val_losses = []
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
                total_loss += self.compute_loss(y, y_pred,regs)
            loss_ = total_loss / n_samples
            train_losses.append(loss_)
            # Imprimir la pérdida cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss / n_samples}")
            if x_val is not None:
                val_loss = np.mean([self.compute_loss(y_val[:, j].reshape(-1, 1), self.forward(x_val[:, j].reshape(-1, 1))[0][-1],regs) for j in range(x_val.shape[1])])
                val_losses.append(val_loss)
        if plot:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'Error vs Epochs')
            plt.legend()
            plt.grid()
            plt.show()
    
    def train3(self, x_train, y_train, epochs, lr, optimizer, batch_size, x_val, y_val):
        self.learning_rate = lr

        # Parámetros de optimizadores
        v_weights = [np.zeros_like(w) for w in self.weights]  # Momentum para SGD con momentum
        v_biases = [np.zeros_like(b) for b in self.biases]
        m_weights = [np.zeros_like(w) for w in self.weights]  # Para Adam
        m_biases = [np.zeros_like(b) for b in self.biases]
        v_hat_weights = [np.zeros_like(w) for w in self.weights]  # Para Adam
        v_hat_biases = [np.zeros_like(b) for b in self.biases]
        beta1 = 0.9  # Parámetro Adam
        beta2 = 0.999  # Parámetro Adam
        epsilon = 1e-8  # Parámetro Adam

        train_losses = []
        val_losses = []


        for epoch in range(epochs):
            total_loss = 0
            indices = np.arange(x_train.shape[1])
            np.random.shuffle(indices)  # Mezclar los datos

            # Mini-batch SGD
            for start in range(0, x_train.shape[1], batch_size):
                end = min(start + batch_size, x_train.shape[1])
                batch_indices = indices[start:end]
                x_batch = x_train[:, batch_indices]
                y_batch = y_train[:, batch_indices]

                delta_weights = [np.zeros_like(w) for w in self.weights]
                delta_biases = [np.zeros_like(b) for b in self.biases]

                # Acumular gradientes para el mini-batch
                for i in range(x_batch.shape[1]):
                    x = x_batch[:, i].reshape(-1, 1)
                    y = y_batch[:, i].reshape(-1, 1)
                    dw, db = self.backward(x, y)
                    delta_weights = [dw_sum + dw_i for dw_sum, dw_i in zip(delta_weights, dw)]
                    delta_biases = [db_sum + db_i for db_sum, db_i in zip(delta_biases, db)]

                # Promediar gradientes en el mini-batch
                delta_weights = [dw / batch_size for dw in delta_weights]
                delta_biases = [db / batch_size for db in delta_biases]

                # Actualizar parámetros usando el optimizador seleccionado
                if optimizer == 'sgd':
                    self.update_parameters(delta_weights, delta_biases)
                elif optimizer == 'momentum':
                    momentum_factor = 0.9
                    v_weights = [momentum_factor * vw + dw for vw, dw in zip(v_weights, delta_weights)]
                    v_biases = [momentum_factor * vb + db for vb, db in zip(v_biases, delta_biases)]
                    self.update_parameters(v_weights, v_biases)
                elif optimizer == 'adam':
                    beta1_pow = beta1 ** (epoch + 1)
                    beta2_pow = beta2 ** (epoch + 1)
                    m_weights = [beta1 * mw + (1 - beta1) * dw for mw, dw in zip(m_weights, delta_weights)]
                    m_biases = [beta1 * mb + (1 - beta1) * db for mb, db in zip(m_biases, delta_biases)]
                    v_hat_weights = [beta2 * vw + (1 - beta2) * (dw ** 2) for vw, dw in zip(v_hat_weights, delta_weights)]
                    v_hat_biases = [beta2 * vb + (1 - beta2) * (db ** 2) for vb, db in zip(v_hat_biases, delta_biases)]
                    m_weights_hat = [mw / (1 - beta1_pow) for mw in m_weights]
                    m_biases_hat = [mb / (1 - beta1_pow) for mb in m_biases]
                    v_hat_weights_hat = [vw / (1 - beta2_pow) for vw in v_hat_weights]
                    v_hat_biases_hat = [vb / (1 - beta2_pow) for vb in v_hat_biases]
                    adam_weights = [mw_hat / (np.sqrt(vw_hat) + epsilon) for mw_hat, vw_hat in zip(m_weights_hat, v_hat_weights_hat)]
                    adam_biases = [mb_hat / (np.sqrt(vb_hat) + epsilon) for mb_hat, vb_hat in zip(m_biases_hat, v_hat_biases_hat)]
                    self.update_parameters(adam_weights, adam_biases)

                y_pred = self.forward(x)[0][-1]
                total_loss += self.compute_loss(y, y_pred,False)
            loss_ = total_loss / x_train.shape[1]
            train_losses.append(loss_)
            # Calcular el error en el conjunto de entrenamiento y validación
            if x_val is not None:
                val_loss = np.mean([self.compute_loss(y_val[:, j].reshape(-1, 1), self.forward(x_val[:, j].reshape(-1, 1))[0][-1],False) for j in range(x_val.shape[1])])
                val_losses.append(val_loss)

            # Imprimir progreso cada 100 épocas
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Train Loss: {loss_}, Validation Loss: {val_loss}")

        return train_losses, val_losses
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

