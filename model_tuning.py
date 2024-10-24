import metrics as mc
import matplotlib.pyplot as plt
import curacion as cc
import model as md

def search_hyperparameters(model,x_train,y_train,x_dev,y_dev,decay_type,decay_rate):
    best_val_loss = float('inf')
    best_params = None
    
    for dr in decay_rate:
            
        # Create the model with the current hyperparameters
        model.train2(True,x_train,y_train,x_dev,y_dev.T,500,0.00000001,decay_type,dr)
        y_pred = model.predict(x_dev)

        
        
        final_val_loss = mc.RMSE(y_dev,y_pred)
        
        # Update the best parameters if the current validation loss is lower
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = dr
                
    
    print(f"\nBest decay rate found: {best_params} with Validation Loss: {best_val_loss:.4f}")
    return best_params, best_val_loss


def find_lambda(model,x_train,y_train,x_dev,y_dev,lambassss):
    best_val_loss = float('inf')
    best_params = None
    for lambda_ in lambassss:
        
        model.train(True,x_train,y_train,x_dev,y_dev.T,500,0.00000001,True,lambda_)
        y_pred = model.predict(x_dev)
        
        final_val_loss = mc.RMSE(y_dev,y_pred)
        
        # Update the best parameters if the current validation loss is lower
        if final_val_loss < best_val_loss:
            best_val_loss = final_val_loss
            best_params = lambda_
    print(f"\nBest lambda found: {best_params} with Validation Loss: {best_val_loss:.4f}")
    return best_params, best_val_loss


def compare_optimizers(model,x_train, y_train, x_val, y_val, epochs, lr,batch_size):
    optimizers = ['sgd', 'momentum', 'adam']
    results = {}

    for optimizer in optimizers:
        print(f"\nEntrenando con {optimizer.upper()}")
        train_losses, val_losses = model.train3(x_train, y_train, epochs, lr,optimizer, batch_size, x_val, y_val.T)
        y_pred = model.predict(x_val)
        print('el RMSE de la Red Neuronal con optimizador  ',optimizer, 'es : ',mc.RMSE(y_val,y_pred))
        print('el MAE de la Red Neuronal con optimizador  ',optimizer, 'es : ',mc.mean_absolute_error(y_val,y_pred))
        print('el R2 de la Red Neuronal con optimizador  ',optimizer, 'es : ',mc.rsquare(y_val,y_pred))
        results[optimizer] = (train_losses, val_losses)

    # Graficar los errores de entrenamiento y validación
    plt.figure(figsize=(12, 6))
    for optimizer in optimizers:
        plt.plot(results[optimizer][0], label=f'Train Loss - {optimizer}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Error for Different Optimizers')
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(12, 6))
    for optimizer in optimizers:
        plt.plot(results[optimizer][1], label=f'Validation Loss - {optimizer}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Error for Different Optimizers')
    plt.legend()
    plt.grid()
    plt.show()

    return results



import numpy as np



def Layers_neurons_search(x_data, y_data, layer_options, neuron_options,k_folds=5):
    """
    Selección de hiperparámetros (L y M(l)) usando cross-validation con 5 folds.
    """
    best_score = float('inf')
    best_hyperparams = None

    # Generar los indices de los k folds
    
    # Iterar sobre cada configuración de capas y neuronas
    for num_layers in layer_options:
        for num_neurons in neuron_options:
            print(f"Probando configuración: {num_layers} capas ocultas, {num_neurons} neuronas por capa.")
            fold_scores = []

            # Cross-validation manual
            for fold in range(k_folds):
                # Dividir en conjunto de entrenamiento y validación
                val_indices = cc.folds[fold]
                train_indices = np.concatenate([cc.folds[i] for i in range(k_folds) if i != fold])

                x_train, y_train = x_data[:, train_indices], y_data[:, train_indices]
                x_val, y_val = x_data[:, val_indices], y_data[:, val_indices]

                # Configurar la arquitectura de la red: capa de entrada, ocultas y salida
                layer_sizes = [x_data.shape[0]] + [num_neurons] * num_layers + [1]
                nn = md.NeuralNetwork(layer_sizes)

                # Entrenar la red con la configuración actual
                nn.train(False, x_train, y_train,x_val,y_val, 200,0.00000001,False,0)

                # Calcular el error en el conjunto de validación
                y_pred_val = nn.predict(x_val)
                val_loss = mc.RMSE(y_val,y_pred_val)
                fold_scores.append(val_loss)

            # Promediar el error en los k folds
            avg_score = np.mean(fold_scores)
            print(f"Configuración {num_layers} capas, {num_neurons} neuronas, RMSE promedio: {avg_score}")

            # Actualizar mejor configuración si el score es mejor
            if avg_score < best_score:
                best_score = avg_score
                best_hyperparams = (num_layers, num_neurons)

    print(f"\nMejor configuración: {best_hyperparams[0]} capas ocultas, {best_hyperparams[1]} neuronas por capa, con score promedio: {best_score}")
    return best_hyperparams



