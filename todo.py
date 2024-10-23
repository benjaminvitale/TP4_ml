import model as md
import preprocessing as ps
import data_splitting as ds
import curacion as cc
import metrics as mc
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler


def save_results():
    data_frame_dev = cc.cd
    data_frame_test = cc.data_2_frame
   
    


  
    
    neural_net = md.NeuralNetwork([15,10,8,4,1],0.01)
    neural_net.train(x_train.T,y_train.T,250,[15,10,8,4,1],0.01)
    y_pred = np.array(neural_net.predict(x_dev.T))
    y_pred.reshape(-1,1)
    print(y_pred)
    print(mc.RMSE(y_dev,y_pred))

    
save_results()

x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

#print(x_train.shape)
#print(y_train.shape)