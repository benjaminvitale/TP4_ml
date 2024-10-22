import model as md
import preprocessing as ps
import data_splitting as ds
import curacion as cc
import metrics as mc
import pandas as pd
def save_results():
    data_frame_dev = cc.cd
    data_frame_test = cc.data_2_frame
    x_train,y_train,x_dev,y_dev,vals_min,vals_max = ds.data_split_dev(data_frame_dev)
    x_test_2,y_test_2 = ds.data_create_test(data_frame_test,vals_min,vals_max)
    X_kf,Y_kf = ds.data_create_test(data_frame_dev,vals_min,vals_max)


    neural_net = md.MLP
    neural_net.fit((x_train,y_train),10,200,0.1)
    print(neural_net.evaluate((x_dev,y_dev)))
save_results()
