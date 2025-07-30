# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 10:29:59 2022

@author: egarate
"""
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


def retrain_scaler_controllable(data_all, data_path_scaler = "", save_pickle = True, return_data = False):
    X = data_all[["---"]] #Variables selection is confidential
    
    X_num = X.drop(["---"], axis = 1) #Confidential
    
    scaler = StandardScaler()
    scaler = scaler.fit(X_num)
    
    if save_pickle:
        with open(data_path_scaler, "wb") as file:
            pickle.dump(scaler, file)

    if return_data:
        X_num_sc = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
        X_all = X_num_sc.join(X[["---"]]) #Confidential
        
        data_to_return = X_all
        
    else:
        data_to_return = NULL
    
    return data_to_return


def load_scaler_and_transform(data_all, data_path_scaler = "/Outputs/Models/M1_contr_scaler.pkl"):
    
    X_num = data_all[["---"]].drop(["---"], axis = 1) #Variables selection is confidential
    
    with open(data_path_scaler, 'rb') as file:
        scaler = pickle.load(file)

    data_not_num = data_all.drop(["---"], axis = 1) #Variables are confidential
    
    X_num_sc = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    data_all_to_return = X_num_sc.join(data_not_num)
    
    return(data_all_to_return)
                     








