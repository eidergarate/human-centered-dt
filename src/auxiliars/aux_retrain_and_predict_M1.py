# -*- coding: utf-8 -*-
"""
Retraining and predict of M1

Created on Mon Oct 17 15:43:59 2022

@author: egarate
"""


import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
import imblearn
from imblearn.combine import SMOTETomek
from Python.pipeline_auxiliars.filter_select_imp_vars import filter_select, save_importances

#mirar si hay problemas para NAs
def predict_M1(data_scaled_and_filtered, model_path, importances_path):
  
  data_scaled_and_filtered = filter_select(data_scaled_and_filtered, importances_path)
  
  with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
  pred = model.predict(data_scaled_and_filtered)
  
  return pred


  
def retrain_M1(XY_scaled_and_filtered, path_scaler, model_path, save_pickle, importances_path):
  
  XY_scaled_and_filtered.dropna(inplace = True)
  
  X = XY_scaled_and_filtered.drop("quality_status", axis = 1)
  Y = XY_scaled_and_filtered[['quality_status']]
  
  retrain_scaler_controllable(data_all, data_path_scaler = path_scaler, save_pickle = True, return_data = True)

  smt = SMOTETomek(sampling_strategy = 'all')
  
  min_samples_leaf = 5
  max_depth = 7
  rf_best = RandomForestClassifier(min_samples_leaf = min_samples_leaf, max_depth = max_depth)

  X_smt, Y_smt = smt.fit_resample(X, Y)
  
  rf_best.fit(X_smt, Y_smt)
  
  importance = rf_best.feature_importances_
  listvars = X.columns.values
  importancias_df = save_importances(importance, listvars, importances_path)
  
  X_smt_filtered = X_smt[importancias_df['Var_name']]
  
  rf_best_filtered = RandomForestClassifier(min_samples_leaf = min_samples_leaf, max_depth = max_depth)
  rf_best_filtered.fit(X_smt_filtered, Y_smt)
  
  if save_pickle:
      with open(model_path, "wb") as file:
          pickle.dump(rf_best_filtered, file)

  return True


def scale_and_predict_M1(data_filtered, scaler_path, model_path, importances_path):
  
  with open(scaler_path, 'rb') as file:
    scaler = pickle.load(file)
    
    data_filtered_num = data_filtered[["---"]] #Confidential
                  
    data_filtered_num_sc = pd.DataFrame(scaler.transform(data_filtered_num), index = data_filtered_num.index, columns = data_filtered_num.columns)
    
    data_scaled_and_filtered = data_filtered_num_sc.join(data_filtered[["---"]]) #Confidential
    
    
    M1_pred = predict_M1(data_scaled_and_filtered, model_path, importances_path)
    
    return(M1_pred)
    
    
    
    
