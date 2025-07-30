# -*- coding: utf-8 -*-
"""
Retraining and predict of M2

Created on Wed Feb 08 12:17:59 2023

@author: egarate
"""


import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

def predict_M2(data_scaled_and_filtered, model_path):
  
  with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
  pred = model.predict(data_scaled_and_filtered)

  return pred


  
def retrain_M2(XY_filtered_scaled_with_recipes, model_path):
  
  XY_filtered_scaled_with_recipes.dropna(inplace = True)
  
  X = XY_filtered_scaled_with_recipes.drop("quality", axis = 1)
  Y = XY_filtered_scaled_with_recipes[['quality']]
  
  #retraining
  best_model = xgb.XGBRegressor(max_depth = 2, n_estimators = 10)
  best_model.fit(X, Y)

  
  #pickle saving
  if save_pickle:
      with open(model_path, "wb") as file:
          pickle.dump(best_model, file)

  return True

def scale_and_predict_M2(data_filtered, scaler_path, model_path):
  
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    data_filtered_num = data_filtered[["---"]] #Confidential
                  
    data_filtered_num_sc = pd.DataFrame(scaler.transform(data_filtered_num), index = data_filtered_num.index, columns = data_filtered_num.columns)
    
    data_scaled_and_filtered = data_filtered_num_sc.join(data_filtered.iloc[:, 64:68]).join(data_filtered.iloc[:, 118:229])
    
    
    M2_pred = predict_M2(data_scaled_and_filtered, model_path)
  
  
    return M2_pred
