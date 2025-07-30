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

#mirar si hay problemas para NAs
def predict_M2(data_scaled_and_filtered, model_path = "C:/Users/egarate/Desktop/AI_PROFICIENT-UC2_DT/Outputs/Models/M2/M2_final_sin_filtrado.pkl"):
  
  with open(model_path, 'rb') as file:
    model = pickle.load(file)
    
  pred = model.predict(data_scaled_and_filtered)

  return pred


  
def retrain_M2(XY_filtered_scaled_with_recipes, model_path = "C:/Users/egarate/Desktop/AI_PROFICIENT-UC2_DT/Outputs/Models/M2/M2_final_sin_filtrado.pkl"):
  
  #definition of X and Y
  
    #drop NA values, seguramente no sea necesario, ya que lo voy a hacer en el anterior paso
  XY_filtered_scaled_with_recipes.dropna(inplace = True)
  
    #Define X and Y
  X = XY_filtered_scaled_with_recipes.drop("EX_DS_Hot_prof_real_first", axis = 1)
  Y = XY_filtered_scaled_with_recipes[['EX_DS_Hot_prof_real_first']]
  
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
    
    data_filtered_num = data_filtered[["EX2_mean_press_OFF", "EX2_min_press_OFF", "EX2_max_press_OFF", 
                  "EX2_sd_press_OFF", "EX2_first_press_OFF", "EX2_last_press_OFF", 
                  "EX1_mean_press_OFF", "EX1_min_press_OFF", "EX1_max_press_OFF", 
                  "EX1_first_press_OFF", "EX1_last_press_OFF", "EX1_sd_press_OFF", 
                  "EX3_mean_press_OFF", "EX3_min_press_OFF", "EX3_max_press_OFF", 
                  "EX3_first_press_OFF", "EX3_last_press_OFF", "EX3_sd_press_OFF", 
                  "EX4_mean_press_OFF", "EX4_min_press_OFF", "EX4_max_press_OFF", 
                  "EX4_first_press_OFF", "EX4_last_press_OFF",  "EX4_sd_press_OFF",  
                  "EX2_mean_temp_OFF", "EX2_min_temp_OFF", "EX2_max_temp_OFF", 
                  "EX2_sd_temp_OFF", "EX2_first_temp_OFF", "EX2_last_temp_OFF", 
                  "EX1_mean_temp_OFF" , "EX1_min_temp_OFF", "EX1_max_temp_OFF", 
                  "EX1_sd_temp_OFF", "EX1_first_temp_OFF", "EX1_last_temp_OFF", 
                  "EX3_mean_temp_OFF", "EX3_min_temp_OFF", "EX3_max_temp_OFF", 
                  "EX3_sd_temp_OFF", "EX3_first_temp_OFF", "EX3_last_temp_OFF", 
                  "EX4_mean_temp_OFF", "EX4_min_temp_OFF", "EX4_max_temp_OFF", 
                  "EX4_sd_temp_OFF", "EX4_first_temp_OFF", "EX4_last_temp_OFF", 
                  "EX5_mean_temp_OFF", "EX5_min_temp_OFF", "EX5_max_temp_OFF", 
                  "EX5_sd_temp_OFF", "EX5_first_temp_OFF", "EX5_last_temp_OFF", 
                  "EX2_setpoint_speed", "EX2_speed_slope", "EX1_setpoint_speed", 
                  "EX1_speed_slope","EX3_setpoint_speed", "EX3_speed_slope", 
                  "EX4_setpoint_speed", "EX4_speed_slope","EX5_setpoint_speed", 
                  "EX5_speed_slope","EX1_mean_press_OFF_int", "EX1_min_press_OFF_int", 
                  "EX1_max_press_OFF_int", "EX1_first_press_OFF_int", "EX1_last_press_OFF_int", 
                  "EX1_sd_press_OFF_int", "EX3_mean_press_OFF_int", "EX3_min_press_OFF_int",
                  "EX3_max_press_OFF_int", "EX3_first_press_OFF_int", "EX3_last_press_OFF_int", 
                  "EX3_sd_press_OFF_int", "EX4_mean_press_OFF_int", "EX4_min_press_OFF_int",
                  "EX4_max_press_OFF_int", "EX4_first_press_OFF_int", "EX4_last_press_OFF_int",
                  "EX4_sd_press_OFF_int", "EX1_mean_temp_OFF_int", "EX1_min_temp_OFF_int",
                  "EX1_max_temp_OFF_int", "EX1_sd_temp_OFF_int", "EX1_first_temp_OFF_int",
                  "EX1_last_temp_OFF_int", "EX3_mean_temp_OFF_int", "EX3_min_temp_OFF_int",
                  "EX3_max_temp_OFF_int", "EX3_sd_temp_OFF_int", "EX3_first_temp_OFF_int",
                  "EX3_last_temp_OFF_int", "EX4_mean_temp_OFF_int", "EX4_min_temp_OFF_int",
                  "EX4_max_temp_OFF_int", "EX4_sd_temp_OFF_int", "EX4_first_temp_OFF_int",
                  "EX4_last_temp_OFF_int", "EX5_mean_temp_OFF_int", "EX5_min_temp_OFF_int",
                  "EX5_max_temp_OFF_int", "EX5_sd_temp_OFF_int", "EX5_first_temp_OFF_int",
                  "EX5_last_temp_OFF_int","EX1_setpoint_speed_int", "EX1_speed_slope_int",
                  "EX3_setpoint_speed_int", "EX3_speed_slope_int", "EX4_setpoint_speed_int",
                  "EX4_speed_slope_int", "EX5_setpoint_speed_int", "EX5_speed_slope_int"]] 
                  
    data_filtered_num_sc = pd.DataFrame(scaler.transform(data_filtered_num), index = data_filtered_num.index, columns = data_filtered_num.columns)
    
    data_scaled_and_filtered = data_filtered_num_sc.join(data_filtered.iloc[:, 64:68]).join(data_filtered.iloc[:, 118:229])
    
    
    M2_pred = predict_M2(data_scaled_and_filtered, model_path)
  
  
    return M2_pred
