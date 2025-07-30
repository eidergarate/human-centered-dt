import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle


def unscale_data(X_num, scaler_path):
  
  with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
  
  X_num_unscaled = pd.DataFrame(scaler.inverse_transform(X_num), columns = X_num.columns)
  
  
  return X_num_unscaled
  

def scale_data_opt(X_num, scaler_path):
  
  with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
  
  X_num_scaled = pd.DataFrame(scaler.transform(X_num), columns = X_num.columns)
  
  return X_num_scaled
