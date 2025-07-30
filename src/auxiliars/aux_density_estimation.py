import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import pickle

def estimate_gaussian_kernel(X_num, save_scaler, scaler_path):
  
  scaler = StandardScaler().fit(X_num)
  
  X_num_sc = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
  
  X_num_sc = X_num_sc.dropna()
  
  kde = KernelDensity(kernel = 'gaussian').fit(X_num_sc)
  
  if save_scaler:
    
    with open(scaler_path, "wb") as file:
          pickle.dump(scaler, file)
  
  return X_num_sc, kde


def get_samples(X_num, n_samples, save_scaler = False, scaler_path):
  
  X_num_sc, kde = estimate_gaussian_kernel(X_num, save_scaler, scaler_path)
  
  numerical_samples = pd.DataFrame(kde.sample(int(n_samples)), columns = X_num.columns)
  
  return X_num_sc, numerical_samples
