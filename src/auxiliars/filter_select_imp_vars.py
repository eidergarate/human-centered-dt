import pandas as pd
import numpy as np

def save_importances(importances, list_var, path = ""):
  importancias_df = pd.DataFrame(np.array([listvars, importance]).transpose(), columns = ['Var_name', 'importance'])
  importancias_df = importancias_df[importancias_df['importance'] >= 0.01 ]
  importancias_df.to_csv(path)



def filter_select(X_all, importances_path):
  importances_df = pd.read_csv(importances_path)
  
  X_filtered = X_all[importances_df["Var_name"]]
  
  return(X_filtered)

