# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:00:36 2024

@author: egarate
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import imblearn
from imblearn.combine import SMOTETomek
import pickle
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as IMBPipeline

import warnings

warnings.filterwarnings("ignore")

def filter_by_date(filtered_data):
  
    filtered_data['Year'] = filtered_data['datetime_ini'].dt.year
    filtered_data['Month'] = filtered_data['datetime_ini'].dt.month
    filtered_data['Day'] = filtered_data['datetime_ini'].dt.day

    filtered_data = filtered_data[(filtered_data['datetime_ini'] >= pd.to_datetime("---"))] #confidential

    filtered_data = filtered_data.drop(columns=['Year', 'Month', 'Day', 'datetime_ini'])
    
    return filtered_data

def filter_data(data_update): #Confidential

    
    return filtered_data

def select_vars(filtered_data): #Confidential
    
    return selected_data

def get_interactions(filtered_and_selected_data): #Confidential
    
    return data_interactions

def split_X_Y(data_interactions):
    
    X = data_interactions.drop('quality_status', axis = 1)
    Y = data_interactions['quality_status']
    
    return X, Y

def train_test_split(data_interactions, split = 0.8):
    
    X, Y = split_X_Y(data_interactions)
    
    ntrain = int(X.shape[0]*split)
    
    X_train = X.sample(n = ntrain, replace = False, random_state = 42)
    X_train_idx = X_train.index
    
    X_test_idx = X.index.difference(X_train_idx)
    X_test = X.loc[X_test_idx, :]
    
    Y_train = Y.loc[X_train_idx]
    Y_test = Y.loc[X_test_idx]
    
    return X_train, Y_train, X_test, Y_test

def filter_select_vars(data_update):
    
    filtered_data = filter_data(data_update)
    
    filtered_and_selected_data = select_vars(filtered_data)
    
    filtered_and_selected_data = filtered_and_selected_data.dropna()
    
    return filtered_and_selected_data    

def scale_data(X_train, X_test):
    
    X_train_cat = X_train[['---']] #Confidential variables selected
    X_train_num = X_train.drop(['---'], axis = 1) #Confidential variables dropped
    
    X_test_cat = X_test[['---']] #Confidential variables selected
    X_test_num = X_test.drop(['---'], axis = 1) #Confidential variables dropped
    
    scaler = StandardScaler()
    
    scaler_trained = scaler.fit(X_train_num)
    X_train_num_scaled = pd.DataFrame(scaler_trained.transform(X_train_num), index = X_train_num.index, columns = X_train_num.columns)
    X_test_num_scaled = pd.DataFrame(scaler_trained.transform(X_test_num), index = X_test_num.index, columns = X_test_num.columns)
    
    X_train_scaled = X_train_num_scaled.join(X_train_cat)
    X_test_scaled = X_test_num_scaled.join(X_test_cat)
    
    return X_train_scaled, X_test_scaled, scaler

def balance_by_smote(X_train_scaled, Y_train):
    
    min_samples_class = Y_train.value_counts()
    min_samples_class = np.min(min_samples_class)
    smote = SMOTE(sampling_strategy='all', k_neighbors = min(3, max(1, min_samples_class - 1)))
    
    smt = SMOTETomek(smote = smote)
    X_train_scaled_smt, Y_train_smt = smt.fit_resample(X_train_scaled, Y_train)
    
    return X_train_scaled_smt, Y_train_smt
    

def get_all_sets_training(data_update, split):
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X_train, Y_train, X_test, Y_test = train_test_split(data_interactions, split)
    
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    X_train_scaled_smt, Y_train_smt = balance_by_smote(X_train_scaled, Y_train)
    
    train_mask = ~np.isnan(X_train_scaled_smt).any(axis = 1) & ~np.isinf(X_train_scaled_smt).any(axis = 1)
    X_train_scaled_smt = X_train_scaled_smt[train_mask]
    Y_train_smt = Y_train_smt[train_mask]
    
    test_mask = ~np.isnan(X_test_scaled).any(axis = 1) & ~np.isinf(X_test_scaled).any(axis = 1)
    X_test_scaled = X_test_scaled[test_mask]
    Y_test = Y_test[test_mask]
    
    return X_train_scaled_smt, Y_train_smt, X_test_scaled, Y_test, scaler


def filter_by_importances(X_scaled, importances_path):
    
    importances_df = pd.read_csv(importances_path)
    
    X_scaled_filtered = X_scaled[importances_df["Var_name"]]
    
    return X_scaled_filtered


def predict_by_old_model(X_scaled,  model_path, importances_path):
    
    X_scaled_filtered = filter_by_importances(X_scaled, importances_path)
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
      
    Y_pred = model.predict(X_scaled_filtered)
    
    return Y_pred

def retrain_not_really(data_update, model_path, importances_path, scaler_path):
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['---']] #Confidential variables selected
    X_num = X.drop(['---'], axis = 1) #Confidential variables dropped
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    X_scaled = X_num_scaled.join(X_cat)
    
    Y_pred = predict_by_old_model(X_scaled,  model_path, importances_path)
    
    computational_time = 0
    
    f1_cv = f1_test = f1_score(y_true = Y, y_pred = Y_pred, average = "macro")
    
    return computational_time, f1_cv, f1_test
        
def concat_and_get_k_plus_one_data(data_sk, data_update):

    data_k_plus_one = pd.concat([data_sk, data_update], axis=0)    
    
    data_k_plus_one = data_k_plus_one.reset_index(drop = True)
    
    return data_k_plus_one
    
def get_objects_retraining_simple(rf_params, data_update):
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['---']] #Confidential variables selected
    X_num = X.drop(['---'], axis = 1) #Confidential variables dropped
    
    scaler = StandardScaler()
    
    scaler_trained = scaler.fit(X_num)
    
    X_num_scaled = pd.DataFrame(scaler_trained.transform(X_num), index = X_num.index, columns = X_num.columns)
    X_scaled = X_num_scaled.join(X_cat)
    
    
    
    rf_best = RandomForestClassifier(**rf_params)
    
    rf_best.fit(X_scaled, Y)
    
    Y_pred = rf_best.predict(X_scaled)
    
    cm_new = confusion_matrix(Y, Y_pred)
    
    
    importances = rf_best.feature_importances_

    importances = pd.DataFrame({"Var_name" : X_scaled.columns, 'Values' : importances})   
    
    return rf_best, cm_new, scaler, importances

def retrain_simple(data_sk, data_update, split, new_names, cm_old):
    
    data_update = concat_and_get_k_plus_one_data(data_sk, data_update)
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X_train, Y_train, X_test, Y_test = train_test_split(data_interactions, split)
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    pipe = IMBPipeline([("smote", SMOTETomek()), ("randomforestclassifier", RandomForestClassifier())])
    
    params = [{ "randomforestclassifier__max_depth" : [2, 3, 4, 5, 6, 7], 
               'randomforestclassifier__min_samples_leaf' : [5, 10, 30, 50, 100], 
               "randomforestclassifier__bootstrap" : [True, False]}]
    
    ini_time = time.time()
    
    grid = GridSearchCV(
        estimator = pipe,
        param_grid = params,
        scoring = 'f1_macro',
        cv = 5
        )
    
    grid.fit(X_train_scaled, Y_train)
    
    end_time = time.time()
    
    computational_time = end_time - ini_time
    
    f1_cv = grid.best_score_
    
    Y_test_pred = grid.predict(X_test_scaled)
    
    f1_test = f1_score(y_true = Y_test, y_pred = Y_test_pred, average = "macro")
    
    
    best_params = grid.best_params_

    rf_params = {keys.replace('randomforestclassifier__', ''): values for keys, values in best_params.items()}
    
    rf_model, cm_new, scaler, importances_filtered = get_objects_retraining_simple(rf_params, data_update)
    
    save_models_results(rf_model, scaler, importances_filtered, new_names)
    
    return computational_time, f1_cv, f1_test
    
def get_model_weighted(Nk_plus_1, class_error_rate, Y_train, Y_update_pred):
    
    threshold = 1e-4
    
    class_error_rate[class_error_rate < threshold] = 0.1
    
    inside_ln_value = (1 - np.array(class_error_rate))/np.array(class_error_rate)
    
    inside_ln_value[np.abs(inside_ln_value) < threshold] = 0.1
    
    alphas = 1/2*np.log(inside_ln_value)
    
    weights = []
    
    for j_index in range(len(Y_update_pred)):
        
        xj = Y_train.iloc[j_index]
        xj_pred = Y_update_pred[j_index]
        
        alpha_i = alphas[int(xj)]
        
        w_xj = 1/Nk_plus_1*np.exp(-alpha_i*xj_pred)
        
        weights.append(w_xj)        
        
    return weights


def get_first_adaboost_weights(X_train_scaled, Y_train,  model_path, importances_path, scaler_path):
    
    Nk_plus_1 = X_train_scaled.shape[0] #M_{k+1}
    
    Y_update_pred = predict_by_old_model(X_train_scaled,  model_path, importances_path) #{M_k(xj)  |  xj in S_{k+1}}

    
    
    cm = confusion_matrix(Y_train, Y_update_pred, labels = [0, 1, 2])
    
    class_accuracy = cm.diagonal() / cm.sum(axis = 1)

    class_error_rate = 1 - class_accuracy
    
    W_k_plus_1 = get_model_weighted(Nk_plus_1, class_error_rate, Y_train, Y_update_pred)
    
    
    #normalize the weights
    sum_W_k_plus_1 = np.sum(W_k_plus_1)
    
    normalized_W_k_plus_1 = np.array(W_k_plus_1)/sum_W_k_plus_1
    
    return normalized_W_k_plus_1    

def get_objects_retraining_adaboost(hyperparameters, data_update, model_path, importances_path, scaler_path):
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['---']]#Confidential variables selected
    X_num = X.drop(['---'], axis = 1) #Confidential variables dropped
    
    scaler = StandardScaler()
    
    scaler_trained = scaler.fit(X_num)
    
    X_num_scaled = pd.DataFrame(scaler_trained.transform(X_num), index = X_num.index, columns = X_num.columns)
    X_scaled = X_num_scaled.join(X_cat)
    
    X_scaled = pd.concat([X_num_scaled, X_cat], axis = 1)
    
    X_scaled_smt, Y_smt = balance_by_smote(X_scaled, Y)
    
    importances = pd.DataFrame({"Var_name" : X_scaled.columns, 'Values' : [1/X_scaled.shape[1]] * X_scaled.shape[1]})  
    
    initial_weights = get_first_adaboost_weights(X_scaled_smt, Y_smt, model_path, importances_path, scaler_path)
        
    rf = RandomForestClassifier(max_depth = hyperparameters['max_depth'], min_samples_leaf = hyperparameters['min_samples_leaf'], bootstrap = hyperparameters['bootstrap'])
    
    ada_boost = AdaBoostClassifier(base_estimator = rf)
    
    ada_boost.fit(X_scaled_smt, Y_smt, sample_weight = initial_weights)
    
    Y_pred = ada_boost.predict(X_scaled)
    
    cm_new = confusion_matrix(Y, Y_pred)
    

    return ada_boost, cm_new, scaler, importances
    
def save_models_results(trained_model, scaler, importances_filtered, new_names):    
    
    with open(new_names['model'], 'wb') as file:
        
        pickle.dump(trained_model, file)
    
    with open(new_names['scaler'], 'wb') as file:
        
        pickle.dump(scaler, file)
    
    
    importances_filtered.to_csv(new_names['importances'])
    
    return True


def retrain_by_adaboost(data_sk, data_update, split, model_path, importances_path, scaler_path, new_names, cm_old):
    
    number_cv = 5
    
    data_update = concat_and_get_k_plus_one_data(data_sk, data_update)
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X_train, Y_train, X_test, Y_test = train_test_split(data_interactions, split)
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    X_train_scaled_smt, Y_train_smt, X_test_scaled, Y_test, scaler = get_all_sets_training(data_update, split)
    
    initial_weights = get_first_adaboost_weights(X_train_scaled_smt, Y_train_smt, model_path, importances_path, scaler_path)
    
    pipe = IMBPipeline([("smote", SMOTETomek()), ("randomforestclassifier", RandomForestClassifier())])
    
    params = { "randomforestclassifier__max_depth" : [2, 3, 4, 5, 6, 7], 
               'randomforestclassifier__min_samples_leaf' : [5, 10, 30, 50, 100], 
               "randomforestclassifier__bootstrap" : [True, False]}

    
    ini_time = time.time()
    
    grid = GridSearchCV(
        estimator = pipe,
        param_grid = params,
        scoring = 'f1_macro',
        cv = 5
        )
    
    grid.fit(X_train_scaled, Y_train)
    
    end_time_cv = time.time()
    
    computational_time_cv = end_time_cv - ini_time
    
    best_params_cv = grid.best_params_
    
    best_params_rf = {key.replace('randomforestclassifier__', ''): value for key, value in best_params_cv.items()}

    rf = RandomForestClassifier(**best_params_rf)
    
    ada_boost = AdaBoostClassifier(base_estimator = rf)
    
    t_ini_ada = time.time() 
    ada_boost.fit(X_train_scaled_smt, Y_train_smt, sample_weight = initial_weights)
    t_train_ada = time.time() - t_ini_ada
    
    Y_train_pred = ada_boost.predict(X_train_scaled)
    Y_test_pred = ada_boost.predict(X_test_scaled)
    
    
    computational_time = computational_time_cv + t_train_ada
    f1_cv = f1_score(y_true = Y_train, y_pred = Y_train_pred, average = "macro")
    f1_test = f1_score(y_true = Y_test, y_pred = Y_test_pred, average = "macro")
    
    ada_boost, cm_new, scaler, importances_filtered = get_objects_retraining_adaboost(best_params_rf, data_update, model_path, importances_path, scaler_path)
    
    save_models_results(ada_boost, scaler, importances_filtered, new_names)
    
 
    return computational_time, f1_cv, f1_test          
    

def retraining_each_step(data_sk, data_update, split, retraining_method, model_path, importances_path, scaler_path, new_names, cm_old):
    
    if retraining_method == "not retrain":
        
        computational_time, f1_cv, f1_test = retrain_not_really(data_update, model_path, importances_path, scaler_path)
        
        cm_updated = None
    
    elif retraining_method == "simple":
        
        computational_time, f1_cv, f1_test = retrain_simple(data_sk, data_update, split, new_names, cm_old)
        
    elif retraining_method == "adaboost":
        
        computational_time, f1_cv, f1_test = retrain_by_adaboost(data_sk, data_update, split, model_path, importances_path, scaler_path, new_names, cm_old)
        
    
    return computational_time, f1_cv, f1_test


def update_names(model_path, importances_path, scaler_path, retraining_type):
    
    model_list_split = model_path.split("/")
    model_name = model_list_split[-1]
    
    importances_list_split = importances_path.split("/")
    scaler_list_split = scaler_path.split("/")
    
    if model_name == 'name.pkl':
        
        if retraining_type == "simple":
            
            model_name_new = "name.pkl"
            
            model_path_new = "/".join(model_list_split[:-1])
            model_path_new = model_path_new + "/" + model_name_new
            
            importances_path_new = "/".join(importances_list_split[:-1])
            importances_path_new = importances_path_new + "/" + "name.csv"
            
            scaler_path_new = "/".join(scaler_list_split[:-1])
            scaler_path_new = scaler_path_new + "/" + "name.pkl"
            
            
        elif retraining_type == "adaboost":
            
            model_name_new = "name.pkl"
            
            model_path_new = "/".join(model_list_split[:-1])
            model_path_new = model_path_new + "/" + model_name_new
            
            importances_path_new = "/".join(importances_list_split[:-1])
            importances_path_new = importances_path_new + "/" + "name.csv"
            
            scaler_path_new = "/".join(scaler_list_split[:-1])
            scaler_path_new = scaler_path_new + "/" + "name_1.pkl"
    
    else:
        
        model_name_split = model_name.split("_")
        model_name_number = model_name_split[-1].split(".")
        model_name_number = int(model_name_number[0]) + 1
        
        model_name_new = model_name_split[0] + "_" + model_name_split[1] + "_" + str(model_name_number) + ".pkl"
        model_path_new = "/".join(model_list_split[:-1]) + "/" + model_name_new
        
        if retraining_type == "simple":
            
            importances_path_new = "/".join(importances_list_split[:-1])
            importances_path_new = importances_path_new + "/" + "importances_simple_" + str(model_name_number) + ".csv"
            
            scaler_path_new = "/".join(scaler_list_split[:-1])
            scaler_path_new = scaler_path_new + "/" + "scaler_simple_" + str(model_name_number) + ".pkl"
        
        elif retraining_type == "adaboost":
            
            importances_path_new = "/".join(importances_list_split[:-1])
            importances_path_new = importances_path_new + "/" + "importances_adaboost_" + str(model_name_number) + ".csv"
            
            scaler_path_new = "/".join(scaler_list_split[:-1])
            scaler_path_new = scaler_path_new + "/" + "scaler_adaboost_" + str(model_name_number) + ".pkl"
            
    
    return model_path_new, importances_path_new, scaler_path_new

def retraining_decider(data_update, model_path, importances_path, scaler_path, confusion_matrix_old, retraining_type):
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['---']] #Confidential variables selected
    X_num = X.drop(['---'], axis = 1) #Confidential variables dropped
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    X_scaled = X_num_scaled.join(X_cat)
    
    Y_pred = predict_by_old_model(X_scaled,  model_path, importances_path)

    f1_value_update = f1_score(y_true = Y, y_pred = Y_pred, average = "macro")
    
    cm_update = confusion_matrix(y_true = Y, y_pred = Y_pred, labels = [0, 1, 2])
    
    cm_new = cm_update + confusion_matrix_old

    precision = np.diag(cm_new) / np.sum(cm_new, axis = 0)
    recall = np.diag(cm_new) / np.sum(cm_new, axis = 1)
    f1_value_new = np.mean(np.nan_to_num(2 * (precision * recall) / (precision + recall)))
    
    if f1_value_new < 0.8:
        
        retrain = True
        model_path_new, importances_path_new, scaler_path_new = update_names(model_path, importances_path, scaler_path, retraining_type)
        
    else:
        
        retrain = False
        
        model_path_new = model_path
        importances_path_new = importances_path
        scaler_path_new = scaler_path
        
    return model_path_new, importances_path_new, scaler_path_new, retrain, cm_new, f1_value_new

def convert_to_datetime(datetime_value):
    try:
        
        return pd.to_datetime(datetime_value, unit = 's', errors = 'coerce')
    
    except ValueError:
        
        return None

def simulate_retraining_all_methods(data_sk_path, 
                        data_server_path,
                        model_path,
                        importances_path,
                        scaler_path,
                        split = 0.8):
    
    data_sk = pd.read_csv(data_sk_path)    
    data_update_ini = pd.read_csv(data_server_path)    
    
    retrainings_simple = []
    retrainings_adaboost = []
    
    data_sk = data_sk.reset_index(drop = True)
    data_update_ini = data_update_ini.reset_index(drop = True)
    
    data_sk.loc[:, 'datetime_ini'] = pd.to_datetime(data_sk['---'], unit = "s") #Confidential
    
    data_sk_adaboost = data_sk.copy()
    data_sk_simple = data_sk.copy()
    
    data_update_ini['datetime_temp'] = data_update_ini['---'].apply(convert_to_datetime) #Confidential
    data_update_ini.loc[data_update_ini['datetime_temp'].isnull(), 'datetime_temp'] = pd.to_datetime(data_update_ini.loc[data_update_ini['datetime_temp'].isnull(), 'EX2_restart_ini_time_OFF'].astype(str), format = '%Y%m%d%H%M%S', errors = 'coerce')
    data_update_ini['datetime_ini'] = data_update_ini['datetime_temp']
    del data_update_ini['datetime_temp']
    

    max_old_data_datetime = data_sk['datetime_ini'].max()
    data_update_ini = data_update_ini[data_update_ini['datetime_ini'] > max_old_data_datetime] 

    confusion_matrix_old = confusion_matrix_old_simple = confusion_matrix_old_adaboost = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    
    data_update_ini.sort_values('datetime_ini', inplace=True)
    unique_datetimes = data_update_ini['datetime_ini'].dt.date.unique()
    
    number_of_previous_obs = 0
    
    computational_cost_no_retrain = []
    f1_cv_no_retrain = []
    f1_test_no_retrain = []
    
    computational_cost_simple = []
    f1_cv_simple = []
    f1_test_simple = []
    
    computational_cost_adaboost = []
    f1_cv_adaboost = []
    f1_test_adaboost = []
    
    model_simple_path = model_adaboost_path = model_path
    importances_simple_path = importances_adaboost_path = importances_path
    scaler_simple_path = scaler_adaboost_path = scaler_path
    
    do_retraining_simple = do_retraining_adaboost = False
    
    datetime_4am_previous_simple = datetime_4am_previous_adaboost =  pd.Timestamp(year = unique_datetimes[0].year, month = unique_datetimes[0].month, day = unique_datetimes[0].day - 1, hour = 4)
    
    for i, date in enumerate(unique_datetimes):
        
        if i == 0:
            datetime_4am = pd.Timestamp(year = date.year, month = date.month, day = date.day, hour = 4)
        
            data_update = data_update_simple =  data_update_adaboost = data_update_ini[data_update_ini['datetime_ini'] < datetime_4am]
        
        else:
            
            datetime_4am = pd.Timestamp(year = date.year, month = date.month, day = date.day, hour = 4)
        
            data_update = data_update_ini[data_update_ini['datetime_ini'] < datetime_4am]
    
            data_update_simple = data_update_ini[(data_update_ini['datetime_ini'] >= datetime_4am_previous_simple) & (data_update_ini['datetime_ini'] < datetime_4am)]
            
            data_update_adaboost = data_update_ini[(data_update_ini['datetime_ini'] >= datetime_4am_previous_adaboost) & (data_update_ini['datetime_ini'] < datetime_4am)]

        if data_update_simple.shape[0]!= 0:
            
            model_path_new_simple, importances_path_new_simple, scaler_path_new_simple, do_retraining_simple, cm_simple_iter, f1_simple_iter = retraining_decider(data_update_simple, model_simple_path, importances_simple_path, scaler_simple_path, confusion_matrix_old_simple, "simple")

            retrainings_simple.append(do_retraining_simple)
            
            #retrain simple
            if do_retraining_simple:

                new_names_simple = {'model' : model_path_new_simple, 'importances' : importances_path_new_simple, 'scaler' : scaler_path_new_simple}
                    
                computational_time, f1_cv, f1_test = retraining_each_step(data_sk_simple, data_update_simple, split, 'simple', model_simple_path, importances_simple_path, scaler_simple_path, new_names_simple, confusion_matrix_old_simple)

                computational_cost_simple.append(computational_time)
                f1_cv_simple.append(f1_cv)
                f1_test_simple.append(f1_test)
                    
                model_simple_path = model_path_new_simple
                importances_simple_path = importances_path_new_simple
                scaler_simple_path = scaler_path_new_simple
                    
                confusion_matrix_old_simple = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    
                data_sk_simple =  pd.concat([data_sk_simple, data_update_simple])
                
                datetime_4am_previous_simple = datetime_4am
                    
                do_retraining_simple = False
                    
            else:
                    
                confusion_matrix_old_simple = cm_simple_iter
                computational_cost_simple.append(0)
                f1_cv_simple.append(f1_simple_iter)
                f1_test_simple.append(f1_simple_iter)
            
            
        if data_update_adaboost.shape[0]!= 0:
                        
            model_path_new_adaboost, importances_path_new_adaboost, scaler_path_new_adaboost, do_retraining_adaboost, cm_adaboost_iter, f1_adaboost_iter = retraining_decider(data_update_adaboost, model_adaboost_path, importances_adaboost_path, scaler_adaboost_path, confusion_matrix_old_adaboost, "adaboost")
            
            retrainings_adaboost.append(do_retraining_adaboost)
            
            #retrain adaboost
            if do_retraining_adaboost:

                new_names_adaboost = {'model' : model_path_new_adaboost, 'importances' : importances_path_new_adaboost, 'scaler' : scaler_path_new_adaboost}
                     
                computational_time, f1_cv, f1_test = retraining_each_step(data_sk_adaboost, data_update_adaboost, split, 'adaboost', model_adaboost_path, importances_adaboost_path, scaler_adaboost_path, new_names_adaboost, confusion_matrix_old_adaboost)

                computational_cost_adaboost.append(computational_time)
                f1_cv_adaboost.append(f1_cv)
                f1_test_adaboost.append(f1_test)
                     
                model_adaboost_path = model_path_new_adaboost
                importances_adaboost_path = importances_path_new_adaboost
                scaler_adaboost_path = scaler_path_new_adaboost
                     
                confusion_matrix_old_adaboost = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                     
                data_sk_adaboost =  pd.concat([data_sk_simple, data_update_simple])
                     
                datetime_4am_previous_adaboost = datetime_4am
                
                do_retraining_adaboost = False
                
            else:
                    
                confusion_matrix_old_adaboost = cm_adaboost_iter
                computational_cost_adaboost.append(0)
                f1_cv_adaboost.append(f1_adaboost_iter)
                f1_test_adaboost.append(f1_adaboost_iter)
                    
               
        if data_update.shape[0]!= 0:

            computational_time, f1_cv, f1_test = retrain_not_really(data_update, model_path, importances_path, scaler_path)      
            computational_cost_no_retrain.append(computational_time)
            f1_cv_no_retrain.append(f1_cv)
            f1_test_no_retrain.append(f1_test)
            
        no_retraining_results = {"computational cost" : computational_cost_no_retrain, "f1_cv" : f1_cv_no_retrain, "f1_test" : f1_test_no_retrain} 
        simple_results = {"computational cost" : computational_cost_simple, "f1_cv" : f1_cv_simple, "f1_test" : f1_test_simple, "retrainings" : retrainings_simple}
        adaboost_results = {"computational cost" : computational_cost_adaboost, "f1_cv" : f1_cv_adaboost, "f1_test" : f1_test_adaboost, "retrainings" : retrainings_adaboost}
         
         
        folder_path = "name"

        no_retraining_filename = folder_path + "/name.pkl"
        simple_filename =  folder_path + "/name.pkl"
        adaboost_filename =  folder_path + "/name.pkl"

        with open(no_retraining_filename, 'wb') as file:
            pickle.dump(no_retraining_results, file)

        with open(simple_filename, 'wb') as file:
            pickle.dump(simple_results, file)

        with open(adaboost_filename, 'wb') as file:
            pickle.dump(adaboost_results, file)    
           
    no_retraining_results = {"computational cost" : computational_cost_no_retrain, "f1_cv" : f1_cv_no_retrain, "f1_test" : f1_test_no_retrain} 
    simple_results = {"computational cost" : computational_cost_simple, "f1_cv" : f1_cv_simple, "f1_test" : f1_test_simple, "retrainings" : retrainings_simple}
    adaboost_results = {"computational cost" : computational_cost_adaboost, "f1_cv" : f1_cv_adaboost, "f1_test" : f1_test_adaboost, "retrainings" : retrainings_adaboost}
    
    return no_retraining_results, simple_results, adaboost_results
            
            
no_retraining_results, simple_results, adaboost_results = simulate_retraining_all_methods()       
            
            
folder_path = "name"

no_retraining_filename = folder_path + "/name.pkl"
simple_filename =  folder_path + "/name.pkl"
adaboost_filename =  folder_path + "/name.pkl"

with open(no_retraining_filename, 'wb') as file:
    pickle.dump(no_retraining_results, file)

with open(simple_filename, 'wb') as file:
    pickle.dump(simple_results, file)

with open(adaboost_filename, 'wb') as file:
    pickle.dump(adaboost_results, file)            
            
            
            
            





