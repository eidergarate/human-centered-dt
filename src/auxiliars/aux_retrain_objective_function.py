import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, r2_score
from scipy import stats
import xgboost as xgb
import time


def compute_kendall_correlation(y, y_pred):
    
    score = stats.kendalltau(np.exp(y), np.exp(y_pred))
    
    return score.correlation
kendall_score = make_scorer(compute_kendall_correlation, greater_is_better = True, needs_proba = False, needs_threshold = False)


def convert_to_datetime(datetime_value):
    try:
        
        return pd.to_datetime(datetime_value, unit = 's', errors = 'coerce')
    
    except ValueError:
        
        return None

def filter_by_date(filtered_data):
    
    filtered_data['Year'] = filtered_data['datetime_ini'].dt.year
    filtered_data['Month'] = filtered_data['datetime_ini'].dt.month
    filtered_data['Day'] = filtered_data['datetime_ini'].dt.day

    filtered_data = filtered_data[(filtered_data['datetime_ini'] >= ini_analyze]

    filtered_data = filtered_data.drop(columns=['Year', 'Month', 'Day'])
    
    return filtered_data

def filter_data(data_update): #Confidential

    
    return filtered_data

def select_vars(filtered_data): #Confidential
    
    selected_data = filtered_data[["---"]]
    
    return selected_data


def filter_select_vars(data_update):
    
    filtered_data = filter_data(data_update)
    
    filtered_and_selected_data = select_vars(filtered_data)
    
    filtered_and_selected_data = filtered_and_selected_data.dropna()
    
    return filtered_and_selected_data

def get_interactions(filtered_and_selected_data): #Confidential
  
    data_interactions = filtered_and_selected_data[["---"]]
    
    return data_interactions

def split_X_Y(data_interactions):
    
    X = data_interactions.drop('quality', axis = 1)
    Y = data_interactions['quality']
   
    return X, Y

def predict_by_old_model(X_scaled,  model_path):
  
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    Y_pred = model.predict(X_scaled)
    
    return Y_pred

def load_model_and_scaler(model_path, scaler_path):
    
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    return model, scaler
    
def change_datetimes(data_sk, data_update): #Confidential
    
    return data_sk, data_update
    
def get_most_used_recipes_as_dummies(X_scaled, recipes_retrain_path):
    
    with open(recipes_retrain_path, 'rb') as file:
              allowed_recipes = pickle.load(file)
    
    X_scaled['recipe'] = X_scaled['recipe'].apply(lambda x: x if x in allowed_recipes else 'others')
    
    X_scaled['recipe'] = pd.Categorical(X_scaled['recipe'], categories = allowed_recipes)

    dummy_df = pd.get_dummies(X_scaled['recipe'], prefix = 'recipe')

    X_scaled_temp = pd.concat([X_scaled.drop(['recipe', '---'], axis = 1), dummy_df], axis = 1) #Confidential variables dropped

    X_scaled_temp = pd.concat([X_scaled_temp, X_scaled[['---']]], axis = 1) #Confidential variables selected

    X_scaled_temp.drop('recipe', axis = 1, inplace = True)
    
    return X_scaled_temp

def update_names(model_path, scaler_path, recipes_retrain_path):
    
    
    model_list_split = model_path.split("/")
    model_name = model_list_split[-1]
   
    scaler_list_split = scaler_path.split("/")
    recipes_list_split = recipes_retrain_path.split("/")
    
   
    if model_name == 'M2_log_kendall.pkl':

        model_name_new = "M2_log_kendall_1.pkl"
           
        model_path_new = "/".join(model_list_split[:-1])
        model_path_new = model_path_new + "/" + model_name_new
            
           
        scaler_path_new = "/".join(scaler_list_split[:-1])
        scaler_path_new = scaler_path_new + "/" + "scaler_M2_log_kendall_1.pkl"
        
        recipes_path_new = "/".join(recipes_list_split[:-1])
        recipes_path_new = recipes_path_new + "/" + "recipes_M2_log_kendall_1.pkl"
    
    else:
        
        model_name_split = model_name.split("_")
        model_name_number = model_name_split[-1].split(".")
        model_name_number = int(model_name_number[0]) + 1
       
        model_name_new = model_name_split[0] + "_" + model_name_split[1] + "_" + str(model_name_number) + ".pkl"
        model_path_new = "/".join(model_list_split[:-1]) + "/" + model_name_new
       
           
           
        scaler_path_new = "/".join(scaler_list_split[:-1])
        scaler_path_new = scaler_path_new + "/" + "scaler_M2_log_kendall_" + str(model_name_number) + ".pkl"
        
        recipes_path_new = "/".join(recipes_list_split[:-1])
        recipes_path_new = recipes_path_new + "/" + "recipes_M2_log_kendall_" + str(model_name_number) + ".pkl"
        
    return model_path_new, scaler_path_new, recipes_path_new
       
def scale_data(X):

    scaler = StandardScaler()
    
    X_cat = X[['recipe', '---']] #Confidential variables selected
    X_num = X.drop(['recipe', '---'], axis = 1) #Confidential variables dropped
    
    scaler.fit(X_num)
    
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    X_scaled = X_num_scaled.join(X_cat)
    
    return X_scaled, scaler
      
def select_most_important_recipes(X_recipes, recipes_new_path): #Confidential
    
    return True
        

def retraining_decider(data_update_retrain, model_retrain_path, recipes_retrain_path, scaler_retrain_path, kendall_retrain_old, Y_old, Y_old_pred):
                                              
    filtered_and_selected_data = filter_select_vars(data_update_retrain)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['EX_INF_recipe_ON', '---']] #Confidential variables selected
    X_num = X.drop(['EX_INF_recipe_ON', '---'], axis = 1) #Confindential variables dropped
    
    with open(scaler_retrain_path, 'rb') as file:
        scaler = pickle.load(file)
    
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    X_scaled = X_num_scaled.join(X_cat)
    
    X_scaled_dummy_recipes = get_most_used_recipes_as_dummies(X_scaled, recipes_retrain_path)
    
    Y_pred = predict_by_old_model(X_scaled_dummy_recipes,  model_retrain_path)
    
    if Y_old is not None and not Y_old.empty:
        if isinstance(Y_old_pred, np.ndarray):
            Y_old_pred = pd.Series(Y_old_pred)
        if isinstance(Y_pred, np.ndarray):
            Y_pred = pd.Series(Y_pred)
        Y = pd.concat([Y_old, Y])
        Y_pred = pd.concat([Y_old_pred, Y_pred])
    
    kendall_value_update = compute_kendall_correlation(y = np.log(Y), y_pred = Y_pred)
    
    if kendall_value_update < 0.6:
        
        retrain = True
        model_path_new, scaler_path_new, recipes_path_new = update_names(model_retrain_path, scaler_retrain_path, recipes_retrain_path)
        
    else:
        
        retrain = False
        
        model_path_new = model_retrain_path
        recipes_path_new = recipes_retrain_path
        scaler_path_new = scaler_retrain_path              
        
              

    return model_path_new, recipes_path_new, scaler_path_new, retrain, kendall_value_update, Y, Y_pred

def concat_and_get_k_plus_one_data(data_sk, data_update):

    data_k_plus_one = pd.concat([data_sk, data_update], axis = 0)    
    
    data_k_plus_one = data_k_plus_one.reset_index(drop = True)
    
    return data_k_plus_one

def train_test_split(X,  Y, split):
    
    ntrain = int(X.shape[0]*split)
    
    X_train = X.sample(n = ntrain, replace = False, random_state = 42)
    X_train_idx = X_train.index
    
    X_test_idx = X.index.difference(X_train_idx)
    X_test = X.loc[X_test_idx, :]
    
    Y_train = Y.loc[X_train_idx]
    Y_test = Y.loc[X_test_idx]
    
    return  X_train, X_test, Y_train, Y_test
    
def get_objects_retraining_simple(xgb_params, X_scaled_dummy_recipes, Y_log):
    
    xgb_best = xgb.XGBRegressor(**xgb_params)
    
    xgb_best.fit(X_scaled_dummy_recipes, Y_log)
    
    return xgb_best

def save_models_results(xgb_model, scaler, new_names):
    
    with open(new_names['model'], 'wb') as file:
        pickle.dump(xgb_model, file)
    
    with open(new_names['scaler'], 'wb') as file:
        pickle.dump(scaler, file)
    
    return True
    
    

def retraining_each_step(data_sk_retrain, data_update_retrain, split, new_names):
    
    data_update = concat_and_get_k_plus_one_data(data_sk_retrain, data_update_retrain)
    
    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_scaled, scaler = scale_data(X)
    
    select_most_important_recipes(X_scaled['recipe'], new_names['recipes'])
    
    X_scaled_dummy_recipes = get_most_used_recipes_as_dummies(X_scaled, new_names['recipes'])
    
    Y_log = np.log(Y)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled_dummy_recipes,  Y_log, split)
    
    pipe = make_pipeline(xgb.XGBRegressor())
    params = [{'xgbregressor__n_estimators' : [5, 10, 30, 20, 50, 100, 150], 'xgbregressor__max_depth' : [2, 4, 3, 5],
               'xgbregressor__gamma' : [0, 0.3, 0.4, 0.7, 2, 5, 8], 'xgbregressor__eta' : [0.1, 0.2, 0.3, 0.5, 0.7], 'xgbregressor__min_child_weight' : [0.4, 0.7, 1, 2, 4],
               'xgbregressor__sampling_method' : ['uniform'], 'xgbregressor__subsample' : [0.1, 0.4, 0.5, 0.7, 1]}]
    
    ini_time = time.time()
    
    grid = RandomizedSearchCV(
        estimator = pipe,
        param_distributions = params,
        scoring = kendall_score, #meter aqui el scorer
        cv = 5,
        n_iter = 40)
    
    grid.fit(X_train, Y_train)
    
    end_time = time.time()
    
    computational_time = end_time - ini_time
    
    kendall_cv = grid.best_score_
    
    Y_test_pred = grid.predict(X_test)
    
    kendall_test = compute_kendall_correlation(y = Y_test, y_pred = Y_test_pred)
    
    best_params = grid.best_params_
    
    xgb_params = {keys.replace('xgbregressor__', ''): values for keys, values in best_params.items()}
    
    xgb_model = get_objects_retraining_simple(xgb_params, X_scaled_dummy_recipes,  Y_log)
    
    save_models_results(xgb_model, scaler, new_names)
    
    return computational_time, kendall_cv, kendall_test
    
def retrain_not_really(data_update, model_path, recipes_path, scaler_path, Y_old, Y_old_pred):

    filtered_and_selected_data = filter_select_vars(data_update)
    
    data_interactions = get_interactions(filtered_and_selected_data)
    
    X, Y = split_X_Y(data_interactions)
    
    X_cat = X[['recipe', '---']] #Selecting confidential variables
    X_num = X.drop(['recipe', '---'], axis = 1) #Dropping confidential variables
    
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
    
    X_num_scaled = pd.DataFrame(scaler.transform(X_num), index = X_num.index, columns = X_num.columns)
    
    X_scaled = X_num_scaled.join(X_cat)
    
    X_scaled_dummy_recipes = get_most_used_recipes_as_dummies(X_scaled, recipes_path)
    
    t0 = time.time()
    Y_pred = predict_by_old_model(X_scaled_dummy_recipes,  model_path)
    tf = time.time() - t0
    if Y_old is not None and not Y_old.empty:
        if isinstance(Y_old_pred, np.ndarray):
            Y_old_pred = pd.Series(Y_old_pred)
        if isinstance(Y_pred, np.ndarray):
            Y_pred = pd.Series(Y_pred)
        Y = pd.concat([Y_old, Y])
        Y_pred = pd.concat([Y_old_pred, Y_pred])
        
    
    kendall = compute_kendall_correlation(y = np.log(Y), y_pred = Y_pred)
    
    return tf, kendall, kendall, Y, Y_pred
    
def get_dtypes_olds_data(): #Confidential

    return data_types_old_data    

def get_dtypes_server_data(): #Confidential

    return data_types_data  

def simulate_retraining_M2(model_ini_path, 
                          scaler_ini_path, 
                          recipes_ini_path, 
                          data_sk_path,
                          data_server_path,
                          r2_score_ini = 0.47,
                          split = 0.8):
                            
    dtypes_old = get_dtypes_olds_data()
    dtypes_update = get_dtypes_server_data()
    
    data_sk = pd.read_csv(data_sk_path, dtype = dtypes_old)
    data_update_ini = pd.read_csv(data_server_path, dtype = dtypes_update)
    
    retrainings = []
    
    data_sk = data_sk.reset_index(drop = True)
    data_update_ini = data_update_ini.reset_index(drop = True)
    
    data_update_ini = data_update_ini[data_update_ini['quality_status'].isin([0, 1])]
    data_sk = data_sk[(data_sk['---'] == "---") & (data_sk['quality_status'].isin([0, 1]))] #There is a confidential filter

    
    
    data_sk.loc[:, 'datetime_ini'] = pd.to_datetime(data_sk['---'], unit = "s") #Confidential variable
    data_sk = filter_by_date(data_sk)
    
    data_update_ini['datetime_temp'] = data_update_ini['---'].apply(convert_to_datetime) #Confidential variable
    data_update_ini.loc[data_update_ini['datetime_temp'].isnull(), 'datetime_temp'] = pd.to_datetime(data_update_ini.loc[data_update_ini['datetime_temp'].isnull(), 'EX2_restart_ini_time_OFF'].astype(str), format = '%Y%m%d%H%M%S', errors = 'coerce')
    data_update_ini['datetime_ini'] = data_update_ini['datetime_temp']
    del data_update_ini['datetime_temp']
    

    max_old_data_datetime = data_sk['datetime_ini'].max()
    data_update_ini = data_update_ini[data_update_ini['datetime_ini'] > max_old_data_datetime]
    
    data_update_ini.sort_values('datetime_ini', inplace = True)
    unique_datetimes = data_update_ini['datetime_ini'].dt.date.unique()
    
    
    number_of_previous_obs = 0
    
    computational_cost_no_retrain = []
    kendall_cv_no_retrain = []
    kendall_test_no_retrain = []
    
    computational_cost_retrain = []
    kendall_cv_retrain = []
    kendall_test_retrain = []
    
    model_retrain_path = model_ini_path
    scaler_retrain_path = scaler_ini_path
    recipes_retrain_path = recipes_ini_path
    
    do_retraining = False
    
    datetime_4am_previous_retrain = pd.Timestamp(year = unique_datetimes[0].year, month = unique_datetimes[0].month, day = unique_datetimes[0].day - 1, hour = 4)
    
    kendall_retrain_old = None
    Y_retrain_old_pred = Y_old_pred = Y_retrain_old = Y_old = None
    
    data_sk_retrain = data_sk.copy()
    
    for i, date in enumerate(unique_datetimes):
        
        if i == 0:
            datetime_4am = pd.Timestamp(year = date.year, month = date.month, day = date.day, hour = 4)
        
            data_update = data_update_retrain = data_update_ini[data_update_ini['datetime_ini'] < datetime_4am]
        
        else:
            
            datetime_4am = pd.Timestamp(year = date.year, month = date.month, day = date.day, hour = 4)
        
            data_update = data_update_ini[data_update_ini['datetime_ini'] < datetime_4am]
    
            data_update_retrain = data_update_ini[(data_update_ini['datetime_ini'] >= datetime_4am_previous_retrain) & (data_update_ini['datetime_ini'] < datetime_4am)]
            
        if data_update_retrain.shape[0]!= 0:
            
            model_retrain_path_new, recipes_retrain_path_new, scaler_retrain_path_new, do_retraining, kendall_retrain_iter, Y_retrain_iter, Y_retrain_pred_iter = retraining_decider(data_update_retrain, model_retrain_path, recipes_retrain_path, scaler_retrain_path, kendall_retrain_old, Y_retrain_old, Y_retrain_old_pred)

            retrainings.append(do_retraining)
              
            if do_retraining:
            
                new_names = {'model' : model_retrain_path_new, 'recipes' : recipes_retrain_path_new, 'scaler' : scaler_retrain_path_new}
                    
                computational_time, kendall_cv, kendall_test = retraining_each_step(data_sk_retrain, data_update_retrain, split, new_names)

                computational_cost_retrain.append(computational_time)
                kendall_cv_retrain.append(kendall_cv)
                kendall_test_retrain.append(kendall_test)
                    
                model_retrain_path = model_retrain_path_new
                recipes_retrain_path = recipes_retrain_path_new
                scaler_retrain_path = scaler_retrain_path_new
                    
                Y_retrain_old = None
                    
                data_sk_retrain =  pd.concat([data_sk_retrain, data_update_retrain])
                
                datetime_4am_previous_retrain = datetime_4am
                    
                do_retraining = False
            
            else:
            
                computational_cost_retrain.append(0)
                kendall_cv_retrain.append(kendall_retrain_iter)
                kendall_test_retrain.append(kendall_retrain_iter)
            
                Y_retrain_old = Y_retrain_iter
                Y_retrain_old_pred = Y_retrain_pred_iter
    
        if data_update.shape[0]!= 0:

            computational_time, kendall_cv, kendall_test, Y_new, Y_new_pred = retrain_not_really(data_update, model_ini_path, recipes_ini_path, scaler_ini_path, Y_old, Y_old_pred)      
            computational_cost_no_retrain.append(computational_time)
            kendall_cv_no_retrain.append(kendall_cv)
            kendall_test_no_retrain.append(kendall_test)
    
            Y_old = Y_new
            Y_old_pred = Y_new_pred
    
            
            no_retraining_results = {"computational cost" : computational_cost_no_retrain, "kendall_cv" : kendall_cv_no_retrain, "kendall_test" : kendall_test_no_retrain} 
            retraining_results = {"computational cost" : computational_cost_retrain, "kendall_cv" : kendall_cv_retrain, "kendall_test" : kendall_test_retrain, "retrainings" : retrainings}

         
            folder_path = "your_folder"

            no_retraining_filename = folder_path + "/name.pkl"
            simple_filename =  folder_path + "/name.pkl"

            with open(no_retraining_filename, 'wb') as file:
                pickle.dump(no_retraining_results, file)

            with open(simple_filename, 'wb') as file:
                pickle.dump(retraining_results, file)
                
    return no_retraining_results, retraining_results
  
