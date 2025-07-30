library(DBI)
library(dplyr)
library(reticulate)
library(RMariaDB)
source_python(here::here("src", "auxiliars", "retrain_and_predict_M1.py"), convert = T, envir = globalenv())
source(here::here("src", "auxiliars", "main_scale_data.R"))
source(here::here("src", "auxiliars", "aux_connect_historicDB.R"))
source(here::here("src", "auxiliars", "get_interactions.R"))
source(here::here("src", "auxiliars", "aux_input_NAs_production.R"))
source(here::here("src", "auxiliars", "aux_version_names.R"))

compute_prediction_errors <- function(preprocessed_data, feedback_table, data_path_scaler, model_path, importances_path){

  Y_real <- preprocessed_data %>% select(quality_status)
  
  unique_ID <- preprocessed_data %>% select(id)
  
  print("Scaling data")
  tryCatch({
    preprocessed_data %<>% main_scale_data(., data_path_scaler)},
    error = function(e){
      print(e)
      print("ERROR: Problem scaling data when getting last day's predictions.")
    }
  )
  
  print("Predicting readiness")
  tryCatch({
    pred <- predict_M1(preprocessed_data, model_path, importances_path)},
    error = function(e){
      print(e)
      print("ERROR: Error predicting last day's steadiness.") 
    }
  )
  print("Getting errors between real and predicted readiness")
  preprocessed_data %<>% mutate(
    EX_DS_Hot_prof_status = Y_real$EX_DS_Hot_prof_status,
    M1_pred = pred,
    M1_error = abs(EX_DS_Hot_prof_status - M1_pred),
    unique_ID = unique_ID$id
  )
  
  model_version <- substr(model_path, 41, 70)
  
  preprocessed_data$model_version <- rep(model_version, nrow(preprocessed_data))

  preprocessed_data %>% select(unique_ID, EX_DS_Hot_prof_status, M1_pred, M1_error, model_version) -> errors_df
  
  return(errors_df)
}

update_error_table <- function(feedback_table_name, importances_path, 
                               data_path_scaler, model_path,
                               preprocessed_data){
  
  feedback_table <- NULL
  print("Getting interactions and dropping NA values")
  tryCatch({
    preprocessed_data %<>% get_interactions_production(.) %>% input_NAs_df(.)},
    error = function(e){
      print(e)
      print("ERROR: Problem when getting interactions when updating last day's M1 model's errors.")
    }
  )
  
  tryCatch({
    erros_df <- compute_prediction_errors(preprocessed_data, feedback_table, data_path_scaler, model_path, importances_path)},
    error = function(e){
      print(e)
      print("ERROR: Error while computing last day's predictions errors of M1 model.")
    }
  )
  
  #connect to historicDB
  tryCatch({
    con_hs <- connect_to_historicDB()},
    error = function(e){
      print(e)
      print("ERROR: Error while connection to historical SQLite DB to update errors table.")
    }
  )
  
  errors_table_name <- Sys.getenv("table_errors")
  
  tryCatch({
    if(!RMariaDB::dbExistsTable(con_hs, errors_table_name)){
      
      RMariaDB::dbWriteTable(con_hs, errors_table_name, erros_df)
      
    }else{
      
      RMariaDB::dbAppendTable(con_hs, errors_table_name, erros_df)
      
    }},
    error = function(e){
      print(e)
      print("ERROR: Error updating errors table with the last day's prediction errors.")
    } 
  )
  
  tryCatch({
    disconnect_from_historicDB(con_hs)},
    error = function(e){
      print(e)
      print("ERROR: Error while disconnecting historical SQLite DB.")
    }
  )
  
}

